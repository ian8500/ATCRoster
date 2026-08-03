"""Effective-dated working patterns and staff eligibility rules.

The service is deliberately independent of Flask routes. Existing CSV patterns
remain the compatibility fallback when ``resolve_pattern_day`` returns ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Any, Callable, Iterable


PATTERN_DAY_TYPES = frozenset({
    "FIXED_SHIFT", "WORK_ANY", "WORK_ALLOWED_SET", "OFF",
    "OPTIONAL_WORK", "PROTECTED_NON_OPERATIONAL",
})
STAFF_RULE_TYPES = frozenset({
    "NO_NIGHT", "AVOID_NIGHT", "NO_EARLY", "AVOID_EARLY",
    "ALLOWED_SHIFT", "DISALLOWED_SHIFT", "MAX_NIGHTS_PER_CYCLE",
    "MAX_SHIFTS_PER_CYCLE", "AVAILABLE_WEEKDAYS",
    "UNAVAILABLE_WEEKDAYS", "MAX_CONTRACTED_MINUTES", "PREFERRED_SHIFT",
    "PREFERRED_DAY_OFF",
})


@dataclass(frozen=True)
class PatternResolution:
    assignment: Any
    pattern: Any
    cycle_index: int
    pattern_day: Any
    allowed_shift_type_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EligibilityReason:
    code: str
    explanation: str
    rule_id: int | None = None


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason_code: str
    explanation: str
    reasons: tuple[EligibilityReason, ...] = ()
    soft_penalty: int = 0


@dataclass(frozen=True)
class EligibilityContext:
    """Preloaded data used when validating many assignments at once."""

    staff_by_id: dict[int, Any]
    shift_by_id: dict[int, Any]
    leaves_by_staff: dict[int, tuple[Any, ...]]
    pattern_assignments_by_staff: dict[int, tuple[Any, ...]]
    patterns_by_id: dict[int, Any]
    pattern_days_by_key: dict[tuple[int, int], Any]
    allowed_shift_ids_by_day: dict[int, frozenset[int]]
    rules_by_staff: dict[int, tuple[Any, ...]]
    assignments_by_staff: dict[int, tuple[Any, ...]]
    shifts_by_code: dict[str, Any]


@dataclass(frozen=True)
class WorkPatternDependencies:
    Staff: Any
    ShiftType: Any
    Leave: Any
    Assignment: Any
    WorkPattern: Any
    WorkPatternDay: Any
    WorkPatternDayAllowedShift: Any
    StaffPatternAssignment: Any
    StaffRule: Any
    shift_group: Callable[[Any], str]
    early_start_before: time = time(6, 30)


class WorkPatternService:
    def __init__(self, dependencies: WorkPatternDependencies) -> None:
        self.dependencies = dependencies

    def validate_pattern(self, pattern: Any, days: Iterable[Any]) -> None:
        if int(pattern.cycle_length_days or 0) <= 0:
            raise ValueError("Pattern cycle length must be greater than zero.")
        if int(pattern.contracted_minutes_per_cycle or 0) < 0:
            raise ValueError("Contracted cycle minutes cannot be negative.")
        day_rows = list(days)
        indexes = {int(row.day_index) for row in day_rows}
        expected = set(range(int(pattern.cycle_length_days)))
        if indexes != expected or len(day_rows) != len(expected):
            raise ValueError("A pattern must define every cycle day exactly once.")
        for row in day_rows:
            if row.day_type not in PATTERN_DAY_TYPES:
                raise ValueError(f"Unsupported pattern day type: {row.day_type}.")
            if row.day_type == "FIXED_SHIFT" and not row.fixed_shift_type_id:
                raise ValueError("A fixed-shift day must reference a shift type.")
            if row.day_type != "FIXED_SHIFT" and row.fixed_shift_type_id:
                raise ValueError("Only a fixed-shift day may reference a fixed shift.")

    def validate_staff_pattern_assignment(self, candidate: Any) -> None:
        pattern = self.dependencies.WorkPattern.query.filter_by(
            id=candidate.work_pattern_id,
            unit_id=candidate.unit_id,
        ).first()
        if not pattern:
            raise ValueError("The selected pattern is unavailable in this airport.")
        if not 0 <= int(candidate.anchor_day_index) < int(pattern.cycle_length_days):
            raise ValueError("Anchor cycle day falls outside the selected pattern.")
        if candidate.effective_to and candidate.effective_to < candidate.effective_from:
            raise ValueError("Pattern end date cannot be before its start date.")
        query = self.dependencies.StaffPatternAssignment.query.filter_by(
            unit_id=candidate.unit_id, staff_id=candidate.staff_id
        )
        if getattr(candidate, "id", None):
            query = query.filter(
                self.dependencies.StaffPatternAssignment.id != candidate.id
            )
        for existing in query.all():
            if _date_ranges_overlap(
                candidate.effective_from,
                candidate.effective_to,
                existing.effective_from,
                existing.effective_to,
            ):
                raise ValueError(
                    "This pattern assignment overlaps an existing effective period."
                )

    def validate_staff_rule(self, rule: Any) -> None:
        if rule.rule_type not in STAFF_RULE_TYPES:
            raise ValueError("Choose a supported staff rule type.")
        if rule.hardness not in {"HARD", "SOFT"}:
            raise ValueError("Rule hardness must be HARD or SOFT.")
        hard_only = {
            "NO_NIGHT", "NO_EARLY", "ALLOWED_SHIFT", "DISALLOWED_SHIFT",
            "MAX_NIGHTS_PER_CYCLE", "MAX_SHIFTS_PER_CYCLE",
            "AVAILABLE_WEEKDAYS", "UNAVAILABLE_WEEKDAYS",
            "MAX_CONTRACTED_MINUTES",
        }
        soft_only = {
            "AVOID_NIGHT", "AVOID_EARLY", "PREFERRED_SHIFT",
            "PREFERRED_DAY_OFF",
        }
        if rule.rule_type in hard_only and rule.hardness != "HARD":
            raise ValueError("This restriction must be configured as a hard rule.")
        if rule.rule_type in soft_only and rule.hardness != "SOFT":
            raise ValueError("This preference must be configured as a soft rule.")
        if rule.effective_to and rule.effective_to < rule.effective_from:
            raise ValueError("Rule end date cannot be before its start date.")
        if int(rule.penalty_weight or 0) < 0:
            raise ValueError("Rule penalty cannot be negative.")
        if rule.rule_type in {
            "ALLOWED_SHIFT", "DISALLOWED_SHIFT", "PREFERRED_SHIFT",
        } and not (rule.shift_type_id or rule.shift_group):
            raise ValueError("This rule must target a shift or shift group.")
        if rule.rule_type in {
            "AVAILABLE_WEEKDAYS", "UNAVAILABLE_WEEKDAYS", "PREFERRED_DAY_OFF",
        } and rule.weekdays_mask is None:
            raise ValueError("This rule must select at least one weekday.")
        if rule.weekdays_mask is not None and not 0 <= int(rule.weekdays_mask) <= 127:
            raise ValueError("Weekday selection is invalid.")
        if rule.rule_type in {
            "MAX_NIGHTS_PER_CYCLE", "MAX_SHIFTS_PER_CYCLE",
            "MAX_CONTRACTED_MINUTES",
        }:
            if rule.maximum_count is None or int(rule.maximum_count) < 0:
                raise ValueError("This rule requires a non-negative maximum.")
            if not rule.rolling_period_days or int(rule.rolling_period_days) <= 0:
                raise ValueError("This rule requires a positive rolling period.")

    def get_pattern_day_for_staff(
        self, staff_id: int, on_date: date
    ) -> PatternResolution | None:
        staff = self.dependencies.Staff.query.filter_by(id=staff_id).first()
        if not staff:
            return None
        assignment = (
            self.dependencies.StaffPatternAssignment.query.filter(
                self.dependencies.StaffPatternAssignment.unit_id == staff.unit_id,
                self.dependencies.StaffPatternAssignment.staff_id == staff.id,
                self.dependencies.StaffPatternAssignment.effective_from <= on_date,
                (
                    self.dependencies.StaffPatternAssignment.effective_to.is_(None)
                    | (self.dependencies.StaffPatternAssignment.effective_to >= on_date)
                ),
            )
            .order_by(
                self.dependencies.StaffPatternAssignment.effective_from.desc(),
                self.dependencies.StaffPatternAssignment.id.desc(),
            )
            .first()
        )
        if not assignment:
            return None
        pattern = self.dependencies.WorkPattern.query.filter_by(
            id=assignment.work_pattern_id,
            unit_id=staff.unit_id,
        ).first()
        if not pattern or int(pattern.cycle_length_days or 0) <= 0:
            return None
        cycle_index = (
            int(assignment.anchor_day_index)
            + (on_date - assignment.anchor_date).days
        ) % int(pattern.cycle_length_days)
        pattern_day = self.dependencies.WorkPatternDay.query.filter_by(
            unit_id=staff.unit_id,
            work_pattern_id=pattern.id,
            day_index=cycle_index,
        ).first()
        if not pattern_day:
            return None
        allowed = frozenset(
            int(row.shift_type_id)
            for row in self.dependencies.WorkPatternDayAllowedShift.query.filter_by(
                unit_id=staff.unit_id, work_pattern_day_id=pattern_day.id
            ).all()
        )
        return PatternResolution(
            assignment=assignment,
            pattern=pattern,
            cycle_index=cycle_index,
            pattern_day=pattern_day,
            allowed_shift_type_ids=allowed,
        )

    def get_effective_staff_rules(self, staff_id: int, on_date: date) -> tuple[Any, ...]:
        return tuple(
            self.dependencies.StaffRule.query.filter(
                self.dependencies.StaffRule.staff_id == staff_id,
                self.dependencies.StaffRule.is_active.is_(True),
                self.dependencies.StaffRule.effective_from <= on_date,
                (
                    self.dependencies.StaffRule.effective_to.is_(None)
                    | (self.dependencies.StaffRule.effective_to >= on_date)
                ),
            ).order_by(self.dependencies.StaffRule.id).all()
        )

    def build_eligibility_context(
        self, unit_id: int, staff_ids: Iterable[int], start: date, end: date
    ) -> EligibilityContext:
        """Load eligibility inputs in a fixed number of database queries.

        Interactive single-cell validation still uses the direct lookup path.
        Month-wide reporting and roster validation should build this context
        once and reuse it for every populated cell.
        """
        ids = tuple(sorted({int(staff_id) for staff_id in staff_ids}))
        if not ids:
            return EligibilityContext({}, {}, {}, {}, {}, {}, {}, {}, {}, {})

        staff_rows = self.dependencies.Staff.query.filter(
            self.dependencies.Staff.unit_id == unit_id,
            self.dependencies.Staff.id.in_(ids),
        ).all()
        shift_rows = self.dependencies.ShiftType.query.filter_by(
            unit_id=unit_id
        ).all()
        leave_rows = self.dependencies.Leave.query.filter(
            self.dependencies.Leave.unit_id == unit_id,
            self.dependencies.Leave.staff_id.in_(ids),
            self.dependencies.Leave.start <= end,
            self.dependencies.Leave.end >= start,
        ).all()
        pattern_assignment_rows = self.dependencies.StaffPatternAssignment.query.filter(
            self.dependencies.StaffPatternAssignment.unit_id == unit_id,
            self.dependencies.StaffPatternAssignment.staff_id.in_(ids),
            self.dependencies.StaffPatternAssignment.effective_from <= end,
            (
                self.dependencies.StaffPatternAssignment.effective_to.is_(None)
                | (self.dependencies.StaffPatternAssignment.effective_to >= start)
            ),
        ).order_by(
            self.dependencies.StaffPatternAssignment.effective_from.desc(),
            self.dependencies.StaffPatternAssignment.id.desc(),
        ).all()
        pattern_ids = {
            row.work_pattern_id for row in pattern_assignment_rows
        }
        pattern_rows = (
            self.dependencies.WorkPattern.query.filter(
                self.dependencies.WorkPattern.unit_id == unit_id,
                self.dependencies.WorkPattern.id.in_(pattern_ids),
            ).all()
            if pattern_ids else []
        )
        pattern_day_rows = (
            self.dependencies.WorkPatternDay.query.filter(
                self.dependencies.WorkPatternDay.unit_id == unit_id,
                self.dependencies.WorkPatternDay.work_pattern_id.in_(pattern_ids),
            ).all()
            if pattern_ids else []
        )
        pattern_day_ids = {row.id for row in pattern_day_rows}
        allowed_rows = (
            self.dependencies.WorkPatternDayAllowedShift.query.filter(
                self.dependencies.WorkPatternDayAllowedShift.unit_id == unit_id,
                self.dependencies.WorkPatternDayAllowedShift.work_pattern_day_id.in_(
                    pattern_day_ids
                ),
            ).all()
            if pattern_day_ids else []
        )
        rule_rows = self.dependencies.StaffRule.query.filter(
            self.dependencies.StaffRule.unit_id == unit_id,
            self.dependencies.StaffRule.staff_id.in_(ids),
            self.dependencies.StaffRule.is_active.is_(True),
            self.dependencies.StaffRule.effective_from <= end,
            (
                self.dependencies.StaffRule.effective_to.is_(None)
                | (self.dependencies.StaffRule.effective_to >= start)
            ),
        ).order_by(self.dependencies.StaffRule.id).all()
        maximum_lookback = max(
            (int(row.rolling_period_days or 1) for row in rule_rows),
            default=1,
        )
        assignment_rows = self.dependencies.Assignment.query.filter(
            self.dependencies.Assignment.unit_id == unit_id,
            self.dependencies.Assignment.staff_id.in_(ids),
            self.dependencies.Assignment.day
            >= start - timedelta(days=max(1, maximum_lookback) - 1),
            self.dependencies.Assignment.day <= end,
        ).all()

        return EligibilityContext(
            staff_by_id={row.id: row for row in staff_rows},
            shift_by_id={row.id: row for row in shift_rows},
            leaves_by_staff=_group_rows(leave_rows, "staff_id"),
            pattern_assignments_by_staff=_group_rows(
                pattern_assignment_rows, "staff_id"
            ),
            patterns_by_id={row.id: row for row in pattern_rows},
            pattern_days_by_key={
                (row.work_pattern_id, int(row.day_index)): row
                for row in pattern_day_rows
            },
            allowed_shift_ids_by_day={
                day_id: frozenset(
                    row.shift_type_id
                    for row in allowed_rows
                    if row.work_pattern_day_id == day_id
                )
                for day_id in pattern_day_ids
            },
            rules_by_staff=_group_rows(rule_rows, "staff_id"),
            assignments_by_staff=_group_rows(assignment_rows, "staff_id"),
            shifts_by_code={row.code: row for row in shift_rows},
        )

    def is_staff_eligible_for_shift(
        self, staff_id: int, on_date: date, shift_type_id: int,
        *, existing_assignment: bool = False,
        context: EligibilityContext | None = None,
    ) -> EligibilityResult:
        if context is None:
            staff = self.dependencies.Staff.query.filter_by(id=staff_id).first()
            shift = self.dependencies.ShiftType.query.filter_by(
                id=shift_type_id, is_active=True
            ).first()
        else:
            staff = context.staff_by_id.get(staff_id)
            shift = context.shift_by_id.get(shift_type_id)
        if (
            not staff or not shift or not shift.is_active
            or shift.unit_id != staff.unit_id
        ):
            return _blocked(
                "SHIFT_UNAVAILABLE",
                "The staff member or shift is unavailable in this airport.",
            )
        if context is None:
            leave = self.dependencies.Leave.query.filter(
                self.dependencies.Leave.staff_id == staff.id,
                self.dependencies.Leave.start <= on_date,
                self.dependencies.Leave.end >= on_date,
            ).first()
        else:
            leave = next((
                row for row in context.leaves_by_staff.get(staff.id, ())
                if row.start <= on_date <= row.end
            ), None)
        if leave:
            return _blocked("APPROVED_LEAVE", "Employee is on approved leave.")

        resolution = (
            self.get_pattern_day_for_staff(staff.id, on_date)
            if context is None
            else self._pattern_resolution_from_context(staff.id, on_date, context)
        )
        if resolution:
            pattern_reason = self._pattern_eligibility(resolution, shift)
            if pattern_reason:
                return _blocked(pattern_reason.code, pattern_reason.explanation)

        group = self.dependencies.shift_group(shift).upper()
        is_night = group == "N"
        is_early = bool(shift.start_time and shift.start_time < self.dependencies.early_start_before)
        rules = (
            self.get_effective_staff_rules(staff.id, on_date)
            if context is None
            else tuple(
                rule for rule in context.rules_by_staff.get(staff.id, ())
                if rule.effective_from <= on_date
                and (rule.effective_to is None or rule.effective_to >= on_date)
            )
        )
        hard_reasons: list[EligibilityReason] = []
        allowed_rules = [
            rule for rule in rules
            if rule.hardness == "HARD" and rule.rule_type == "ALLOWED_SHIFT"
        ]
        if allowed_rules and not any(
            _rule_targets_shift(rule, shift, group) for rule in allowed_rules
        ):
            hard_reasons.append(_rule_reason(
                allowed_rules[0],
                "SHIFT_NOT_ALLOWED_RULE",
                "This shift is outside the employee's allowed shifts.",
            ))
        for rule in rules:
            if rule.hardness != "HARD":
                continue
            reason = self._hard_rule_reason(
                rule, shift, group, is_night, is_early, on_date,
                existing_assignment=existing_assignment,
                context=context,
            )
            if reason:
                hard_reasons.append(reason)
        if hard_reasons:
            first = hard_reasons[0]
            return EligibilityResult(
                eligible=False,
                reason_code=first.code,
                explanation=first.explanation,
                reasons=tuple(hard_reasons),
            )
        penalty, soft_reasons = self._soft_rule_penalty(
            rules, shift, group, is_night, is_early, on_date
        )
        return EligibilityResult(
            eligible=True,
            reason_code="ELIGIBLE",
            explanation="Employee is eligible for this shift.",
            reasons=tuple(soft_reasons),
            soft_penalty=penalty,
        )

    def _pattern_resolution_from_context(
        self, staff_id: int, on_date: date, context: EligibilityContext
    ) -> PatternResolution | None:
        assignment = next((
            row for row in context.pattern_assignments_by_staff.get(staff_id, ())
            if row.effective_from <= on_date
            and (row.effective_to is None or row.effective_to >= on_date)
        ), None)
        if not assignment:
            return None
        pattern = context.patterns_by_id.get(assignment.work_pattern_id)
        if not pattern or int(pattern.cycle_length_days or 0) <= 0:
            return None
        cycle_index = (
            int(assignment.anchor_day_index)
            + (on_date - assignment.anchor_date).days
        ) % int(pattern.cycle_length_days)
        pattern_day = context.pattern_days_by_key.get((pattern.id, cycle_index))
        if not pattern_day:
            return None
        return PatternResolution(
            assignment=assignment,
            pattern=pattern,
            cycle_index=cycle_index,
            pattern_day=pattern_day,
            allowed_shift_type_ids=context.allowed_shift_ids_by_day.get(
                pattern_day.id, frozenset()
            ),
        )

    def get_pattern_day_from_context(
        self, staff_id: int, on_date: date, context: EligibilityContext
    ) -> PatternResolution | None:
        """Resolve a pattern day from preloaded range data without new queries."""
        return self._pattern_resolution_from_context(staff_id, on_date, context)

    def calculate_soft_rule_penalty(
        self, staff_id: int, on_date: date, shift_type_id: int
    ) -> int:
        result = self.is_staff_eligible_for_shift(staff_id, on_date, shift_type_id)
        return result.soft_penalty if result.eligible else 0

    def _pattern_eligibility(self, resolution: PatternResolution, shift: Any) -> EligibilityReason | None:
        day = resolution.pattern_day
        if day.day_type in {"OFF", "PROTECTED_NON_OPERATIONAL"}:
            return EligibilityReason(
                "PATTERN_NON_WORKING_DAY",
                "This date is a protected non-working day in the active pattern.",
            )
        if day.day_type == "FIXED_SHIFT" and day.fixed_shift_type_id != shift.id:
            return EligibilityReason(
                "PATTERN_FIXED_SHIFT_MISMATCH",
                "The active pattern requires a different fixed shift on this date.",
            )
        if (
            day.day_type == "WORK_ALLOWED_SET"
            and shift.id not in resolution.allowed_shift_type_ids
        ):
            return EligibilityReason(
                "PATTERN_SHIFT_NOT_ALLOWED",
                "This shift is not in the allowed set for the active pattern day.",
            )
        return None

    def _hard_rule_reason(
        self, rule: Any, shift: Any, group: str, is_night: bool,
        is_early: bool, on_date: date, *, existing_assignment: bool = False,
        context: EligibilityContext | None = None,
    ) -> EligibilityReason | None:
        applies = _rule_targets_shift(rule, shift, group)
        if rule.rule_type == "NO_NIGHT" and is_night:
            return _rule_reason(rule, "NO_NIGHT_RULE", "Employee has an active hard restriction preventing night duties.")
        if rule.rule_type == "NO_EARLY" and is_early:
            return _rule_reason(rule, "NO_EARLY_RULE", "Employee has an active hard restriction preventing early duties.")
        if rule.rule_type == "DISALLOWED_SHIFT" and applies:
            return _rule_reason(rule, "DISALLOWED_SHIFT_RULE", "This shift is explicitly disallowed for the employee.")
        weekday_allowed = _weekday_in_mask(on_date.weekday(), rule.weekdays_mask)
        if rule.rule_type == "AVAILABLE_WEEKDAYS" and not weekday_allowed:
            return _rule_reason(rule, "WEEKDAY_NOT_AVAILABLE", "Employee is not available on this weekday.")
        if rule.rule_type == "UNAVAILABLE_WEEKDAYS" and weekday_allowed:
            return _rule_reason(rule, "WEEKDAY_UNAVAILABLE", "Employee is unavailable on this weekday.")
        if rule.rule_type in {"MAX_NIGHTS_PER_CYCLE", "MAX_SHIFTS_PER_CYCLE"}:
            if rule.rule_type == "MAX_NIGHTS_PER_CYCLE" and not is_night:
                return None
            count = self._assignment_count(
                rule, on_date,
                nights_only=rule.rule_type == "MAX_NIGHTS_PER_CYCLE",
                context=context,
            )
            limit = int(rule.maximum_count or 0)
            if count > limit or (count >= limit and not existing_assignment):
                return _rule_reason(rule, rule.rule_type, "Employee has reached the configured maximum duty count.")
        if rule.rule_type == "MAX_CONTRACTED_MINUTES":
            proposed_minutes = self._assigned_minutes(
                rule, on_date, context=context
            )
            if not existing_assignment:
                proposed_minutes += _shift_minutes(shift)
            if proposed_minutes > int(rule.maximum_count or 0):
                return _rule_reason(rule, "MAX_CONTRACTED_MINUTES", "This shift would exceed the employee's configured maximum minutes.")
        return None

    def _soft_rule_penalty(
        self, rules: tuple[Any, ...], shift: Any, group: str,
        is_night: bool, is_early: bool, on_date: date,
    ) -> tuple[int, list[EligibilityReason]]:
        penalty = 0
        reasons: list[EligibilityReason] = []
        for rule in rules:
            if rule.hardness != "SOFT":
                continue
            breached = (
                (rule.rule_type == "AVOID_NIGHT" and is_night)
                or (rule.rule_type == "AVOID_EARLY" and is_early)
                or (rule.rule_type == "PREFERRED_SHIFT" and not _rule_targets_shift(rule, shift, group))
                or (rule.rule_type == "PREFERRED_DAY_OFF" and _weekday_in_mask(on_date.weekday(), rule.weekdays_mask))
            )
            if breached:
                weight = max(0, int(rule.penalty_weight or 0))
                penalty += weight
                reasons.append(_rule_reason(
                    rule, f"SOFT_{rule.rule_type}",
                    "Assignment breaches an active staff preference.",
                ))
        return penalty, reasons

    def _assignment_count(
        self, rule: Any, on_date: date, *, nights_only: bool,
        context: EligibilityContext | None = None,
    ) -> int:
        start = on_date - timedelta(days=max(1, int(rule.rolling_period_days or 1)) - 1)
        if context is None:
            rows = self.dependencies.Assignment.query.filter(
                self.dependencies.Assignment.staff_id == rule.staff_id,
                self.dependencies.Assignment.day >= start,
                self.dependencies.Assignment.day <= on_date,
            ).all()
            shifts = self.dependencies.ShiftType.query.filter_by(
                unit_id=rule.unit_id, is_working=True
            ).all()
        else:
            rows = (
                row for row in context.assignments_by_staff.get(rule.staff_id, ())
                if start <= row.day <= on_date
            )
            shifts = context.shift_by_id.values()
        codes = {
            shift.code for shift in shifts
            if shift.is_working
            if not nights_only
            or self.dependencies.shift_group(shift).upper() == "N"
        }
        return sum(1 for row in rows if row.code in codes)

    def _assigned_minutes(
        self, rule: Any, on_date: date,
        *, context: EligibilityContext | None = None,
    ) -> int:
        start = on_date - timedelta(days=max(1, int(rule.rolling_period_days or 1)) - 1)
        if context is None:
            rows = self.dependencies.Assignment.query.filter(
                self.dependencies.Assignment.staff_id == rule.staff_id,
                self.dependencies.Assignment.day >= start,
                self.dependencies.Assignment.day <= on_date,
            ).all()
            shifts = {
                row.code: row for row in self.dependencies.ShiftType.query.filter_by(
                    unit_id=rule.unit_id
                ).all()
            }
        else:
            rows = (
                row for row in context.assignments_by_staff.get(rule.staff_id, ())
                if start <= row.day <= on_date
            )
            shifts = context.shifts_by_code
        return sum(_shift_minutes(shifts.get(row.code)) for row in rows)


def _date_ranges_overlap(
    first_start: date, first_end: date | None,
    second_start: date, second_end: date | None,
) -> bool:
    return first_start <= (second_end or date.max) and second_start <= (first_end or date.max)


def _group_rows(rows: Iterable[Any], attribute: str) -> dict[int, tuple[Any, ...]]:
    grouped: dict[int, list[Any]] = {}
    for row in rows:
        grouped.setdefault(int(getattr(row, attribute)), []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _blocked(code: str, explanation: str) -> EligibilityResult:
    reason = EligibilityReason(code, explanation)
    return EligibilityResult(False, code, explanation, (reason,))


def _rule_reason(rule: Any, code: str, explanation: str) -> EligibilityReason:
    return EligibilityReason(code, explanation, getattr(rule, "id", None))


def _rule_targets_shift(rule: Any, shift: Any, group: str) -> bool:
    if rule.shift_type_id is not None:
        return int(rule.shift_type_id) == int(shift.id)
    if rule.shift_group:
        return str(rule.shift_group).upper() == group.upper()
    return False


def _weekday_in_mask(weekday: int, mask: int | None) -> bool:
    return mask is not None and bool(int(mask) & (1 << weekday))


def _shift_minutes(shift: Any | None) -> int:
    if not shift or not shift.start_time or not shift.end_time:
        return 0
    start = shift.start_time.hour * 60 + shift.start_time.minute
    end = shift.end_time.hour * 60 + shift.end_time.minute
    if end <= start:
        end += 24 * 60
    return end - start
