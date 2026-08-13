"""Deterministic population of generated roster baselines.

The service owns only ``Assignment.generated_*`` fields.  Editor overrides are
never changed here.  It is deliberately route-independent so workforce-event
handlers and maintenance jobs can share one implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Iterable

from roster_horizon import get_unit_automatic_recalculation_start


GENERATION_VERSION = "deterministic-baseline-v1"
AUTOMATIC_MODES = frozenset({"automatic", "event"})


@dataclass(frozen=True)
class PopulationChange:
    staff_id: int
    day: date
    previous_generated_code: str | None
    generated_code: str
    effective_code: str
    pattern_id: int | None
    pattern_day_index: int | None
    action: str


@dataclass(frozen=True)
class PopulationResult:
    requested_from: date
    requested_to: date
    populated_from: date | None
    populated_to: date | None
    mode: str
    dry_run: bool
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    protected_dates: int = 0
    unresolved_flexible_dates: int = 0
    changes: tuple[PopulationChange, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> int:
        return self.created + self.updated


@dataclass(frozen=True)
class PopulationDependencies:
    db: Any
    Unit: Any
    Staff: Any
    Assignment: Any
    ShiftType: Any
    WorkPattern: Any
    WorkPatternDay: Any
    WorkPatternDayAllowedShift: Any
    StaffPatternAssignment: Any
    utcnow: Callable[[], Any]
    legacy_code_resolver: Callable[[Any, date], str]
    workforce_is_active: Callable[[Any, date], bool] | None = None
    watch_id_resolver: Callable[[Any, date], int | None] | None = None
    RosterPeriod: Any = None


def create_population_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> PopulationDependencies:
    """Bind baseline-population records at the roster service boundary."""
    return PopulationDependencies(
        db=db,
        Unit=operational_models.Unit,
        Staff=operational_models.Staff,
        Assignment=operational_models.Assignment,
        ShiftType=operational_models.ShiftType,
        WorkPattern=saas_models.WorkPattern,
        WorkPatternDay=saas_models.WorkPatternDay,
        WorkPatternDayAllowedShift=saas_models.WorkPatternDayAllowedShift,
        StaffPatternAssignment=saas_models.StaffPatternAssignment,
        RosterPeriod=saas_models.RosterPeriod,
        **services,
    )


def create_deterministic_roster_population_service(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> "DeterministicRosterPopulationService":
    """Build the shared baseline population service from domain-owned bindings."""
    return DeterministicRosterPopulationService(
        create_population_dependencies(
            db=db,
            operational_models=operational_models,
            saas_models=saas_models,
            **services,
        )
    )


class DeterministicRosterPopulationService:
    """Populate future generated codes from dated pattern assignments."""

    def __init__(self, dependencies: PopulationDependencies) -> None:
        self.dependencies = dependencies

    def populate_or_recalculate_baseline(
        self,
        unit_id: int,
        effective_from: date,
        effective_to: date,
        *,
        staff_ids: Iterable[int] | None = None,
        watch_ids: Iterable[int] | None = None,
        mode: str = "automatic",
        reason: str | None = None,
        triggered_by_user_id: int | None = None,
        reference_date: date | None = None,
        generation_event_id: int | None = None,
        dry_run: bool = False,
    ) -> PopulationResult:
        del reason, triggered_by_user_id  # reserved for the event/audit stage
        if effective_to < effective_from:
            raise ValueError("Roster population end date cannot precede its start date.")
        normalised_mode = (mode or "automatic").strip().lower()
        if normalised_mode not in AUTOMATIC_MODES | {"manual"}:
            raise ValueError("Roster population mode must be automatic, event or manual.")

        dep = self.dependencies
        unit = dep.db.session.get(dep.Unit, int(unit_id))
        if not unit:
            raise ValueError("The selected airport does not exist.")

        population_start = effective_from
        protected_dates = 0
        if normalised_mode in AUTOMATIC_MODES:
            boundary = get_unit_automatic_recalculation_start(unit, reference_date)
            if effective_from < boundary:
                protected_end = min(effective_to, boundary - timedelta(days=1))
                protected_dates = (protected_end - effective_from).days + 1
                population_start = boundary
        if population_start > effective_to:
            return PopulationResult(
                requested_from=effective_from,
                requested_to=effective_to,
                populated_from=None,
                populated_to=None,
                mode=normalised_mode,
                dry_run=dry_run,
                protected_dates=protected_dates,
            )

        staff_query = dep.Staff.query.filter(
            dep.Staff.unit_id == unit.id,
            dep.Staff.role != "position_monitor",
        )
        selected_staff_ids = tuple(sorted({int(value) for value in staff_ids or ()}))
        selected_watch_ids = tuple(sorted({int(value) for value in watch_ids or ()}))
        if selected_staff_ids:
            staff_query = staff_query.filter(dep.Staff.id.in_(selected_staff_ids))
        staff_rows = staff_query.order_by(dep.Staff.id).all()
        staff_id_values = tuple(row.id for row in staff_rows)
        if not staff_id_values:
            return PopulationResult(
                requested_from=effective_from,
                requested_to=effective_to,
                populated_from=population_start,
                populated_to=effective_to,
                mode=normalised_mode,
                dry_run=dry_run,
                protected_dates=protected_dates,
            )

        existing_rows = dep.Assignment.query.filter(
            dep.Assignment.unit_id == unit.id,
            dep.Assignment.staff_id.in_(staff_id_values),
            dep.Assignment.day >= population_start,
            dep.Assignment.day <= effective_to,
        ).all()
        existing_by_key = {(row.staff_id, row.day): row for row in existing_rows}

        pattern_assignments = dep.StaffPatternAssignment.query.filter(
            dep.StaffPatternAssignment.unit_id == unit.id,
            dep.StaffPatternAssignment.staff_id.in_(staff_id_values),
            dep.StaffPatternAssignment.effective_from <= effective_to,
            (
                dep.StaffPatternAssignment.effective_to.is_(None)
                | (dep.StaffPatternAssignment.effective_to >= population_start)
            ),
        ).order_by(
            dep.StaffPatternAssignment.staff_id,
            dep.StaffPatternAssignment.effective_from.desc(),
            dep.StaffPatternAssignment.id.desc(),
        ).all()
        assignments_by_staff: dict[int, list[Any]] = defaultdict(list)
        for row in pattern_assignments:
            assignments_by_staff[row.staff_id].append(row)

        pattern_ids = tuple(sorted({row.work_pattern_id for row in pattern_assignments}))
        patterns = (
            dep.WorkPattern.query.filter(
                dep.WorkPattern.unit_id == unit.id,
                dep.WorkPattern.id.in_(pattern_ids),
            ).all()
            if pattern_ids else []
        )
        patterns_by_id = {row.id: row for row in patterns}
        pattern_days = (
            dep.WorkPatternDay.query.filter(
                dep.WorkPatternDay.unit_id == unit.id,
                dep.WorkPatternDay.work_pattern_id.in_(pattern_ids),
            ).all()
            if pattern_ids else []
        )
        days_by_key = {
            (row.work_pattern_id, int(row.day_index)): row for row in pattern_days
        }
        pattern_day_ids = tuple(row.id for row in pattern_days)
        allowed_rows = (
            dep.WorkPatternDayAllowedShift.query.filter(
                dep.WorkPatternDayAllowedShift.unit_id == unit.id,
                dep.WorkPatternDayAllowedShift.work_pattern_day_id.in_(pattern_day_ids),
            ).all()
            if pattern_day_ids else []
        )
        allowed_ids_by_day: dict[int, set[int]] = defaultdict(set)
        for row in allowed_rows:
            allowed_ids_by_day[row.work_pattern_day_id].add(row.shift_type_id)
        shifts = dep.ShiftType.query.filter_by(unit_id=unit.id).all()
        shifts_by_id = {row.id: row for row in shifts}
        closed_periods = set()
        if normalised_mode in AUTOMATIC_MODES and dep.RosterPeriod is not None:
            closed_periods = {
                (row.year, row.month) for row in dep.RosterPeriod.query.filter_by(
                    unit_id=unit.id, status="CLOSED"
                ).all()
            }

        created = updated = unchanged = unresolved = 0
        changes: list[PopulationChange] = []
        active_resolver = dep.workforce_is_active or _default_workforce_is_active
        generated_at = dep.utcnow()
        duty_day = population_start
        while duty_day <= effective_to:
            if (duty_day.year, duty_day.month) in closed_periods:
                protected_dates += 1
                duty_day += timedelta(days=1)
                continue
            for person in staff_rows:
                if selected_watch_ids:
                    watch_resolver = dep.watch_id_resolver or _default_watch_id
                    if watch_resolver(person, duty_day) not in selected_watch_ids:
                        continue
                existing = existing_by_key.get((person.id, duty_day))
                code, pattern_id, cycle_index, is_unresolved = self._resolve_code(
                    person,
                    duty_day,
                    existing,
                    assignments_by_staff.get(person.id, ()),
                    patterns_by_id,
                    days_by_key,
                    allowed_ids_by_day,
                    shifts_by_id,
                    active_resolver,
                )
                unresolved += int(is_unresolved)
                previous = existing.generated_code if existing else None
                metadata_matches = bool(
                    existing
                    and existing.generated_code == code
                    and existing.generation_version == GENERATION_VERSION
                    and existing.generated_from_pattern_id == pattern_id
                    and existing.generated_from_pattern_day_index == cycle_index
                )
                if metadata_matches:
                    unchanged += 1
                    continue

                action = "updated" if existing else "created"
                effective_code = (
                    existing.override_code
                    if existing and existing.override_code is not None
                    else code
                )
                changes.append(PopulationChange(
                    staff_id=person.id,
                    day=duty_day,
                    previous_generated_code=previous,
                    generated_code=code,
                    effective_code=effective_code,
                    pattern_id=pattern_id,
                    pattern_day_index=cycle_index,
                    action=action,
                ))
                if action == "created":
                    created += 1
                else:
                    updated += 1
                if dry_run:
                    continue
                if not existing:
                    existing = dep.Assignment(
                        unit_id=unit.id,
                        staff_id=person.id,
                        day=duty_day,
                        code=code,
                    )
                    dep.db.session.add(existing)
                    existing_by_key[(person.id, duty_day)] = existing
                existing.set_generated_baseline(
                    code,
                    generated_at=generated_at,
                    generation_version=GENERATION_VERSION,
                    pattern_id=pattern_id,
                    pattern_day_index=cycle_index,
                    generation_event_id=generation_event_id,
                )
                if existing.override_code is None:
                    existing.source = "auto"
                    existing.note = "deterministic baseline"
            duty_day += timedelta(days=1)

        if not dry_run:
            dep.db.session.flush()
        return PopulationResult(
            requested_from=effective_from,
            requested_to=effective_to,
            populated_from=population_start,
            populated_to=effective_to,
            mode=normalised_mode,
            dry_run=dry_run,
            created=created,
            updated=updated,
            unchanged=unchanged,
            protected_dates=protected_dates,
            unresolved_flexible_dates=unresolved,
            changes=tuple(changes),
        )

    def _resolve_code(
        self,
        person: Any,
        duty_day: date,
        existing: Any,
        pattern_assignments: Iterable[Any],
        patterns_by_id: dict[int, Any],
        days_by_key: dict[tuple[int, int], Any],
        allowed_ids_by_day: dict[int, set[int]],
        shifts_by_id: dict[int, Any],
        active_resolver: Callable[[Any, date], bool],
    ) -> tuple[str, int | None, int | None, bool]:
        if not active_resolver(person, duty_day):
            return "OFF", None, None, False
        dated_assignment = next(
            (
                row for row in pattern_assignments
                if row.effective_from <= duty_day
                and (row.effective_to is None or row.effective_to >= duty_day)
            ),
            None,
        )
        if not dated_assignment:
            code = (self.dependencies.legacy_code_resolver(person, duty_day) or "OFF")
            return code.strip().upper(), None, None, False
        pattern = patterns_by_id.get(dated_assignment.work_pattern_id)
        if not pattern or int(pattern.cycle_length_days or 0) <= 0:
            return "OFF", dated_assignment.work_pattern_id, None, True
        cycle_index = (
            int(dated_assignment.anchor_day_index)
            + (duty_day - dated_assignment.anchor_date).days
        ) % int(pattern.cycle_length_days)
        pattern_day = days_by_key.get((pattern.id, cycle_index))
        if not pattern_day:
            return "OFF", pattern.id, cycle_index, True
        if pattern_day.day_type == "FIXED_SHIFT":
            shift = shifts_by_id.get(pattern_day.fixed_shift_type_id)
            return (
                (shift.code or "OFF").strip().upper() if shift else "OFF",
                pattern.id,
                cycle_index,
                shift is None,
            )
        if pattern_day.day_type in {"OFF", "PROTECTED_NON_OPERATIONAL"}:
            return "OFF", pattern.id, cycle_index, False
        if pattern_day.day_type == "WORK_ALLOWED_SET":
            allowed = allowed_ids_by_day.get(pattern_day.id, set())
            if len(allowed) == 1:
                shift = shifts_by_id.get(next(iter(allowed)))
                if shift:
                    return shift.code.strip().upper(), pattern.id, cycle_index, False
        # A flexible day with multiple/no choices cannot be assigned without a
        # policy decision.  Store OFF explicitly and surface it to the caller.
        return "OFF", pattern.id, cycle_index, True


def _default_workforce_is_active(person: Any, _duty_day: date) -> bool:
    if getattr(person, "membership_status", "active") not in {"active", "no_login"}:
        return False
    if not getattr(person, "is_operational", True):
        return False
    for field_name in (
        "employment_start_date", "unit_join_date", "roster_start_date",
    ):
        value = getattr(person, field_name, None)
        if value and _duty_day < value:
            return False
    for field_name in (
        "employment_end_date", "unit_leave_date", "final_operational_duty_date",
        "final_unit_date",
    ):
        value = getattr(person, field_name, None)
        if value and _duty_day > value:
            return False
    return True


def _default_watch_id(person: Any, _duty_day: date) -> int | None:
    return getattr(person, "watch_id", None)
