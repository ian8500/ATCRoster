"""Roster assignment persistence primitives."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AssignmentRefreshDependencies:
    db: Any
    Assignment: Any
    Staff: Any
    code_from_pattern: Callable[[Any, date], str]
    day_leave_for: Callable[[Any, date], str | None]
    get_shift: Callable[[str], Any]
    absence_types: Callable[..., list[dict[str, Any]]]


def set_generated_assignment(
    staff: Any,
    day: date,
    code: str,
    *,
    dependencies: AssignmentRefreshDependencies,
    source: str = "auto",
    note: str = "",
):
    assignment = dependencies.Assignment.query.filter_by(
        staff_id=staff.id, day=day
    ).first()
    if assignment is None:
        assignment = dependencies.Assignment(staff=staff, day=day, code=code)
        dependencies.db.session.add(assignment)
    assignment.set_generated_baseline(
        code, generation_version="legacy-pattern-compat-v1"
    )
    if assignment.override_code is None:
        assignment.source = source
        assignment.note = note
    return assignment


def set_absence_override(
    staff: Any,
    day: date,
    code: str,
    *,
    dependencies: AssignmentRefreshDependencies,
    note: str = "",
):
    assignment = dependencies.Assignment.query.filter_by(
        staff_id=staff.id, day=day
    ).first()
    if assignment is None:
        assignment = dependencies.Assignment(staff=staff, day=day, code=code)
        dependencies.db.session.add(assignment)
    assignment.set_editor_override(
        code,
        reason=note or "System-managed absence",
        override_type="SYSTEM_ABSENCE",
    )
    assignment.source = "leave"
    assignment.note = note or assignment.note
    return assignment


def refresh_pattern_day(
    staff: Any,
    day: date,
    dependencies: AssignmentRefreshDependencies,
):
    """Apply pattern and leave overlays while preserving explicit edits."""
    assignment = dependencies.Assignment.query.filter_by(
        staff_id=staff.id, day=day
    ).first()
    previous_code = assignment.effective_code if assignment else None
    if (
        assignment
        and (assignment.code or "").strip()
        and assignment.source in {"manual", "ai"}
    ):
        return None
    sickness_codes = {
        item["code"]
        for item in dependencies.absence_types(
            "sickness", active_only=False, unit_id=staff.unit_id
        )
    }
    if assignment and assignment.code in sickness_codes | {"TOU8", "TOUI"}:
        return assignment

    pattern_code = dependencies.code_from_pattern(staff, day)
    leave_code = dependencies.day_leave_for(staff, day)
    if leave_code == "AL":
        pattern_shift = dependencies.get_shift(pattern_code)
        if pattern_shift and pattern_shift.is_working:
            result = set_absence_override(
                staff,
                day,
                "AL",
                dependencies=dependencies,
                note="leave",
            )
            result.annotation = ""
            return result
        result = set_generated_assignment(
            staff,
            day,
            pattern_code,
            dependencies=dependencies,
            source="auto",
            note="pattern",
        )
        if previous_code is None or (
            previous_code != result.code and result.source != "manual"
        ):
            result.annotation = ""
        return result
    if leave_code:
        result = set_absence_override(
            staff,
            day,
            leave_code,
            dependencies=dependencies,
            note="leave",
        )
        result.annotation = ""
        return result
    if assignment and assignment.override_type in {
        "MIGRATED_ABSENCE",
        "SYSTEM_ABSENCE",
    }:
        assignment.clear_editor_override()
    result = set_generated_assignment(
        staff,
        day,
        pattern_code,
        dependencies=dependencies,
        source="auto",
        note="pattern",
    )
    if previous_code is None or (
        previous_code != result.code and result.source != "manual"
    ):
        result.annotation = ""
    return result


def generate_assignment_range(
    start_day: date,
    end_day: date,
    *,
    iter_year_months: Callable[[date, date], Any],
    ensure_month_requirement: Callable[[int, int], Any],
    generate_month: Callable[[int, int], None],
) -> None:
    for year, month in iter_year_months(start_day, end_day):
        ensure_month_requirement(year, month)
        generate_month(year, month)


def generate_month_assignments(
    year: int,
    month: int,
    *,
    db: Any,
    Staff: Any,
    month_range: Callable[[int, int], tuple[Any, list[date]]],
    refresh_day: Callable[[Any, date], Any],
) -> None:
    """Refresh every non-monitor assignment in a roster month."""
    _, days = month_range(year, month)
    for staff in Staff.query.filter(Staff.role != "position_monitor").order_by(
        Staff.id
    ):
        for day in days:
            refresh_day(staff, day)
    db.session.commit()


def allocate_day_shift_shortfall(
    day: date,
    requirement: Any,
    staff: list[Any],
    assignments_by_staff: dict[int, dict[date, Any]],
    weekday_code: str,
    sunday_code: str,
    *,
    db: Any,
    Assignment: Any,
    is_working_day_code: Callable[[str], bool],
    has_leave_or_sickness: Callable[[int, date], bool],
    is_empty_like: Callable[[Any], bool],
    passes_fatigue: Callable[[Any, date, str], bool],
    set_code: Callable[..., Any],
) -> int:
    """Fill an unmet day-duty requirement without replacing protected cells."""
    existing = Assignment.query.filter_by(day=day).all()
    assigned = sum(
        1 for assignment in existing if is_working_day_code(assignment.code or "")
    )
    required = getattr(requirement, "req_d", 0) if requirement else 0
    shortfall = max(0, required - assigned)
    if not shortfall:
        return 0
    changes = 0
    code = weekday_code if day.weekday() < 6 else sunday_code
    for person in staff:
        if not shortfall:
            break
        if has_leave_or_sickness(person.id, day):
            continue
        assignment = assignments_by_staff[person.id].get(day)
        if not is_empty_like(assignment.code if assignment else ""):
            continue
        if not passes_fatigue(person, day, code):
            continue
        if assignment is None:
            assignment = Assignment(staff_id=person.id, day=day)
            db.session.add(assignment)
            assignments_by_staff[person.id][day] = assignment
        set_code(assignment, code, source="ai", note="AI fill D")
        changes += 1
        shortfall -= 1
    return changes


def assignment_for_day(db: Any, Assignment: Any, staff_id: int, day: date) -> Any:
    """Load or create the one assignment row for a staff member and date."""
    assignment = Assignment.query.filter_by(staff_id=staff_id, day=day).first()
    if assignment is None:
        assignment = Assignment(staff_id=staff_id, day=day)
        db.session.add(assignment)
    return assignment


@dataclass(frozen=True)
class AssignmentRuntimeDependencies:
    refresh: AssignmentRefreshDependencies
    Requirement: Any
    SpecialRequirement: Any
    month_range: Callable[[int, int], tuple[Any, list[date]]]
    shift_minutes: Callable[[Any], int]
    daily_requirements: Callable[..., dict[str, int]]
    ensure_month_requirement: Callable[..., Any]
    requirements_for_day: Callable[..., dict[str, int]]


class AssignmentRuntime:
    """Own roster assignment refresh, requirements, and month generation."""

    def __init__(self, dependencies: AssignmentRuntimeDependencies):
        self.dependencies = dependencies

    def set_assignment(
        self,
        staff: Any,
        day: date,
        code: str,
        source: str = "auto",
        note: str = "",
    ):
        return set_generated_assignment(
            staff,
            day,
            code,
            dependencies=self.dependencies.refresh,
            source=source,
            note=note,
        )

    def overwrite_assignment(self, staff: Any, day: date, code: str, note: str = ""):
        return set_absence_override(
            staff,
            day,
            code,
            dependencies=self.dependencies.refresh,
            note=note,
        )

    def refresh_day(self, staff: Any, day: date):
        return refresh_pattern_day(staff, day, self.dependencies.refresh)

    def shift_duration_minutes(self, shift: Any) -> int:
        return self.dependencies.shift_minutes(shift)

    def ensure_month_requirement(
        self, year: int, month: int, default: tuple[int, int, int, int] = (4, 4, 4, 2)
    ):
        deps = self.dependencies
        return deps.ensure_month_requirement(
            deps.refresh.db, deps.Requirement, year, month, default
        )

    def requirements_for_day(
        self, requirement: Any, day: date, special: Any = None
    ) -> dict[str, int]:
        deps = self.dependencies
        return deps.requirements_for_day(
            requirement, day, special, deps.daily_requirements
        )

    def generate_month(self, year: int, month: int, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        deps = self.dependencies
        return generate_month_assignments(
            year,
            month,
            db=deps.refresh.db,
            Staff=deps.refresh.Staff,
            month_range=deps.month_range,
            refresh_day=self.refresh_day,
        )
