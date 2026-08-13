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


def assignment_for_day(db: Any, Assignment: Any, staff_id: int, day: date) -> Any:
    """Load or create the one assignment row for a staff member and date."""
    assignment = Assignment.query.filter_by(staff_id=staff_id, day=day).first()
    if assignment is None:
        assignment = Assignment(staff_id=staff_id, day=day)
        db.session.add(assignment)
    return assignment
