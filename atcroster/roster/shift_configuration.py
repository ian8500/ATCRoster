"""Shift definition administration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ShiftConfigurationDependencies:
    db: Any
    ShiftType: Any
    QualificationType: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    parse_hhmm: Callable[[str | None], Any]
    record_roster_impact: Callable[..., None]
    prune_roster_code_settings: Callable[[int], int]
    refresh_shift_cache: Callable[[], None]
    clear_shift_groups_cache: Callable[[], None]


def update_shift_definition(
    action: str,
    values: Mapping[str, str],
    dependencies: ShiftConfigurationDependencies,
) -> tuple[str, str]:
    """Create, edit, or remove one unit-scoped shift definition."""
    unit_id = dependencies.current_unit_id()
    if action == "shift_delete":
        shift = dependencies.ShiftType.query.filter_by(
            id=int(values.get("shift_id") or 0), unit_id=unit_id
        ).first_or_404()
        code = shift.code
        dependencies.db.session.delete(shift)
        dependencies.db.session.flush()
        dependencies.prune_roster_code_settings(unit_id)
        _record_impact(dependencies, f"Shift {code} removed.")
        _finish(dependencies)
        return "Shift deleted.", "ok"

    required_qualification = (
        (values.get("required_qualification") or "").strip().upper()
    )
    allowed = {
        row.code
        for row in dependencies.QualificationType.query.filter_by(
            unit_id=unit_id, is_active=True
        ).all()
    } | {""}
    if required_qualification not in allowed:
        return "Unknown required qualification.", "error"
    requested = bool(values.get("is_requestable"))
    active = bool(values.get("is_active"))
    working = bool(values.get("is_working"))
    if requested and (not active or not working):
        return "Only active working shifts can be requestable.", "error"

    if action == "shift_new":
        code = (values.get("code") or "").strip().upper()
        if not code:
            return "Shift code is required.", "error"
        if dependencies.ShiftType.query.filter_by(unit_id=unit_id, code=code).first():
            return "Shift code already exists.", "error"
        shift = dependencies.ShiftType(
            unit_id=unit_id,
            code=code,
            name=(values.get("name") or "").strip() or code,
            start_time=dependencies.parse_hhmm(values.get("start")),
            end_time=dependencies.parse_hhmm(values.get("end")),
            is_working=working,
            is_training=bool(values.get("is_training")),
            is_active=active,
            is_requestable=requested,
            required_qualification=required_qualification,
        )
        dependencies.db.session.add(shift)
        _record_impact(dependencies, f"Shift {code} created.")
        _finish(dependencies)
        return "Shift added.", "ok"

    shift = dependencies.ShiftType.query.filter_by(
        id=int(values.get("shift_id") or 0), unit_id=unit_id
    ).first_or_404()
    shift.name = (values.get("name") or "").strip() or shift.name
    shift.start_time = dependencies.parse_hhmm(values.get("start"))
    shift.end_time = dependencies.parse_hhmm(values.get("end"))
    shift.is_working = working
    shift.is_training = bool(values.get("is_training"))
    shift.is_active = active
    shift.is_requestable = requested
    shift.required_qualification = required_qualification
    _record_impact(dependencies, f"Shift {shift.code} definition changed.")
    _finish(dependencies)
    return "Shift updated.", "ok"


def _record_impact(dependencies: ShiftConfigurationDependencies, reason: str) -> None:
    dependencies.record_roster_impact(
        dependencies.RosterImpactEventType.SHIFT_DEFINITION_CHANGE,
        date.today(),
        rebuild_baseline=True,
        reason=reason,
    )


def _finish(dependencies: ShiftConfigurationDependencies) -> None:
    dependencies.db.session.commit()
    dependencies.refresh_shift_cache()
    dependencies.clear_shift_groups_cache()
