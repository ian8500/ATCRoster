"""View-model construction for the roster administration page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable


@dataclass(frozen=True)
class AdminContextDependencies:
    db: Any
    Watch: Any
    ShiftType: Any
    QualificationType: Any
    WorkPattern: Any
    Staff: Any
    Requirement: Any
    SpecialRequirement: Any
    Leave: Any
    Unit: Any
    current_unit_id: Callable[[], int]
    roster_settings_snapshot: Callable[[int], dict[str, str]]
    validate_pattern: Callable[[str | None], list[str]]
    shift_counter_group: Callable[[str, int], str]
    sms_number_options: Callable[[str], list[dict[str, str]]]
    sms_operational_options: Callable[[], list[dict[str, str]]]
    sms_default_number: Callable[[str, list[dict[str, str]]], str]
    absence_types: Callable[..., list[dict[str, Any]]]
    default_base_pattern: str
    pattern_codes: Any


def build_admin_context(dependencies: AdminContextDependencies) -> dict[str, Any]:
    """Build the read-only data required by the administration template."""
    unit_id = dependencies.current_unit_id()
    watches = dependencies.Watch.query.order_by(dependencies.Watch.order_index).all()
    shifts = dependencies.ShiftType.query.order_by(dependencies.ShiftType.code).all()
    qualification_types = (
        dependencies.QualificationType.query.filter_by(unit_id=unit_id, is_active=True)
        .order_by(dependencies.QualificationType.code)
        .all()
    )
    work_patterns = (
        dependencies.WorkPattern.query.filter_by(unit_id=unit_id, is_active=True)
        .order_by(dependencies.WorkPattern.name)
        .all()
    )
    staff = (
        dependencies.Staff.query.outerjoin(
            dependencies.Watch,
            dependencies.Staff.watch_id == dependencies.Watch.id,
        )
        .filter(dependencies.Staff.role != "position_monitor")
        .order_by(dependencies.Watch.order_index, dependencies.Staff.name)
        .all()
    )
    cursor = date.today().replace(day=1)
    months = []
    for _ in range(24):
        months.append((cursor.year, cursor.month))
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12
            else cursor.replace(month=cursor.month + 1)
        )
    settings = dependencies.roster_settings_snapshot(unit_id)
    sms_senders = dependencies.sms_number_options("sms_sender_numbers")
    sms_destinations = dependencies.sms_operational_options()
    return {
        "shifts": shifts,
        "staff": staff,
        "watches": watches,
        "months": months,
        "requirements_by_month": {
            (row.year, row.month): row for row in dependencies.Requirement.query.all()
        },
        "special_requirements": dependencies.SpecialRequirement.query.order_by(
            dependencies.SpecialRequirement.day
        ).all(),
        "leaves": dependencies.Leave.query.order_by(
            dependencies.Leave.start.desc()
        ).all(),
        "qualification_types": qualification_types,
        "work_patterns": work_patterns,
        "today": date.today(),
        "base_pattern": ",".join(
            dependencies.validate_pattern(
                settings.get("base_pattern_csv") or dependencies.default_base_pattern
            )
        ),
        "base_anchor": settings.get("base_pattern_anchor") or "2025-01-01",
        "night_active_days": {
            int(value)
            for value in settings.get("night_active_weekdays", "0,1,2,3,4,5,6").split(
                ","
            )
            if value.strip().isdigit()
        },
        "pattern_codes": dependencies.pattern_codes,
        "shift_counter_mapping": {
            shift.code: dependencies.shift_counter_group(shift.code, unit_id)
            for shift in shifts
        },
        "sms_senders": sms_senders,
        "sms_operational_numbers": sms_destinations,
        "sms_default_sender": dependencies.sms_default_number(
            "sms_default_sender", sms_senders
        ),
        "sms_default_operational_number": dependencies.sms_default_number(
            "sms_default_operational_number", sms_destinations
        ),
        "absence_types": dependencies.absence_types(active_only=False),
        "current_unit": dependencies.db.session.get(dependencies.Unit, unit_id),
    }
