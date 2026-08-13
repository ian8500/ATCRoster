"""Roster shift catalogue grouping."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping


def shift_groups_snapshot(
    ShiftType: Any, unit_id: int, banned_codes: Callable[[], set[str]]
) -> tuple[list[Any], list[Any], list[Any]]:
    """Group configured shifts by operational role for roster selectors."""
    shifts = ShiftType.query.filter_by(unit_id=unit_id).order_by(ShiftType.code).all()
    allowed = [shift for shift in shifts if shift.code not in banned_codes()]
    return (
        sorted(
            (shift for shift in allowed if shift.is_working and not shift.is_training),
            key=lambda shift: shift.code,
        ),
        sorted(
            (shift for shift in allowed if shift.is_training),
            key=lambda shift: shift.code,
        ),
        sorted(
            (
                shift
                for shift in allowed
                if not shift.is_working and not shift.is_training
            ),
            key=lambda shift: shift.code,
        ),
    )


def duration_minutes(shift: Any, shift_minutes: Callable[[Any], int]) -> int:
    """Return the canonical duration for one configured shift."""
    return shift_minutes(shift)


def save_counter_mapping(
    values: Mapping[str, str],
    *,
    db: Any,
    ShiftType: Any,
    unit_id: int,
    save_setting: Callable[[str, str], None],
) -> None:
    """Validate and persist the unit's shift-to-counter mapping."""
    mapping: dict[str, str] = {}
    for shift in ShiftType.query.filter_by(unit_id=unit_id).all():
        group = (values.get(f"counter_group_{shift.id}") or "").strip().upper()
        if group not in {"", "M", "D", "A", "N"}:
            raise ValueError("Invalid roster counter group.")
        mapping[shift.code.upper()] = group
    save_setting("shift_counter_map", json.dumps(mapping, sort_keys=True))
    db.session.commit()


def counter_group(
    code: str | None,
    unit_id: int,
    *,
    counter_map: Callable[[int], dict[str, str]],
    get_shift: Callable[[str, int], Any],
) -> str:
    """Resolve one configured shift into its staffing counter group."""
    value = (code or "").strip().upper()
    if not value:
        return ""
    mapping = counter_map(unit_id)
    if value in mapping:
        return mapping[value]
    shift = get_shift(value, unit_id)
    if not shift or not shift.is_active or not shift.is_working or shift.is_training:
        return ""
    if value == "EM":
        return "M"
    if value == "LA":
        return "A"
    return value if value in {"M", "D", "A", "N"} else ""


def counter_group_for_day(
    code: str | None,
    on_date: Any,
    unit_id: int,
    *,
    resolve_group: Callable[[str | None, int], str],
    night_active_on: Callable[[int, Any], bool],
) -> str:
    """Suppress night coverage when the airport is closed that day."""
    group = resolve_group(code, unit_id)
    return "" if group == "N" and not night_active_on(unit_id, on_date) else group
