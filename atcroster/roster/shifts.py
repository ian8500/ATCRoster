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
