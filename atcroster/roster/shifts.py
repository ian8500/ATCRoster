"""Roster shift catalogue grouping."""

from __future__ import annotations

from typing import Any, Callable


def shift_groups_snapshot(ShiftType: Any, unit_id: int, banned_codes: Callable[[], set[str]]) -> tuple[list[Any], list[Any], list[Any]]:
    """Group configured shifts by operational role for roster selectors."""
    shifts = ShiftType.query.filter_by(unit_id=unit_id).order_by(ShiftType.code).all()
    allowed = [shift for shift in shifts if shift.code not in banned_codes()]
    return (
        sorted((shift for shift in allowed if shift.is_working and not shift.is_training), key=lambda shift: shift.code),
        sorted((shift for shift in allowed if shift.is_training), key=lambda shift: shift.code),
        sorted((shift for shift in allowed if not shift.is_working and not shift.is_training), key=lambda shift: shift.code),
    )
