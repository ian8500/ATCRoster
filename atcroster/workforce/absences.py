"""Workforce absence checks for roster safety rules."""

from __future__ import annotations

from datetime import date
from typing import Any


def has_leave_or_sickness(Leave: Any, Sickness: Any, staff_id: int, day: date) -> bool:
    """Return whether a staff member has a leave or sickness record on a day."""
    return bool(
        Leave.query.filter(Leave.staff_id == staff_id, Leave.start <= day, Leave.end >= day).first()
        or Sickness.query.filter(Sickness.staff_id == staff_id, Sickness.start <= day, Sickness.end >= day).first()
    )
