"""Effective watch membership resolution."""

from __future__ import annotations

from datetime import date
from typing import Any


def watch_ids_for_staff_on(StaffWatchHistory: Any, staff: list[Any], unit_id: int, on_date: date) -> dict[int, int | None]:
    """Resolve a roster's watch memberships with one history query."""
    staff_by_id = {person.id: person for person in staff}
    if not staff_by_id:
        return {}
    rows = StaffWatchHistory.query.filter(
        StaffWatchHistory.unit_id == unit_id,
        StaffWatchHistory.staff_id.in_(staff_by_id),
        StaffWatchHistory.effective_date <= on_date,
    ).order_by(
        StaffWatchHistory.staff_id, StaffWatchHistory.effective_date.desc(),
        StaffWatchHistory.id.desc(),
    ).all()
    resolved: dict[int, int | None] = {}
    for row in rows:
        resolved.setdefault(row.staff_id, row.watch_id)
    for staff_id, person in staff_by_id.items():
        resolved.setdefault(staff_id, person.watch_id)
    return resolved


def watch_id_for_staff_on(db: Any, StaffWatchHistory: Any, Staff: Any, unit_id: int, staff_id: int, on_date: date) -> int | None:
    """Resolve one effective watch, falling back to the staff record."""
    history = StaffWatchHistory.query.filter(
        StaffWatchHistory.unit_id == unit_id,
        StaffWatchHistory.staff_id == staff_id,
        StaffWatchHistory.effective_date <= on_date,
        db.or_(StaffWatchHistory.effective_to.is_(None), StaffWatchHistory.effective_to >= on_date),
    ).order_by(StaffWatchHistory.effective_date.desc()).first()
    if history:
        return history.watch_id
    staff = Staff.query.filter_by(id=staff_id, unit_id=unit_id).first()
    return staff.watch_id if staff else None
