"""Roster assignment persistence primitives."""

from __future__ import annotations

from datetime import date
from typing import Any


def assignment_for_day(db: Any, Assignment: Any, staff_id: int, day: date) -> Any:
    """Load or create the one assignment row for a staff member and date."""
    assignment = Assignment.query.filter_by(staff_id=staff_id, day=day).first()
    if assignment is None:
        assignment = Assignment(staff_id=staff_id, day=day)
        db.session.add(assignment)
    return assignment
