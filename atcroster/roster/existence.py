"""Efficient roster-period data checks."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def month_has_data(db: Any, Assignment: Any, year: int, month: int, add_months: Callable[[int, int, int], tuple[int, int]]) -> bool:
    """Return whether any assignment exists in the requested roster month."""
    start = date(year, month, 1)
    next_year, next_month = add_months(year, month, 1)
    end = date(next_year, next_month, 1)
    return db.session.query(Assignment.id).filter(
        Assignment.day >= start, Assignment.day < end,
    ).limit(1).first() is not None
