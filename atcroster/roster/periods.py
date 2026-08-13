"""Roster month arithmetic and lock policy."""

from __future__ import annotations

from datetime import date
from typing import Callable


def month_add(year: int, month: int, delta: int, add_months: Callable[[int, int, int], tuple[int, int]]) -> tuple[int, int]:
    return add_months(year, month, delta)


def lock_date_for_month(year: int, month: int, roster_lock_date: Callable[[int, int], date]) -> date:
    return roster_lock_date(year, month)


def is_month_locked(year: int, month: int, today: date | None, roster_month_is_locked: Callable[[int, int, date | None], bool]) -> bool:
    return roster_month_is_locked(year, month, today)
