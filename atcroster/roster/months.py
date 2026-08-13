"""Configured roster-month operations used by routes and services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .existence import month_has_data
from .locking import lock_roster_month


@dataclass(frozen=True)
class RosterMonthService:
    """Bind generic roster-month policy to an application's persistence layer."""

    db: Any
    Assignment: Any
    Requirement: Any
    add_months: Callable[[int, int, int], tuple[int, int]]
    days_for_month: Callable[[int, int], tuple[Any, list[Any]]]
    parse_year_month: Callable[[str], tuple[int, int]]
    ensure_month_requirement: Callable[[int, int], Any]

    def has_data(self, year: int, month: int) -> bool:
        return month_has_data(self.db, self.Assignment, year, month, self.add_months)

    def lock(self, unit_id: int, year: int, month: int) -> Any:
        return lock_roster_month(
            self.db,
            self.Requirement,
            unit_id,
            year,
            month,
            self.ensure_month_requirement,
        )

    def range(self, year: int, month: int) -> tuple[Any, list[Any]]:
        return self.days_for_month(year, month)

    def parse(self, value: str) -> tuple[int, int]:
        return self.parse_year_month(value)
