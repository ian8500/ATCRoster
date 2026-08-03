"""Whole-month boundaries for deterministic roster maintenance."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from roster_logic import add_months


DEFAULT_PROTECTED_ROSTER_MONTHS_AHEAD = 2


def get_automatic_recalculation_start(
    reference_date: date | None = None,
    *,
    protected_roster_months_ahead: int = DEFAULT_PROTECTED_ROSTER_MONTHS_AHEAD,
    timezone_name: str = "Europe/London",
) -> date:
    """Return the first day after the protected whole-month horizon.

    A value of two protects the current calendar month plus the following two
    months, so automatic maintenance begins on the first day of month three.
    When no date is supplied, the unit's local calendar date is used.
    """
    months_ahead = int(protected_roster_months_ahead)
    if months_ahead < 0:
        raise ValueError("Protected roster months ahead cannot be negative.")
    local_date = reference_date or datetime.now(ZoneInfo(timezone_name)).date()
    year, month = add_months(
        local_date.year,
        local_date.month,
        months_ahead + 1,
    )
    return date(year, month, 1)


def get_unit_automatic_recalculation_start(
    unit,
    reference_date: date | None = None,
) -> date:
    """Resolve the boundary using a unit's stored policy and timezone."""
    return get_automatic_recalculation_start(
        reference_date,
        protected_roster_months_ahead=getattr(
            unit,
            "protected_roster_months_ahead",
            DEFAULT_PROTECTED_ROSTER_MONTHS_AHEAD,
        ),
        timezone_name=getattr(unit, "timezone", "Europe/London"),
    )
