from datetime import date

import pytest

from roster_horizon import (
    get_automatic_recalculation_start,
    get_unit_automatic_recalculation_start,
)


@pytest.mark.parametrize(
    ("reference_date", "expected"),
    (
        (date(2026, 8, 1), date(2026, 11, 1)),
        (date(2026, 8, 31), date(2026, 11, 1)),
        (date(2026, 11, 30), date(2027, 2, 1)),
        (date(2026, 12, 15), date(2027, 3, 1)),
        (date(2028, 2, 29), date(2028, 5, 1)),
    ),
)
def test_default_boundary_uses_whole_calendar_months(reference_date, expected):
    assert get_automatic_recalculation_start(reference_date) == expected


def test_configurable_protection_value_changes_boundary():
    assert get_automatic_recalculation_start(
        date(2026, 8, 17), protected_roster_months_ahead=0
    ) == date(2026, 9, 1)
    assert get_automatic_recalculation_start(
        date(2026, 8, 17), protected_roster_months_ahead=5
    ) == date(2027, 2, 1)


def test_negative_protection_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        get_automatic_recalculation_start(
            date(2026, 8, 1), protected_roster_months_ahead=-1
        )


def test_unit_policy_and_timezone_are_used():
    unit = type("UnitPolicy", (), {
        "protected_roster_months_ahead": 3,
        "timezone": "Europe/London",
    })()
    assert get_unit_automatic_recalculation_start(
        unit, date(2026, 8, 31)
    ) == date(2026, 12, 1)
