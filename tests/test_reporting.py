from datetime import date
from types import SimpleNamespace

import pytest

from reporting import (
    csv_safe_cell,
    current_leave_year_window,
    financial_year_start,
    group_consecutive_days,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            '=HYPERLINK("https://attacker.invalid")',
            '\'=HYPERLINK("https://attacker.invalid")',
        ),
        (" +SUM(1,1)", "' +SUM(1,1)"),
        ("\t@cmd", "'\t@cmd"),
        ("Normal name", "Normal name"),
        (42, 42),
    ],
)
def test_csv_safe_cell_neutralises_spreadsheet_formulas(value, expected):
    assert csv_safe_cell(value) == expected


def test_financial_year_start_handles_both_sides_of_april_boundary():
    assert financial_year_start(date(2026, 3, 31)) == date(2025, 4, 1)
    assert financial_year_start(date(2026, 4, 1)) == date(2026, 4, 1)


def test_group_consecutive_days_counts_distinct_periods():
    assert group_consecutive_days([]) == 0
    assert (
        group_consecutive_days([date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 4)])
        == 2
    )


def test_current_leave_year_window_uses_configured_start_month():
    person = SimpleNamespace(leave_year_start_month=4)

    assert current_leave_year_window(person, date(2026, 3, 31)) == (
        date(2025, 4, 1),
        date(2026, 3, 31),
    )
    assert current_leave_year_window(person, date(2026, 4, 1)) == (
        date(2026, 4, 1),
        date(2027, 3, 31),
    )
