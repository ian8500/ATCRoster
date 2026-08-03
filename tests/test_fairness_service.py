from datetime import date, time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fairness_service import FairnessAssignment, FairnessStaff, calculate_fairness


def test_fairness_targets_are_proportional_and_exclude_night_ineligible_staff():
    people = [
        FairnessStaff(1, "Full time", 4800, eligible_nights=True),
        FairnessStaff(2, "Part time", 2400, eligible_nights=True),
        FairnessStaff(3, "No nights", 4800, eligible_nights=False),
    ]
    assignments = [
        FairnessAssignment(1, date(2026, 7, 4), "LATE", 480, time(21), is_night=True),
        FairnessAssignment(1, date(2026, 7, 5), "N", 480, time(21)),
        FairnessAssignment(2, date(2026, 7, 6), "N", 480, time(21)),
        FairnessAssignment(3, date(2026, 7, 7), "M", 480, time(7)),
    ]

    rows = {row.staff_id: row for row in calculate_fairness(people, assignments)}

    assert rows[1].target_night_count == pytest.approx(2.0)
    assert rows[2].target_night_count == pytest.approx(1.0)
    assert rows[3].target_night_count == 0
    assert rows[1].target_weekend_count == pytest.approx(0.8)
    assert rows[3].early_count == 1


def test_fairness_reports_hours_deviation_preferences_and_audit_counts():
    people = [FairnessStaff(7, "Controller", 480)]
    assignments = [
        FairnessAssignment(7, date(2026, 7, 1), "A", 600, time(13))
    ]
    rows = calculate_fairness(
        people,
        assignments,
        expected_code_for=lambda _staff_id, _day: "M",
        preference_breach_for=lambda _staff_id, _day, code: code == "A",
        manual_change_counts={7: 2},
    )

    row = rows[0]
    assert row.actual_minutes == 600
    assert row.target_minutes == 480
    assert row.difference_minutes == 120
    assert row.overtime_minutes == 120
    assert row.pattern_deviations == 1
    assert row.preference_breaches == 1
    assert row.manual_roster_changes == 2


def test_fairness_handles_zero_expected_minutes_without_division_error():
    rows = calculate_fairness(
        [FairnessStaff(1, "No contracted duties", 0, eligible_nights=False)],
        [],
    )
    assert rows[0].contracted_ratio == 0
    assert rows[0].target_night_count == 0
