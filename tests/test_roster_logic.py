from datetime import date, time
from types import SimpleNamespace

from roster_logic import (
    daily_requirements,
    expand_pattern,
    iter_year_months,
    month_days,
    normalise_assignment_snapshot,
    roster_month_is_locked,
    shift_minutes,
    validated_pattern,
)


def test_month_days_handles_leap_year():
    start, days = month_days(2028, 2)
    assert start == date(2028, 2, 1)
    assert len(days) == 29
    assert days[-1] == date(2028, 2, 29)


def test_patterns_support_legacy_multipliers_and_reject_unknown_codes():
    assert expand_pattern("2xM, A*2, off") == ["M", "M", "A", "A", "OFF"]
    assert validated_pattern("M,A,N,OFF") == ["M", "A", "N", "OFF"]
    assert validated_pattern("M,UNKNOWN") == []


def test_shift_minutes_handles_overnight_duty():
    shift = SimpleNamespace(start_time=time(22, 0), end_time=time(6, 0))
    assert shift_minutes(shift) == 480


def test_daily_requirements_select_weekend_values_and_clamp_negatives():
    requirement = SimpleNamespace(
        req_m=4,
        req_d=4,
        req_a=4,
        req_n=2,
        req_sat_m=3,
        req_sat_d=-1,
        req_sat_a=3,
        req_sat_n=1,
    )
    assert daily_requirements(requirement, date(2026, 8, 1)) == {
        "M": 3,
        "D": 0,
        "A": 3,
        "N": 1,
    }


def test_year_month_iteration_and_lock_boundary():
    assert list(iter_year_months(date(2026, 11, 4), date(2027, 2, 8))) == [
        (2026, 11),
        (2026, 12),
        (2027, 1),
        (2027, 2),
    ]
    assert roster_month_is_locked(2026, 8, date(2026, 6, 20)) is True
    assert roster_month_is_locked(2026, 8, date(2026, 6, 19)) is False


def test_assignment_snapshot_normalisation_is_order_independent():
    first = {"staff_id": 2, "day": "2026-08-02", "code": "M"}
    second = {"staff_id": 1, "day": "2026-08-01", "code": "A", "annotation": None}
    assert normalise_assignment_snapshot(
        [first, second]
    ) == normalise_assignment_snapshot([second, first])
