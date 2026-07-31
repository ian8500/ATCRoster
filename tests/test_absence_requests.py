from datetime import date
from types import SimpleNamespace

from absence_requests import (
    group_sickness_instances,
    normalise_request_rules,
    request_date_bounds,
    request_lock_date,
    request_month_is_locked,
    safe_admin_month,
)


def _sickness(staff_id, day, code="SC"):
    return SimpleNamespace(
        staff_id=staff_id,
        staff=SimpleNamespace(id=staff_id),
        day=day,
        code=code,
    )


def test_sickness_instances_group_consecutive_days_per_person():
    instances = group_sickness_instances(
        [
            _sickness(1, date(2026, 7, 2), "SSC"),
            _sickness(2, date(2026, 7, 1)),
            _sickness(1, date(2026, 7, 1)),
            _sickness(1, date(2026, 7, 4)),
        ]
    )
    assert [(item["staff_id"], item["duration"]) for item in instances] == [
        (1, 2),
        (1, 1),
        (2, 1),
    ]
    assert instances[0]["codes"] == ["SC", "SSC"]


def test_request_rules_are_clamped_to_safe_ranges():
    assert normalise_request_rules(0, 0) == (3, 20)
    assert normalise_request_rules(99, 31) == (24, 28)


def test_request_window_starts_next_month_and_crosses_year_boundary():
    assert request_date_bounds(date(2026, 11, 20), 3) == (
        date(2026, 12, 1),
        date(2027, 2, 28),
    )


def test_request_lock_date_and_boundary_cross_year():
    assert request_lock_date(2027, 1, 20) == date(2026, 12, 20)
    assert request_month_is_locked(2027, 1, 20, date(2026, 12, 20)) is True
    assert request_month_is_locked(2027, 1, 20, date(2026, 12, 19)) is False


def test_safe_admin_month_rejects_malformed_redirect_values():
    fallback = date(2026, 7, 31)
    assert safe_admin_month("2027-01", fallback) == "2027-01"
    assert safe_admin_month("//external.example", fallback) == "2026-07"
    assert safe_admin_month("2027-13", fallback) == "2026-07"
