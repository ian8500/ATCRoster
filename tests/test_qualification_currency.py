from datetime import date

from atcroster.qualifications.currency import (
    currency_window,
    load_currency_requirement,
)


DEFAULTS = {
    "enabled": False,
    "period_type": "rolling_days",
    "period_days": 90,
    "period_months": 3,
    "hours_per_ue": 10,
    "ojti_credit_percent": 25,
    "start_date": "",
}


def test_currency_requirement_bounds_untrusted_settings():
    requirement = load_currency_requirement(
        7,
        current_unit_id=lambda: 1,
        settings_snapshot=lambda unit_id: {
            "currency": (
                '{"enabled": true, "period_days": 9999, '
                '"hours_per_ue": 0, "ojti_credit_percent": 150}'
            )
        },
        setting_key="currency",
        defaults=DEFAULTS,
    )
    assert requirement["enabled"] is True
    assert requirement["period_days"] == 731
    assert requirement["hours_per_ue"] == 0.25
    assert requirement["ojti_credit_percent"] == 100


def test_currency_window_supports_calendar_months_and_configured_start():
    requirement = {
        **DEFAULTS,
        "period_type": "calendar_months",
        "period_months": 3,
        "start_date": "2026-06-15",
    }
    assert currency_window(requirement, date(2026, 8, 13)) == (
        date(2026, 6, 15),
        date(2026, 8, 13),
    )


def test_currency_window_supports_rolling_days():
    requirement = {**DEFAULTS, "period_days": 30}
    assert currency_window(requirement, date(2026, 8, 13)) == (
        date(2026, 7, 15),
        date(2026, 8, 13),
    )
