from datetime import datetime, timedelta

from live_position_timing import (
    PositionRecoveryPolicy,
    cumulative_position_seconds,
    minutes_between,
    required_recovery_seconds,
)


def test_minutes_between_rounds_elapsed_time_and_rejects_negative_durations():
    start = datetime(2026, 8, 13, 9, 0)

    assert minutes_between(start, start + timedelta(seconds=89)) == 1
    assert minutes_between(start, start + timedelta(seconds=90)) == 2
    assert minutes_between(start, start - timedelta(minutes=1)) == 0


def test_recovery_break_escalates_after_two_hours_and_caps_at_four_hours():
    policy = PositionRecoveryPolicy()

    assert required_recovery_seconds(120 * 60, policy) == 30 * 60
    assert required_recovery_seconds(121 * 60, policy) == 45 * 60
    assert required_recovery_seconds(180 * 60, policy) == 45 * 60
    assert required_recovery_seconds(181 * 60, policy) == 60 * 60
    assert required_recovery_seconds(10 * 60 * 60, policy) == 60 * 60


def test_position_time_survives_short_break_and_resets_after_required_break():
    policy = PositionRecoveryPolicy()
    start = datetime(2026, 8, 15, 8, 0)

    accrued, required = cumulative_position_seconds(
        [
            (start, start + timedelta(hours=1)),
            (start + timedelta(hours=1, minutes=20), start + timedelta(hours=2)),
        ],
        policy,
    )
    assert accrued == 100 * 60
    assert required == 30 * 60

    reset, required_after_reset = cumulative_position_seconds(
        [
            (start, start + timedelta(hours=1)),
            (start + timedelta(hours=1, minutes=30), start + timedelta(hours=2)),
        ],
        policy,
    )
    assert reset == 30 * 60
    assert required_after_reset == 30 * 60
