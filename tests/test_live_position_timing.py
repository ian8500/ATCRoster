from datetime import datetime, timedelta

from live_position_timing import minutes_between


def test_minutes_between_rounds_elapsed_time_and_rejects_negative_durations():
    start = datetime(2026, 8, 13, 9, 0)

    assert minutes_between(start, start + timedelta(seconds=89)) == 1
    assert minutes_between(start, start + timedelta(seconds=90)) == 2
    assert minutes_between(start, start - timedelta(minutes=1)) == 0
