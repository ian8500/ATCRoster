"""Shared time calculations for Live Position reporting and currency checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil


def minutes_between(start: datetime, end: datetime) -> int:
    """Return whole elapsed minutes, never yielding a negative duration."""
    return max(0, round((end - start).total_seconds() / 60))


@dataclass(frozen=True)
class PositionRecoveryPolicy:
    base_break_minutes: int = 30
    escalation_after_minutes: int = 120
    extra_break_minutes: int = 15
    escalation_interval_minutes: int = 60
    escalation_cap_minutes: int = 240


def required_recovery_seconds(
    accrued_seconds: int, policy: PositionRecoveryPolicy
) -> int:
    """Return the completed break needed to reset accrued position time."""
    threshold = policy.escalation_after_minutes * 60
    capped = min(max(0, accrued_seconds), policy.escalation_cap_minutes * 60)
    excess = max(0, capped - threshold)
    increments = (
        ceil(excess / (policy.escalation_interval_minutes * 60)) if excess else 0
    )
    return (policy.base_break_minutes + increments * policy.extra_break_minutes) * 60


def cumulative_position_seconds(
    intervals: list[tuple[datetime, datetime]], policy: PositionRecoveryPolicy
) -> tuple[int, int]:
    """Accumulate duty across breaks shorter than the recovery requirement."""
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    accrued = 0
    previous_end: datetime | None = None
    for start, end in merged:
        if previous_end is not None:
            gap = max(0, int((start - previous_end).total_seconds()))
            if gap >= required_recovery_seconds(accrued, policy):
                accrued = 0
        accrued += max(0, int((end - start).total_seconds()))
        previous_end = end
    return accrued, required_recovery_seconds(accrued, policy)
