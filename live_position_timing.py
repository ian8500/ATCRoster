"""Shared time calculations for Live Position reporting and currency checks."""

from __future__ import annotations

from datetime import datetime


def minutes_between(start: datetime, end: datetime) -> int:
    """Return whole elapsed minutes, never yielding a negative duration."""
    return max(0, round((end - start).total_seconds() / 60))
