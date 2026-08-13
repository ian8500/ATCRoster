"""Roster form date and time parsing."""

from __future__ import annotations

from datetime import date, time


def parse_hhmm(value: str | None) -> time | None:
    """Parse an optional strict hour-and-minute form field."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        hour, minute = value.split(":")
        return time(int(hour), int(minute))
    except (TypeError, ValueError):
        return None


def parse_iso_date(value: str | None) -> date | None:
    """Parse an optional ISO calendar-date form field."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
