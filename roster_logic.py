"""Pure roster and assignment calculations used by Flask route handlers."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Iterable, Iterator


PATTERN_CODES = ("M", "A", "D", "N", "OFF")


def month_days(year: int, month: int) -> tuple[date, list[date]]:
    start = date(year, month, 1)
    stop = date(year + (month // 12), (month % 12) + 1, 1)
    return start, [
        start + timedelta(days=offset) for offset in range((stop - start).days)
    ]


def parse_year_month(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def expand_pattern(raw_value: str | None) -> list[str]:
    """Expand a CSV shift pattern, including legacy multiplier notation."""
    raw = [part.strip() for part in (raw_value or "").split(",") if part.strip()]
    expanded = []
    for token in raw:
        token = token.upper()
        prefix_multiplier = re.match(r"^\s*(\d+)\s*[X\*]\s*([A-Z]+)\s*$", token)
        suffix_multiplier = re.match(r"^\s*([A-Z]+)\s*[X\*]\s*(\d+)\s*$", token)
        if prefix_multiplier:
            count, code = int(prefix_multiplier.group(1)), prefix_multiplier.group(2)
            expanded.extend([code] * count)
        elif suffix_multiplier:
            code, count = suffix_multiplier.group(1), int(suffix_multiplier.group(2))
            expanded.extend([code] * count)
        else:
            expanded.append(token)
    return expanded


def validated_pattern(raw_value: str | None) -> list[str]:
    values = expand_pattern(raw_value)
    if not values or any(value not in PATTERN_CODES for value in values):
        return []
    return values


def shift_minutes(shift) -> int:
    if not shift or not shift.start_time or not shift.end_time:
        return 0
    start = datetime.combine(date(2000, 1, 1), shift.start_time)
    end = datetime.combine(date(2000, 1, 1), shift.end_time)
    if end <= start:
        end += timedelta(days=1)
    return int((end - start).total_seconds() // 60)


def daily_requirements(requirement, day: date, special=None) -> dict[str, int]:
    """Return the effective M/D/A/N staffing requirement for a roster date."""
    source = special or requirement
    if not source:
        return {code: 0 for code in ("M", "D", "A", "N")}
    prefix = ""
    if not special:
        prefix = (
            "sat_" if day.weekday() == 5 else ("sun_" if day.weekday() == 6 else "")
        )
    return {
        code: max(0, int(getattr(source, f"req_{prefix}{code.lower()}", 0) or 0))
        for code in ("M", "D", "A", "N")
    }


def iter_year_months(start_day: date, end_day: date) -> Iterator[tuple[int, int]]:
    year, month = start_day.year, start_day.month
    last = date(end_day.year, end_day.month, 1)
    current = date(year, month, 1)
    while current <= last:
        yield year, month
        month += 1
        if month == 13:
            month = 1
            year += 1
        current = date(year, month, 1)


def add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def roster_lock_date(year: int, month: int) -> date:
    lock_year, lock_month = add_months(year, month, -2)
    return date(lock_year, lock_month, 20)


def roster_month_is_locked(year: int, month: int, today: date | None = None) -> bool:
    return (today or date.today()) >= roster_lock_date(year, month)


def normalise_assignment_snapshot(rows: Iterable[dict[str, object]]) -> list[tuple]:
    return sorted(
        (
            int(row["staff_id"]),
            str(row["day"]),
            str(row.get("code") or ""),
            str(row.get("annotation") or ""),
        )
        for row in rows
    )
