"""Pure absence grouping and shift-request date rules."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Iterable


def group_sickness_instances(assignments: Iterable, month_start=None, month_end=None):
    """Group consecutive sickness assignments into display-ready instances."""
    instances = []
    current = None
    for assignment in sorted(assignments, key=lambda row: (row.staff_id, row.day)):
        continues = (
            current
            and current["staff_id"] == assignment.staff_id
            and assignment.day == current["end"] + timedelta(days=1)
        )
        if not continues:
            current = {
                "staff_id": assignment.staff_id,
                "staff": assignment.staff,
                "start": assignment.day,
                "end": assignment.day,
                "days": [],
            }
            instances.append(current)
        current["days"].append(assignment)
        current["end"] = assignment.day
    if month_start and month_end:
        instances = [
            instance
            for instance in instances
            if instance["end"] >= month_start and instance["start"] <= month_end
        ]
    for instance in instances:
        instance["duration"] = (instance["end"] - instance["start"]).days + 1
        instance["codes"] = list(dict.fromkeys(day.code for day in instance["days"]))
    return instances


def normalise_request_rules(months_ahead, lock_day) -> tuple[int, int]:
    months = max(1, min(int(months_ahead or 3), 24))
    day = max(1, min(int(lock_day or 20), 28))
    return months, day


def add_months(first: date, count: int) -> date:
    index = first.year * 12 + first.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def request_lock_date(year: int, month: int, lock_day: int) -> date:
    previous_month = month - 1
    previous_year = year
    if previous_month <= 0:
        previous_month = 12
        previous_year -= 1
    return date(previous_year, previous_month, lock_day)


def request_month_is_locked(
    year: int,
    month: int,
    lock_day: int,
    today: date | None = None,
) -> bool:
    return (today or date.today()) >= request_lock_date(year, month, lock_day)


def request_date_bounds(today: date, months_ahead: int) -> tuple[date, date]:
    start = add_months(date(today.year, today.month, 1), 1)
    return start, add_months(start, months_ahead) - timedelta(days=1)


def safe_admin_month(raw_value: str | None, fallback: date) -> str:
    """Return a canonical admin month without allowing malformed redirects."""
    candidate = (raw_value or "").strip()
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", candidate):
        return f"{fallback.year:04d}-{fallback.month:02d}"
    return candidate
