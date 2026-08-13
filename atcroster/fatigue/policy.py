"""Fail-safe fatigue admission policy."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def assignment_is_fatigue_safe(staff: Any, day: date, code: str, findings: Callable[[Any, date, str], list[Any]]) -> bool:
    """Block a proposed assignment when fatigue analysis fails or finds risk."""
    try:
        return not findings(staff, day, code)
    except Exception:
        return False
