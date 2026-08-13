"""Roster pattern parsing and validation adapters."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def expand(
    raw_value: str | None, expand_pattern: Callable[[str | None], list[str]]
) -> list[str]:
    return expand_pattern(raw_value)


def validate(
    raw_value: str | None, validated_pattern: Callable[[str | None], list[str]]
) -> list[str]:
    return validated_pattern(raw_value)


def unit_pattern_context(
    unit_id, *, settings_snapshot, validate_pattern, default_pattern
):
    settings = settings_snapshot(unit_id)
    pattern = validate_pattern(settings.get("base_pattern_csv") or default_pattern)
    try:
        anchor = date.fromisoformat(settings.get("base_pattern_anchor") or "2025-01-01")
    except ValueError:
        anchor = date(2025, 1, 1)
    return pattern or validate_pattern(default_pattern), anchor


def pattern_context(
    staff,
    on_date,
    *,
    db,
    StaffWatchHistory,
    effective_watch,
    validate_pattern,
    unit_context,
):
    """Resolve personal, effective-watch, then airport pattern precedence."""
    if staff.pattern_override:
        personal = validate_pattern(staff.pattern_csv)
        if personal:
            return personal, staff.pattern_anchor or on_date
    unit_pattern, unit_anchor = unit_context(staff.unit_id)
    watch = effective_watch(staff, on_date)
    if not watch:
        return unit_pattern, unit_anchor
    move = (
        StaffWatchHistory.query.filter(
            StaffWatchHistory.unit_id == staff.unit_id,
            StaffWatchHistory.staff_id == staff.id,
            StaffWatchHistory.effective_date <= on_date,
            db.or_(
                StaffWatchHistory.effective_to.is_(None),
                StaffWatchHistory.effective_to >= on_date,
            ),
        )
        .order_by(StaffWatchHistory.effective_date.desc())
        .first()
    )
    watch_pattern = validate_pattern(watch.pattern_csv)
    return (
        watch_pattern or unit_pattern,
        (move.pattern_anchor if move and move.pattern_anchor else None)
        or watch.pattern_anchor
        or unit_anchor,
    )


def night_active_on(unit_id, on_date, *, settings_snapshot):
    raw = settings_snapshot(unit_id).get("night_active_weekdays", "0,1,2,3,4,5,6")
    try:
        active_days = {int(value) for value in raw.split(",") if value.strip()}
    except ValueError:
        active_days = set(range(7))
    return on_date.weekday() in active_days


def leave_code_for(staff: Any, on_date: date):
    for leave in staff.leaves:
        if leave.start <= on_date <= leave.end:
            return leave.leave_type
    return None


def code_for_day(staff, on_date, *, resolve_context, night_active):
    pattern, anchor = resolve_context(staff, on_date)
    if not pattern:
        return "OFF"
    code = pattern[(on_date - anchor).days % len(pattern)]
    return "OFF" if code == "N" and not night_active(staff.unit_id, on_date) else code
