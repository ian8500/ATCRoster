"""Token-authenticated calendar subscription feed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

from flask import Blueprint, Response, abort


@dataclass(frozen=True)
class CalendarFeedDependencies:
    Staff: Any
    Assignment: Any
    get_shift: Callable[[str, int], Any]


def _calendar_window_today() -> tuple[date, date]:
    today = date.today()
    current_start = date(today.year, today.month, 1)
    next_month = (today.month % 12) + 1
    next_year = today.year + (today.month == 12)
    current_end = (current_start.replace(day=28) + timedelta(days=10)).replace(day=1) - timedelta(days=1)
    next_end = (date(next_year, next_month, 1).replace(day=28) + timedelta(days=10)).replace(day=1) - timedelta(days=1)
    return current_start, next_end if today.day >= 20 else current_end


def _ical_escape(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def create_calendar_feed_blueprint(dependencies: CalendarFeedDependencies) -> Blueprint:
    """Create the public token-credential calendar route."""
    blueprint = Blueprint("calendar_feed", __name__)

    def calendar_feed(sid: int, token: str):
        staff = dependencies.Staff.query.filter_by(id=sid, calendar_token=token).first_or_404()
        if not staff.calendar_token or token != staff.calendar_token:
            abort(403)
        start, end = _calendar_window_today()
        assignments = dependencies.Assignment.query.filter(
            dependencies.Assignment.staff_id == staff.id,
            dependencies.Assignment.day >= start,
            dependencies.Assignment.day <= end,
        ).order_by(dependencies.Assignment.day.asc()).all()
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//ATC Roster//EN", "CALSCALE:GREGORIAN", f"X-WR-CALNAME:{_ical_escape(staff.name)} Roster"]
        for assignment in assignments:
            shift = dependencies.get_shift(assignment.code, unit_id=staff.unit_id)
            lines.extend(("BEGIN:VEVENT", f"UID:{staff.id}-{assignment.day.isoformat()}-{assignment.code}@atcroster"))
            summary = assignment.code + (f" ({assignment.annotation})" if assignment.annotation else "")
            lines.append(f"SUMMARY:{_ical_escape(summary)}")
            if shift and shift.start_time and shift.end_time and shift.is_working:
                started = datetime.combine(assignment.day, shift.start_time)
                ended = datetime.combine(assignment.day, shift.end_time)
                if shift.end_time <= shift.start_time:
                    ended += timedelta(days=1)
                lines.extend((f"DTSTART:{started.strftime('%Y%m%dT%H%M%S')}", f"DTEND:{ended.strftime('%Y%m%dT%H%M%S')}"))
            else:
                lines.extend((f"DTSTART;VALUE=DATE:{assignment.day.strftime('%Y%m%d')}", f"DTEND;VALUE=DATE:{(assignment.day + timedelta(days=1)).strftime('%Y%m%d')}"))
            lines.append("END:VEVENT")
        lines.append("END:VCALENDAR")
        return Response("\r\n".join(lines).encode("utf-8"), mimetype="text/calendar; charset=utf-8")

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/calendar/<int:sid>/<token>.ics", "calendar_feed", calendar_feed, methods=("GET",))

    return blueprint
