"""Token-authenticated calendar subscription feed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

import secrets

from flask import Blueprint, Response, abort, flash, redirect, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class CalendarFeedDependencies:
    Staff: Any
    Assignment: Any
    get_shift: Callable[..., Any]
    db: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]


def create_calendar_feed_dependencies(
    *, db: Any, operational_models: Any, **services: Any
) -> CalendarFeedDependencies:
    """Bind calendar routes to canonical operational models."""
    return CalendarFeedDependencies(
        db=db,
        Staff=operational_models.Staff,
        Assignment=operational_models.Assignment,
        **services,
    )


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

    @login_required
    def calendar_token_create(sid: int):
        dependencies.validate_csrf()
        staff = dependencies.Staff.query.filter_by(
            id=sid, unit_id=dependencies.current_unit_id(),
        ).first_or_404()
        if staff.id != current_user.id and not dependencies.is_admin_user(current_user):
            abort(403)
        staff.calendar_token = secrets.token_hex(24)
        dependencies.db.session.commit()
        flash("A new private calendar subscription link was generated.", "ok")
        return redirect(url_for("staff_profile", sid=staff.id) + "#calendar")

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/calendar/<int:sid>/<token>.ics", "calendar_feed", calendar_feed, methods=("GET",))
        state.app.add_url_rule("/staff/<int:sid>/calendar-token", "calendar_token_create", calendar_token_create, methods=("POST",))

    return blueprint


def register_calendar_feed_blueprint(
    app: Any, *, db: Any, operational_models: Any, **services: Any
) -> None:
    """Register calendar routes from calendar-owned dependencies."""
    app.register_blueprint(create_calendar_feed_blueprint(
        create_calendar_feed_dependencies(
            db=db, operational_models=operational_models, **services
        )
    ))
