"""Roster-impact exception queue and recalculation audit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


QUEUE_STATUSES = ("OPEN", "ACKNOWLEDGED", "RESOLVED", "NOT_APPLICABLE")


@dataclass(frozen=True)
class RosterImpactBlueprintDependencies:
    db: Any
    RosterImpactEvent: Any
    RosterImpactException: Any
    Staff: Any
    Watch: Any
    current_unit_id: Callable[[], int]
    can_edit_roster: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    utcnow: Callable[[], Any]


def create_roster_impact_blueprint(
    dependencies: RosterImpactBlueprintDependencies,
) -> Blueprint:
    blueprint = Blueprint("roster_impact", __name__, url_prefix="/roster-impact")

    def require_access() -> int:
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        return dependencies.current_unit_id()

    @blueprint.get("/exceptions")
    @login_required
    def exceptions():
        unit_id = require_access()
        status = (request.args.get("status") or "ACTIVE").strip().upper()
        exception_type = (request.args.get("type") or "").strip().upper()
        query = dependencies.RosterImpactException.query.filter_by(unit_id=unit_id)
        if status == "ACTIVE":
            query = query.filter(
                dependencies.RosterImpactException.status.in_(("OPEN", "ACKNOWLEDGED"))
            )
        elif status in QUEUE_STATUSES:
            query = query.filter_by(status=status)
        if exception_type:
            query = query.filter_by(exception_type=exception_type)
        rows = query.order_by(
            dependencies.RosterImpactException.created_at.desc(),
            dependencies.RosterImpactException.id.desc(),
        ).limit(500).all()
        event_ids = {row.event_id for row in rows}
        staff_ids = {row.staff_id for row in rows if row.staff_id}
        watch_ids = {row.watch_id for row in rows if row.watch_id}
        events = {
            row.id: row for row in dependencies.RosterImpactEvent.query.filter(
                dependencies.RosterImpactEvent.unit_id == unit_id,
                dependencies.RosterImpactEvent.id.in_(event_ids),
            ).all()
        } if event_ids else {}
        staff = {
            row.id: row for row in dependencies.Staff.query.filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.id.in_(staff_ids),
            ).all()
        } if staff_ids else {}
        watches = {
            row.id: row for row in dependencies.Watch.query.filter(
                dependencies.Watch.unit_id == unit_id,
                dependencies.Watch.id.in_(watch_ids),
            ).all()
        } if watch_ids else {}
        types = [value for (value,) in dependencies.db.session.query(
            dependencies.RosterImpactException.exception_type
        ).filter_by(unit_id=unit_id).distinct().order_by(
            dependencies.RosterImpactException.exception_type
        ).all()]
        recent_events = dependencies.RosterImpactEvent.query.filter_by(
            unit_id=unit_id
        ).order_by(dependencies.RosterImpactEvent.created_at.desc()).limit(50).all()
        return render_template(
            "roster_impact/exceptions.html", rows=rows, events=events,
            staff_by_id=staff, watches_by_id=watches, statuses=QUEUE_STATUSES,
            selected_status=status, selected_type=exception_type,
            exception_types=types, recent_events=recent_events,
        )

    @blueprint.post("/exceptions/<int:exception_id>/status")
    @login_required
    def update_exception(exception_id: int):
        unit_id = require_access()
        dependencies.validate_csrf()
        row = dependencies.RosterImpactException.query.filter_by(
            unit_id=unit_id, id=exception_id
        ).first_or_404()
        new_status = (request.form.get("status") or "").strip().upper()
        if new_status not in QUEUE_STATUSES:
            abort(400, "Unsupported exception status.")
        note = (request.form.get("resolution_note") or "").strip()[:1000]
        if new_status in {"RESOLVED", "NOT_APPLICABLE"} and not note:
            flash("Add a resolution note before closing an exception.", "error")
            return redirect(url_for("roster_impact.exceptions"))
        row.status = new_status
        row.resolution_note = note
        row.resolved_by_user_id = (
            getattr(current_user, "person_id", None)
            or getattr(current_user, "id", None)
        )
        row.resolved_at = dependencies.utcnow()
        dependencies.db.session.commit()
        flash("Roster-impact exception updated.", "ok")
        return redirect(url_for("roster_impact.exceptions"))

    return blueprint
