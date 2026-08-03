"""Roster-impact exception queue and recalculation audit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datetime import date, timedelta

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
    Unit: Any
    Assignment: Any
    RosterPeriod: Any
    current_unit_id: Callable[[], int]
    can_edit_roster: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    utcnow: Callable[[], Any]
    is_admin_user: Callable[[Any], bool]
    population_service: Any
    impact_service: Callable[[], Any]
    automatic_boundary: Callable[[Any, date | None], date]


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

    @blueprint.get("/preview")
    @login_required
    def preview():
        unit_id = require_access()
        unit = dependencies.db.session.get(dependencies.Unit, unit_id)
        today = date.today()
        boundary = dependencies.automatic_boundary(unit, today)
        effective_from = _date_arg("effective_from", boundary)
        effective_to = _date_arg("effective_to", effective_from + timedelta(days=30))
        if effective_to < effective_from:
            abort(400, "Impact end date cannot precede its start date.")
        staff_id = request.args.get("staff_id", type=int)
        watch_id = request.args.get("watch_id", type=int)
        staff_ids = (staff_id,) if staff_id else ()
        watch_ids = (watch_id,) if watch_id else ()
        query = dependencies.Assignment.query.filter(
            dependencies.Assignment.unit_id == unit_id,
            dependencies.Assignment.day >= effective_from,
            dependencies.Assignment.day <= effective_to,
        )
        if staff_id:
            query = query.filter_by(staff_id=staff_id)
        elif watch_id:
            staff_for_watch = dependencies.Staff.query.filter_by(
                unit_id=unit_id, watch_id=watch_id
            ).with_entities(dependencies.Staff.id)
            query = query.filter(dependencies.Assignment.staff_id.in_(staff_for_watch))
        affected_assignments = query.count()
        overrides_to_preserve = query.filter(
            dependencies.Assignment.override_code.isnot(None)
        ).count()
        dry_run = dependencies.population_service.populate_or_recalculate_baseline(
            unit_id, effective_from, effective_to, staff_ids=staff_ids,
            watch_ids=watch_ids, mode="event", reference_date=today, dry_run=True,
        )
        protected_end = min(effective_to, boundary - timedelta(days=1)) \
            if effective_from < boundary else None
        automatic_from = max(effective_from, boundary) if effective_to >= boundary else None
        periods = dependencies.RosterPeriod.query.filter_by(unit_id=unit_id).order_by(
            dependencies.RosterPeriod.year, dependencies.RosterPeriod.month
        ).all()
        return render_template(
            "roster_impact/preview.html", unit=unit,
            effective_from=effective_from, effective_to=effective_to,
            boundary=boundary, protected_end=protected_end,
            automatic_from=automatic_from,
            affected_assignments=affected_assignments,
            overrides_to_preserve=overrides_to_preserve, dry_run=dry_run,
            staff=dependencies.Staff.query.filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.role != "position_monitor",
            ).order_by(dependencies.Staff.name).all(),
            watches=dependencies.Watch.query.filter_by(unit_id=unit_id).order_by(
                dependencies.Watch.order_index
            ).all(), selected_staff_id=staff_id, selected_watch_id=watch_id,
            periods=periods, is_admin=dependencies.is_admin_user(current_user),
        )

    @blueprint.post("/recalculate")
    @login_required
    def recalculate():
        unit_id = require_access()
        dependencies.validate_csrf()
        unit = dependencies.db.session.get(dependencies.Unit, unit_id)
        start, end, reason, staff_ids, watch_ids = _impact_form(request)
        if start < dependencies.automatic_boundary(unit, date.today()):
            abort(400, "Ordinary recalculation cannot modify a protected roster period.")
        dependencies.impact_service().handle_roster_impact_event(
            unit_id, "MANUAL_RECALCULATION", start, effective_to=end,
            staff_ids=staff_ids, watch_ids=watch_ids, rebuild_baseline=True,
            reason=reason, triggered_by_user_id=getattr(current_user, "id", None),
        )
        dependencies.db.session.commit()
        flash("Automatic roster range recalculated; editor overrides were preserved.", "ok")
        return redirect(url_for("roster_impact.exceptions"))

    @blueprint.post("/protected-rebuild")
    @login_required
    def protected_rebuild():
        require_access()
        if not dependencies.is_admin_user(current_user):
            abort(403)
        dependencies.validate_csrf()
        if (request.form.get("confirmation") or "").strip().upper() != "REBUILD":
            abort(400, "Type REBUILD to confirm a protected-period rebuild.")
        start, end, reason, staff_ids, watch_ids = _impact_form(request)
        dependencies.impact_service().handle_roster_impact_event(
            dependencies.current_unit_id(), "MANUAL_RECALCULATION", start,
            effective_to=end, staff_ids=staff_ids, watch_ids=watch_ids,
            rebuild_baseline=True, reason=reason,
            triggered_by_user_id=getattr(current_user, "id", None),
            allow_protected_rebuild=True,
        )
        dependencies.db.session.commit()
        flash("Protected baseline rebuilt; all editor overrides were preserved.", "ok")
        return redirect(url_for("roster_impact.exceptions"))

    return blueprint


def _date_arg(name: str, default: date) -> date:
    raw = request.args.get(name)
    return date.fromisoformat(raw) if raw else default


def _impact_form(req: Any) -> tuple[date, date, str, tuple[int, ...], tuple[int, ...]]:
    start = date.fromisoformat(req.form["effective_from"])
    end = date.fromisoformat(req.form["effective_to"])
    if end < start:
        abort(400, "Impact end date cannot precede its start date.")
    reason = (req.form.get("reason") or "").strip()[:500]
    if not reason:
        abort(400, "A reason is required.")
    staff_id = req.form.get("staff_id", type=int)
    watch_id = req.form.get("watch_id", type=int)
    if staff_id and watch_id:
        abort(400, "Choose either a staff member or a watch, not both.")
    return start, end, reason, (staff_id,) if staff_id else (), (watch_id,) if watch_id else ()
