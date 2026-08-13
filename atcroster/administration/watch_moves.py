"""Effective-dated staff watch transfer routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, request, url_for
from flask_login import current_user


@dataclass(frozen=True)
class WatchMoveDependencies:
    db: Any
    Staff: Any
    Watch: Any
    StaffWatchHistory: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    record_roster_impact: Callable[..., None]
    log_change: Callable[..., None]


def create_watch_move_dependencies(
    *, db: Any, operational_models: Any, roster_impact_event_type: Any,
    **services: Any,
) -> WatchMoveDependencies:
    return WatchMoveDependencies(
        db=db, Staff=operational_models.Staff, Watch=operational_models.Watch,
        StaffWatchHistory=operational_models.StaffWatchHistory,
        RosterImpactEventType=roster_impact_event_type, **services,
    )


def create_watch_move_blueprint(dependencies: WatchMoveDependencies) -> Blueprint:
    blueprint = Blueprint("staff_watch_moves", __name__)
    db = dependencies.db
    Staff = dependencies.Staff
    Watch = dependencies.Watch
    StaffWatchHistory = dependencies.StaffWatchHistory
    RosterImpactEventType = dependencies.RosterImpactEventType
    _current_unit_id = dependencies.current_unit_id
    is_admin_user = dependencies.is_admin_user
    record_roster_impact = dependencies.record_roster_impact
    log_change = dependencies.log_change

    def admin_watch_move(sid):
        if not is_admin_user(current_user):
            abort(403)
        s = (
            Staff.query.filter_by(id=sid, unit_id=_current_unit_id())
            .filter(Staff.role != "position_monitor")
            .first_or_404()
        )
        watch_id_val = request.form.get("watch_id")
        eff = (request.form.get("effective_date") or "").strip()

        if not watch_id_val or not eff:
            flash("Watch and effective date are required.", "error")
            return redirect(url_for("admin_staff_edit", sid=s.id))

        try:
            new_watch_id = int(watch_id_val)
        except (TypeError, ValueError):
            flash("Invalid watch selection.", "error")
            return redirect(url_for("admin_staff_edit", sid=s.id))

        try:
            eff_d = date.fromisoformat(eff)
        except ValueError:
            flash("Invalid effective date.", "error")
            return redirect(url_for("admin_staff_edit", sid=s.id))

        new_watch = Watch.query.filter_by(
            id=new_watch_id, unit_id=_current_unit_id()
        ).first()
        if not new_watch:
            flash("Invalid watch selection.", "error")
            return redirect(url_for("admin_staff_edit", sid=s.id))
        alignment_mode = (
            (request.form.get("alignment_mode") or "ALIGN_WITH_DESTINATION_WATCH")
            .strip()
            .upper()
        )
        if alignment_mode not in {
            "ALIGN_WITH_DESTINATION_WATCH",
            "SELECT_STARTING_CYCLE_DAY",
        }:
            abort(400, "Invalid watch alignment mode.")
        try:
            starting_cycle_day = (
                int(request.form.get("starting_cycle_day") or 0)
                if alignment_mode == "SELECT_STARTING_CYCLE_DAY"
                else None
            )
        except ValueError:
            abort(400, "Starting cycle day must be a number.")
        if starting_cycle_day is not None and starting_cycle_day < 0:
            abort(400, "Starting cycle day cannot be negative.")
        previous = (
            StaffWatchHistory.query.filter(
                StaffWatchHistory.unit_id == s.unit_id,
                StaffWatchHistory.staff_id == s.id,
                StaffWatchHistory.effective_date < eff_d,
                db.or_(
                    StaffWatchHistory.effective_to.is_(None),
                    StaffWatchHistory.effective_to >= eff_d,
                ),
            )
            .order_by(StaffWatchHistory.effective_date.desc())
            .first()
        )
        if previous:
            previous.effective_to = eff_d - timedelta(days=1)
        existing = StaffWatchHistory.query.filter_by(
            unit_id=_current_unit_id(),
            staff_id=s.id,
            effective_date=eff_d,
        ).first()
        if existing:
            existing.watch_id = new_watch_id
            move = existing
        else:
            move = StaffWatchHistory(
                unit_id=_current_unit_id(),
                staff_id=s.id,
                watch_id=new_watch_id,
                effective_date=eff_d,
            )
            db.session.add(move)
        move.reason = (request.form.get("reason") or "").strip()[:500]
        move.alignment_mode = alignment_mode
        move.starting_cycle_day = starting_cycle_day
        move.pattern_anchor = (
            eff_d - timedelta(days=starting_cycle_day)
            if starting_cycle_day is not None
            else None
        )
        old_watch_id = s.watch_id
        if eff_d <= date.today():
            s.watch_id = new_watch_id
        record_roster_impact(
            RosterImpactEventType.WATCH_TRANSFER,
            eff_d,
            staff_ids=[s.id],
            rebuild_baseline=True,
            reason=f"Watch transfer to {new_watch.name} recorded.",
        )
        db.session.commit()

        log_change(
            "Staff",
            s.id,
            "watch_id",
            old_watch_id,
            new_watch_id,
            note=f"effective {eff_d.isoformat()}",
        )
        flash(
            f"Watch move recorded. {s.name} follows {new_watch.name}'s "
            f"pattern from {eff_d.strftime('%d %b %Y')}.",
            "ok",
        )
        return redirect(url_for("admin_staff_edit", sid=s.id))

    def admin_watch_move_edit(hid):
        if not is_admin_user(current_user):
            abort(403)

        hist = StaffWatchHistory.query.filter_by(
            id=hid, unit_id=_current_unit_id()
        ).first_or_404()
        watch_id_val = request.form.get("watch_id")
        eff = (request.form.get("effective_date") or "").strip()

        if not watch_id_val or not eff:
            flash("Watch and effective date are required.", "error")
            return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

        try:
            new_watch_id = int(watch_id_val)
        except (TypeError, ValueError):
            flash("Invalid watch selection.", "error")
            return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

        try:
            eff_d = date.fromisoformat(eff)
        except ValueError:
            flash("Invalid effective date.", "error")
            return redirect(url_for("admin_staff_edit", sid=hist.staff_id))
        if not Watch.query.filter_by(
            id=new_watch_id, unit_id=_current_unit_id()
        ).first():
            flash("Invalid watch selection.", "error")
            return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

        old_watch_id = hist.watch_id
        old_eff = hist.effective_date

        hist.watch_id = new_watch_id
        hist.effective_date = eff_d
        record_roster_impact(
            RosterImpactEventType.WATCH_TRANSFER,
            min(old_eff, eff_d),
            staff_ids=[hist.staff_id],
            rebuild_baseline=True,
            reason="Effective-dated watch transfer changed.",
        )
        db.session.commit()

        if old_watch_id != new_watch_id:
            log_change(
                "StaffWatchHistory", hist.id, "watch_id", old_watch_id, new_watch_id
            )
        if old_eff != eff_d:
            log_change("StaffWatchHistory", hist.id, "effective_date", old_eff, eff_d)

        flash("Watch move updated.", "ok")
        return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

    def admin_watch_move_delete(hid):
        if not is_admin_user(current_user):
            abort(403)

        hist = StaffWatchHistory.query.filter_by(
            id=hid, unit_id=_current_unit_id()
        ).first_or_404()
        sid = hist.staff_id
        old_watch_id = hist.watch_id
        old_eff = hist.effective_date

        db.session.delete(hist)
        record_roster_impact(
            RosterImpactEventType.WATCH_TRANSFER,
            old_eff,
            staff_ids=[sid],
            rebuild_baseline=True,
            reason="Scheduled watch transfer removed.",
        )
        db.session.commit()

        log_change(
            "StaffWatchHistory",
            hid,
            "delete",
            old_watch_id,
            None,
            note=f"effective {old_eff.isoformat()}",
        )
        flash("Watch move deleted.", "ok")
        return redirect(url_for("admin_staff_edit", sid=sid))

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule(
            "/admin/staff/<int:sid>/watch-move",
            "admin_watch_move",
            admin_watch_move,
            methods=("POST",),
        )
        state.app.add_url_rule(
            "/admin/staff/watch-move/<int:hid>/edit",
            "admin_watch_move_edit",
            admin_watch_move_edit,
            methods=("POST",),
        )
        state.app.add_url_rule(
            "/admin/staff/watch-move/<int:hid>/delete",
            "admin_watch_move_delete",
            admin_watch_move_delete,
            methods=("POST",),
        )

    return blueprint
