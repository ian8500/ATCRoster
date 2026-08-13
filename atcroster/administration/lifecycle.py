"""Staff lifecycle administration routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, request, url_for
from flask_login import login_required


@dataclass(frozen=True)
class StaffLifecycleDependencies:
    db: Any
    Staff: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    parse_date: Callable[[str | None], date | None]
    record_roster_impact: Callable[..., None]
    admin_required: Callable[[Callable[..., Any]], Callable[..., Any]]


def create_staff_lifecycle_dependencies(
    *, db: Any, operational_models: Any, roster_impact_event_type: Any,
    **services: Any,
) -> StaffLifecycleDependencies:
    return StaffLifecycleDependencies(
        db=db, Staff=operational_models.Staff,
        RosterImpactEventType=roster_impact_event_type, **services,
    )


def create_staff_lifecycle_blueprint(
    dependencies: StaffLifecycleDependencies,
) -> Blueprint:
    blueprint = Blueprint("staff_lifecycle", __name__)

    @login_required
    @dependencies.admin_required
    def admin_staff_leaving(sid: int):
        person = dependencies.Staff.query.filter_by(
            id=sid, unit_id=dependencies.current_unit_id()
        ).first_or_404()
        action = (request.form.get("action") or "schedule").strip()
        if action == "cancel":
            restore_from = (
                person.final_operational_duty_date
                or person.final_unit_date
                or date.today()
            ) + timedelta(days=1)
            person.final_unit_date = person.final_operational_duty_date = None
            person.employment_end_date = None
            person.leaving_reason_category = person.leaving_notes = ""
            dependencies.record_roster_impact(
                dependencies.RosterImpactEventType.RETURN_TO_UNIT,
                restore_from,
                staff_ids=[person.id],
                rebuild_baseline=True,
                reason=(request.form.get("reason") or "Leaving event cancelled.")[:500],
            )
            dependencies.db.session.commit()
            flash("Leaving event cancelled and future baseline restored.", "ok")
            return redirect(url_for("admin_staff_edit", sid=person.id))
        try:
            final_unit = date.fromisoformat(request.form["final_unit_date"])
            final_operational = date.fromisoformat(
                request.form.get("final_operational_duty_date")
                or final_unit.isoformat()
            )
            employment_end = dependencies.parse_date(
                request.form.get("employment_end_date")
            )
        except (KeyError, ValueError):
            abort(400, "Enter valid leaving dates.")
        if final_operational > final_unit or (
            employment_end and employment_end < final_unit
        ):
            abort(400, "Final operational duty must not follow the final unit date.")
        category = (request.form.get("reason_category") or "OTHER").strip().upper()
        if category not in {
            "TRANSFER",
            "RETIREMENT",
            "RESIGNATION",
            "END_OF_CONTRACT",
            "OTHER",
        }:
            abort(400, "Invalid leaving reason category.")
        person.final_unit_date = final_unit
        person.final_operational_duty_date = final_operational
        person.employment_end_date = employment_end
        person.leaving_reason_category = category
        person.leaving_notes = (request.form.get("notes") or "").strip()[:2000]
        dependencies.record_roster_impact(
            dependencies.RosterImpactEventType.UNIT_LEAVER,
            final_operational + timedelta(days=1),
            staff_ids=[person.id],
            rebuild_baseline=True,
            reason=f"Unit leaver: {category}. {person.leaving_notes}"[:500],
        )
        dependencies.db.session.commit()
        flash("Leaving date saved and future roster baseline updated.", "ok")
        return redirect(url_for("admin_staff_edit", sid=person.id))

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin/staff/<int:sid>/leaving",
            "admin_staff_leaving",
            admin_staff_leaving,
            methods=("POST",),
        )

    return blueprint
