"""Route ownership for operations assurance and planning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import json
import re
from typing import Any, Callable

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required


@dataclass(frozen=True)
class OperationsDependencies:
    db: Any
    OperationalPosition: Any
    PositionEndorsement: Any
    PositionRequirement: Any
    Staff: Any
    ShiftType: Any
    BreakPlan: Any
    Assignment: Any
    AchievedDuty: Any
    FatigueReport: Any
    RosterRuleVersion: Any
    is_admin_user: Callable
    compliance_month: Callable
    validate_csrf: Callable
    current_unit_id: Callable
    utcnow: Callable
    log_change: Callable
    month_add: Callable
    position_assurance: Callable
    coverage_heatmap: Callable
    scenarios_page: Callable


def create_operations_blueprint(dependencies: OperationsDependencies) -> Blueprint:
    blueprint = Blueprint("operations", __name__)

    @login_required
    def operations_assurance(ym):
        if not dependencies.is_admin_user(current_user):
            abort(403)
        year, month = dependencies.compliance_month(ym)
        if request.method == "POST":
            dependencies.validate_csrf()
            action = (request.form.get("action") or "").strip()
            try:
                if action == "create_position":
                    code = (request.form.get("code") or "").strip().upper()
                    label = (request.form.get("label") or "").strip()
                    if not re.fullmatch(r"[A-Z0-9_-]{2,30}", code) or not label:
                        raise ValueError("Position code and label are required.")
                    dependencies.db.session.add(
                        dependencies.OperationalPosition(
                            unit_id=dependencies.current_unit_id(),
                            code=code,
                            label=label,
                            description=(request.form.get("description") or "").strip()[
                                :1000
                            ],
                            is_safety_critical=request.form.get("is_safety_critical")
                            == "on",
                        )
                    )
                elif action == "grant_endorsement":
                    person_id = int(request.form.get("person_id") or 0)
                    position_id = int(request.form.get("position_id") or 0)
                    person = dependencies.Staff.query.filter_by(
                        id=person_id, is_operational=True
                    ).first_or_404()
                    position = dependencies.OperationalPosition.query.filter_by(
                        id=position_id
                    ).first_or_404()
                    row = dependencies.PositionEndorsement.query.filter_by(
                        person_id=person.id, position_id=position.id
                    ).first()
                    if not row:
                        row = dependencies.PositionEndorsement(
                            unit_id=dependencies.current_unit_id(),
                            person_id=person.id,
                            position_id=position.id,
                        )
                        dependencies.db.session.add(row)
                    row.valid_from = date.fromisoformat(request.form["valid_from"])
                    valid_until = (request.form.get("valid_until") or "").strip()
                    row.valid_until = (
                        date.fromisoformat(valid_until) if valid_until else None
                    )
                    row.status = "valid"
                    row.restrictions = (request.form.get("restrictions") or "").strip()[
                        :1000
                    ]
                elif action == "set_position_requirement":
                    position_id = int(request.form.get("position_id") or 0)
                    dependencies.OperationalPosition.query.filter_by(
                        id=position_id, is_active=True
                    ).first_or_404()
                    duty_day = date.fromisoformat(request.form["day"])
                    shift_code = (request.form.get("shift_code") or "").strip().upper()
                    dependencies.ShiftType.query.filter_by(
                        code=shift_code, is_active=True
                    ).first_or_404()
                    required = max(0, int(request.form.get("required_count") or 0))
                    contingency = max(
                        0, int(request.form.get("contingency_count") or 0)
                    )
                    row = dependencies.PositionRequirement.query.filter_by(
                        day=duty_day, shift_code=shift_code, position_id=position_id
                    ).first()
                    if not row:
                        row = dependencies.PositionRequirement(
                            unit_id=dependencies.current_unit_id(),
                            day=duty_day,
                            shift_code=shift_code,
                            position_id=position_id,
                        )
                        dependencies.db.session.add(row)
                    row.required_count = required
                    row.contingency_count = contingency
                elif action == "add_break":
                    duty_day = date.fromisoformat(request.form["day"])
                    start_time = time.fromisoformat(request.form["start_time"])
                    end_time = time.fromisoformat(request.form["end_time"])
                    if end_time <= start_time:
                        raise ValueError("Break end must be after its start.")
                    person_id = int(request.form.get("person_id") or 0)
                    dependencies.Staff.query.filter_by(
                        id=person_id, is_operational=True
                    ).first_or_404()
                    position_id = int(request.form.get("position_id") or 0) or None
                    if position_id:
                        dependencies.OperationalPosition.query.filter_by(
                            id=position_id
                        ).first_or_404()
                    dependencies.db.session.add(
                        dependencies.BreakPlan(
                            unit_id=dependencies.current_unit_id(),
                            day=duty_day,
                            person_id=person_id,
                            position_id=position_id,
                            start_time=start_time,
                            end_time=end_time,
                            kind=(request.form.get("kind") or "break")[:20],
                            recorded_by_id=current_user.id,
                        )
                    )
                elif action == "record_actual":
                    duty_day = date.fromisoformat(request.form["day"])
                    person_id = int(request.form.get("person_id") or 0)
                    dependencies.Staff.query.filter_by(
                        id=person_id, is_operational=True
                    ).first_or_404()
                    actual_start = datetime.fromisoformat(request.form["actual_start"])
                    actual_end = datetime.fromisoformat(request.form["actual_end"])
                    if actual_end <= actual_start:
                        raise ValueError("Actual duty end must be after its start.")
                    assignment = dependencies.Assignment.query.filter_by(
                        staff_id=person_id, day=duty_day
                    ).first()
                    row = dependencies.AchievedDuty.query.filter_by(
                        person_id=person_id, day=duty_day
                    ).first()
                    if not row:
                        row = dependencies.AchievedDuty(
                            unit_id=dependencies.current_unit_id(),
                            person_id=person_id,
                            day=duty_day,
                            recorded_by_id=current_user.id,
                        )
                        dependencies.db.session.add(row)
                    row.planned_assignment_id = assignment.id if assignment else None
                    row.actual_start = actual_start
                    row.actual_end = actual_end
                    row.duty_type = (request.form.get("duty_type") or "operational")[
                        :30
                    ]
                    row.variance_reason = (
                        request.form.get("variance_reason") or ""
                    ).strip()[:500]
                elif action == "review_fatigue":
                    report = dependencies.FatigueReport.query.filter_by(
                        id=int(request.form.get("report_id") or 0)
                    ).first_or_404()
                    response = (request.form.get("manager_response") or "").strip()
                    if len(response) < 10:
                        raise ValueError("Record the assessment and action taken.")
                    report.manager_response = response[:1000]
                    report.status = (
                        request.form.get("status")
                        if request.form.get("status") in {"reviewed", "closed"}
                        else "reviewed"
                    )
                    report.reviewed_by_id = current_user.id
                    report.reviewed_at = dependencies.utcnow()
                    report.closed_at = (
                        dependencies.utcnow() if report.status == "closed" else None
                    )
                elif action == "create_rule_version":
                    latest = (
                        dependencies.db.session.query(
                            dependencies.db.func.max(
                                dependencies.RosterRuleVersion.version
                            )
                        )
                        .filter(
                            dependencies.RosterRuleVersion.unit_id
                            == dependencies.current_unit_id()
                        )
                        .scalar()
                        or 0
                    )
                    rules = request.form.get("rules_json") or "{}"
                    parsed = json.loads(rules)
                    if not isinstance(parsed, dict):
                        raise ValueError("Rules must be a JSON object.")
                    dependencies.db.session.add(
                        dependencies.RosterRuleVersion(
                            unit_id=dependencies.current_unit_id(),
                            version=latest + 1,
                            name=(request.form.get("name") or f"Rule set {latest + 1}")[
                                :120
                            ],
                            rules_json=json.dumps(parsed),
                            change_reference=(
                                request.form.get("change_reference") or ""
                            )[:120],
                            consultation_summary=(
                                request.form.get("consultation_summary") or ""
                            )[:2000],
                        )
                    )
                elif action == "approve_rule_version":
                    rule = dependencies.RosterRuleVersion.query.filter_by(
                        id=int(request.form.get("rule_id") or 0), state="draft"
                    ).first_or_404()
                    if not rule.change_reference or len(rule.consultation_summary) < 20:
                        raise ValueError(
                            "Approval requires a change reference and consultation summary."
                        )
                    dependencies.RosterRuleVersion.query.filter_by(
                        unit_id=dependencies.current_unit_id(), state="approved"
                    ).update(
                        {"state": "superseded"},
                        synchronize_session=False,
                    )
                    rule.state = "approved"
                    rule.effective_from = date.fromisoformat(
                        request.form["effective_from"]
                    )
                    rule.approved_by_id = current_user.id
                    rule.approved_at = dependencies.utcnow()
                else:
                    abort(400)
                dependencies.db.session.commit()
                dependencies.log_change(
                    "OperationalAssurance",
                    0,
                    action,
                    None,
                    "completed",
                    context_day=date(year, month, 1),
                )
                flash("Operational assurance record saved.", "ok")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                dependencies.db.session.rollback()
                flash(str(exc), "error")
            return redirect(url_for("operations_assurance", ym=ym))

        positions = (
            dependencies.OperationalPosition.query.filter_by(is_active=True)
            .order_by(dependencies.OperationalPosition.code)
            .all()
        )
        staff = (
            dependencies.Staff.query.filter_by(is_operational=True)
            .order_by(dependencies.Staff.name)
            .all()
        )
        endorsements = dependencies.PositionEndorsement.query.order_by(
            dependencies.PositionEndorsement.valid_until
        ).all()
        breaks = (
            dependencies.BreakPlan.query.filter(
                dependencies.BreakPlan.day >= date(year, month, 1),
                dependencies.BreakPlan.day
                < date(*dependencies.month_add(year, month, 1), 1),
            )
            .order_by(dependencies.BreakPlan.day, dependencies.BreakPlan.start_time)
            .all()
        )
        actuals = (
            dependencies.AchievedDuty.query.filter(
                dependencies.AchievedDuty.day >= date(year, month, 1),
                dependencies.AchievedDuty.day
                < date(*dependencies.month_add(year, month, 1), 1),
            )
            .order_by(dependencies.AchievedDuty.day.desc())
            .all()
        )
        reports = (
            dependencies.FatigueReport.query.filter(
                dependencies.FatigueReport.status.in_(("open", "reviewed"))
            )
            .order_by(dependencies.FatigueReport.reported_at.desc())
            .all()
        )
        rules = dependencies.RosterRuleVersion.query.order_by(
            dependencies.RosterRuleVersion.version.desc()
        ).all()
        assurance = dependencies.position_assurance(year, month)
        return render_template(
            "operations_assurance.html",
            ym=ym,
            year=year,
            month=month,
            positions=positions,
            staff=staff,
            endorsements=endorsements,
            breaks=breaks,
            actuals=actuals,
            reports=reports,
            rules=rules,
            assurance=assurance,
            staff_by_id={row.id: row for row in staff},
            positions_by_id={row.id: row for row in positions},
            shifts=dependencies.ShiftType.query.filter_by(is_active=True)
            .order_by(dependencies.ShiftType.code)
            .all(),
        )

    @blueprint.record_once
    def register_routes(state):
        routes = (
            (
                "/operations/<ym>",
                "operations_assurance",
                operations_assurance,
                ["GET", "POST"],
            ),
            (
                "/planning/coverage/<ym>",
                "coverage_heatmap",
                dependencies.coverage_heatmap,
                ["GET"],
            ),
            (
                "/planning/scenarios",
                "scenarios_page",
                dependencies.scenarios_page,
                ["GET", "POST"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
