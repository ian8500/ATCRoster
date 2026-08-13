"""Route ownership for training and competency workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
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
class TrainingDependencies:
    db: Any
    Staff: Any
    TrainingLevel: Any
    TrainingSession: Any
    TrainingScore: Any
    current_unit_id: Callable
    training_enabled: Callable
    is_editor_user: Callable
    can_manage_training: Callable
    can_record_training: Callable
    is_under_training: Callable
    training_profile_allowed: Callable
    validate_csrf: Callable
    QualificationType: Any
    PersonQualification: Any
    competency_enabled: Callable
    is_admin_user: Callable
    utcnow: Callable
    record_qualification_history: Callable
    sync_qualification_to_roster_profile: Callable
    record_qualification_roster_impact: Callable
    TrainingObjective: Any


def create_training_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> TrainingDependencies:
    """Bind training routes to canonical operational and SaaS models."""
    return TrainingDependencies(
        db=db,
        Staff=operational_models.Staff,
        TrainingLevel=operational_models.TrainingLevel,
        TrainingSession=operational_models.TrainingSession,
        TrainingScore=operational_models.TrainingScore,
        QualificationType=saas_models.QualificationType,
        PersonQualification=saas_models.PersonQualification,
        TrainingObjective=operational_models.TrainingObjective,
        **services,
    )


def create_training_blueprint(dependencies: TrainingDependencies) -> Blueprint:
    blueprint = Blueprint("training", __name__)

    @login_required
    def training_home():
        unit_id = dependencies.current_unit_id()
        if not dependencies.training_enabled(unit_id):
            abort(404)
        own_sessions = dependencies.TrainingSession.query.filter_by(
            unit_id=unit_id, trainee_id=current_user.id
        ).all()
        own_minutes = sum(row.duration_minutes for row in own_sessions)
        can_view_people = bool(
            dependencies.is_editor_user(current_user)
            or dependencies.can_manage_training(current_user)
            or dependencies.can_record_training(current_user)
        )
        people = []
        if can_view_people:
            people = (
                dependencies.Staff.query.filter_by(unit_id=unit_id, is_operational=True)
                .filter(dependencies.Staff.role != "position_monitor")
                .order_by(dependencies.Staff.name)
                .all()
            )
            people = [
                person for person in people if dependencies.is_under_training(person)
            ]
        trainee_count = sum(
            1 for person in people if dependencies.is_under_training(person)
        )
        return render_template(
            "training_home.html",
            people=people,
            own_minutes=own_minutes,
            can_view_people=can_view_people,
            trainee_count=trainee_count,
            own_under_training=dependencies.is_under_training(current_user),
        )

    @login_required
    def training_profile(sid):
        unit_id = dependencies.current_unit_id()
        if not dependencies.training_enabled(unit_id):
            abort(404)
        person = dependencies.Staff.query.filter_by(
            id=sid, unit_id=unit_id
        ).first_or_404()
        if not dependencies.training_profile_allowed(person):
            abort(403)
        if not dependencies.is_under_training(person):
            abort(404)
        if request.method == "POST":
            dependencies.validate_csrf()
            if not dependencies.can_record_training(
                current_user
            ) or not dependencies.is_under_training(person):
                abort(403)
            level = dependencies.TrainingLevel.query.filter_by(
                id=int(request.form.get("level_id") or 0),
                unit_id=unit_id,
                is_active=True,
            ).first_or_404()
            try:
                training_date = date.fromisoformat(
                    request.form.get("training_date") or ""
                )
                duration_minutes = int(request.form.get("duration_minutes") or 0)
            except (TypeError, ValueError):
                abort(400, "Enter a valid training date and duration.")
            if not 1 <= duration_minutes <= 1440:
                abort(400, "Training duration must be between 1 and 1,440 minutes.")
            session_row = dependencies.TrainingSession(
                unit_id=unit_id,
                trainee_id=person.id,
                ojti_id=current_user.id,
                level_id=level.id,
                training_date=training_date,
                duration_minutes=duration_minutes,
                summary=(request.form.get("summary") or "").strip()[:4000],
            )
            dependencies.db.session.add(session_row)
            dependencies.db.session.flush()
            for objective in sorted(level.objectives, key=lambda row: row.position)[
                :15
            ]:
                raw_attainment = request.form.get(f"attainment_{objective.id}")
                raw_assistance = request.form.get(f"assistance_{objective.id}")
                if not raw_attainment and not raw_assistance:
                    continue
                try:
                    attainment = int(raw_attainment or 0)
                    assistance = int(raw_assistance or 0)
                except ValueError:
                    abort(400, "Objective scores must be whole numbers.")
                if attainment not in {1, 2, 3, 4} or assistance not in {1, 2, 3, 4}:
                    abort(400, "Objective scores must be between 1 and 4.")
                dependencies.db.session.add(
                    dependencies.TrainingScore(
                        unit_id=unit_id,
                        session_id=session_row.id,
                        objective_id=objective.id,
                        attainment=attainment,
                        assistance=assistance,
                        safety_critical=bool(
                            request.form.get(f"safety_{objective.id}")
                        ),
                        note=(request.form.get(f"note_{objective.id}") or "").strip()[
                            :4000
                        ],
                    )
                )
            dependencies.db.session.commit()
            flash("Training report saved.", "ok")
            return redirect(url_for("training_profile", sid=person.id, level=level.id))

        levels = (
            dependencies.TrainingLevel.query.filter_by(unit_id=unit_id, is_active=True)
            .order_by(
                dependencies.TrainingLevel.sort_order, dependencies.TrainingLevel.name
            )
            .all()
        )
        selected_id = request.args.get("level", type=int)
        selected = next(
            (row for row in levels if row.id == selected_id),
            levels[0] if levels else None,
        )
        sessions = []
        if selected:
            sessions = (
                dependencies.TrainingSession.query.filter_by(
                    unit_id=unit_id, trainee_id=person.id, level_id=selected.id
                )
                .order_by(
                    dependencies.TrainingSession.training_date.desc(),
                    dependencies.TrainingSession.id.desc(),
                )
                .all()
            )
        total_minutes = sum(
            row.duration_minutes
            for row in dependencies.TrainingSession.query.filter_by(
                unit_id=unit_id, trainee_id=person.id
            ).all()
        )
        return render_template(
            "training_profile.html",
            person=person,
            levels=levels,
            selected_level=selected,
            sessions=sessions,
            total_minutes=total_minutes,
            under_training=dependencies.is_under_training(person),
            can_record=dependencies.can_record_training(current_user),
        )

    @login_required
    def competency_home():
        unit_id = dependencies.current_unit_id()
        if not dependencies.competency_enabled(unit_id):
            abort(404)
        can_view_people = bool(
            dependencies.is_editor_user(current_user)
            or dependencies.can_manage_training(current_user)
            or dependencies.can_record_training(current_user)
        )
        people = []
        if can_view_people:
            people = (
                dependencies.Staff.query.filter_by(unit_id=unit_id, is_operational=True)
                .filter(dependencies.Staff.role != "position_monitor")
                .order_by(dependencies.Staff.name)
                .all()
            )
        return render_template(
            "competency_home.html",
            people=people,
            can_view_people=can_view_people,
        )

    @login_required
    def competency_profile(sid):
        unit_id = dependencies.current_unit_id()
        if not dependencies.competency_enabled(unit_id):
            abort(404)
        person = dependencies.Staff.query.filter_by(
            id=sid, unit_id=unit_id
        ).first_or_404()
        if not dependencies.training_profile_allowed(person):
            abort(403)
        can_edit = bool(
            dependencies.is_admin_user(current_user)
            or getattr(current_user, "has_assessor", False)
        )
        qualification_types = (
            dependencies.QualificationType.query.filter_by(
                unit_id=unit_id, is_active=True
            )
            .order_by(dependencies.QualificationType.label)
            .all()
        )
        if request.method == "POST":
            dependencies.validate_csrf()
            if not can_edit:
                abort(403)
            person.caa_license_number = (
                request.form.get("caa_license_number") or ""
            ).strip()[:40]
            for qtype in qualification_types:
                raw = (request.form.get(f"expiry_{qtype.id}") or "").strip()
                try:
                    expires_on = date.fromisoformat(raw) if raw else None
                except ValueError:
                    abort(400, "Enter valid competency expiry dates.")
                record = dependencies.PersonQualification.query.filter_by(
                    unit_id=unit_id,
                    person_id=person.id,
                    qualification_type_id=qtype.id,
                ).first()
                old_state = (
                    record.status if record else None,
                    record.valid_from if record else None,
                    record.expires_on if record else None,
                )
                if not record and expires_on:
                    record = dependencies.PersonQualification(
                        unit_id=unit_id,
                        person_id=person.id,
                        qualification_type_id=qtype.id,
                        status="valid",
                    )
                    dependencies.db.session.add(record)
                    dependencies.db.session.flush()
                if record:
                    record.expires_on = expires_on
                    record.updated_at = dependencies.utcnow()
                    dependencies.record_qualification_history(
                        record, "competency_updated"
                    )
                dependencies.sync_qualification_to_roster_profile(
                    person, qtype, expires_on
                )
                if record:
                    dependencies.record_qualification_roster_impact(
                        person, qtype, *old_state, record,
                        reason=f"{qtype.code} competency profile updated.",
                    )
            dependencies.db.session.commit()
            flash("Competency profile updated everywhere.", "ok")
            return redirect(url_for("competency_profile", sid=person.id))
        qualification_rows = dependencies.PersonQualification.query.filter_by(
            unit_id=unit_id, person_id=person.id
        ).all()
        return render_template(
            "competency_profile.html",
            person=person,
            qualification_types=qualification_types,
            qualifications={
                row.qualification_type_id: row for row in qualification_rows
            },
            can_edit=can_edit,
        )

    @login_required
    def training_admin():
        if not dependencies.training_enabled(dependencies.current_unit_id()):
            abort(404)
        if not dependencies.is_admin_user(current_user):
            abort(403)
        unit_id = dependencies.current_unit_id()
        if request.method == "POST":
            dependencies.validate_csrf()
            action = request.form.get("action")
            if action == "create_level":
                name = (request.form.get("name") or "").strip()
                if not name:
                    abort(400, "Enter a level name.")
                level = dependencies.TrainingLevel(unit_id=unit_id, name=name[:80])
                dependencies.db.session.add(level)
                dependencies.db.session.flush()
                for position in range(1, 16):
                    dependencies.db.session.add(
                        dependencies.TrainingObjective(
                            unit_id=unit_id,
                            level_id=level.id,
                            position=position,
                            title=f"Objective {position}",
                            description="Configure this objective.",
                        )
                    )
            elif action == "save_objectives":
                level = dependencies.TrainingLevel.query.filter_by(
                    id=int(request.form.get("level_id") or 0), unit_id=unit_id
                ).first_or_404()
                for objective in level.objectives:
                    objective.title = (
                        request.form.get(f"title_{objective.id}") or objective.title
                    ).strip()[:100]
                    objective.description = (
                        request.form.get(f"description_{objective.id}") or ""
                    ).strip()[:4000]
            else:
                abort(400, "Unknown training administration action.")
            dependencies.db.session.commit()
            flash("Training configuration saved.", "ok")
            return redirect(url_for("training_admin"))
        levels = (
            dependencies.TrainingLevel.query.filter_by(unit_id=unit_id)
            .order_by(
                dependencies.TrainingLevel.sort_order, dependencies.TrainingLevel.name
            )
            .all()
        )
        return render_template("training_admin.html", levels=levels)

    @login_required
    def training_analytics():
        if not dependencies.training_enabled(dependencies.current_unit_id()):
            abort(404)
        if not dependencies.can_manage_training(current_user):
            abort(403)
        unit_id = dependencies.current_unit_id()
        sessions = (
            dependencies.TrainingSession.query.filter_by(unit_id=unit_id)
            .order_by(dependencies.TrainingSession.training_date)
            .all()
        )
        scores = dependencies.TrainingScore.query.filter_by(unit_id=unit_id).all()
        objective_totals = defaultdict(list)
        for score in scores:
            objective_totals[score.objective].append(score.attainment)
        objective_analytics = sorted(
            (
                {
                    "objective": objective,
                    "average": round(sum(values) / len(values), 2),
                    "count": len(values),
                }
                for objective, values in objective_totals.items()
            ),
            key=lambda row: (row["objective"].level_id, row["objective"].position),
        )
        ojti_minutes = defaultdict(int)
        for row in sessions:
            ojti_minutes[row.ojti] += row.duration_minutes
        return render_template(
            "training_analytics.html",
            sessions=sessions,
            objective_analytics=objective_analytics,
            ojti_hours=sorted(ojti_minutes.items(), key=lambda item: item[0].name),
        )

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/training/", "training_home", training_home, ["GET"]),
            (
                "/training/<int:sid>",
                "training_profile",
                training_profile,
                ["GET", "POST"],
            ),
            ("/competency/", "competency_home", competency_home, ["GET"]),
            (
                "/competency/<int:sid>",
                "competency_profile",
                competency_profile,
                ["GET", "POST"],
            ),
            (
                "/training/admin",
                "training_admin",
                training_admin,
                ["GET", "POST"],
            ),
            (
                "/training/analytics",
                "training_analytics",
                training_analytics,
                ["GET"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
