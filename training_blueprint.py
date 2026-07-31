"""Route ownership for training and competency workflows."""

from __future__ import annotations

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
    competency_home: Callable
    competency_profile: Callable
    training_admin: Callable
    training_analytics: Callable


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
            except TypeError, ValueError:
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
            ("/competency/", "competency_home", dependencies.competency_home, ["GET"]),
            (
                "/competency/<int:sid>",
                "competency_profile",
                dependencies.competency_profile,
                ["GET", "POST"],
            ),
            (
                "/training/admin",
                "training_admin",
                dependencies.training_admin,
                ["GET", "POST"],
            ),
            (
                "/training/analytics",
                "training_analytics",
                dependencies.training_analytics,
                ["GET"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
