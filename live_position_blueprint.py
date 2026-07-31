"""HTTP boundary for the Live Position Monitoring module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, jsonify, render_template
from flask_login import current_user, login_required


@dataclass(frozen=True)
class LivePositionDependencies:
    db: Any
    Unit: Any
    OperationalPosition: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    Staff: Any
    utcnow: Callable[[], Any]
    is_admin_user: Callable[[Any], bool]


def create_live_position_blueprint(
    dependencies: LivePositionDependencies,
) -> Blueprint:
    blueprint = Blueprint("live_position", __name__, url_prefix="/live-positions")

    def _unit_id() -> int:
        return int(getattr(current_user, "unit_id", 0) or 0)

    def _require_kiosk_or_admin() -> None:
        if not (
            getattr(current_user, "role", "") == "position_monitor"
            or dependencies.is_admin_user(current_user)
        ):
            abort(403)

    @blueprint.get("/")
    @login_required
    def admin_home():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        return render_template("live_position/admin_home.html")

    @blueprint.get("/kiosk")
    @login_required
    def kiosk_hmi():
        _require_kiosk_or_admin()
        unit = dependencies.db.session.get(dependencies.Unit, _unit_id())
        return render_template("live_position/kiosk.html", unit=unit)

    @blueprint.get("/api/state")
    @login_required
    def live_state():
        _require_kiosk_or_admin()
        unit_id = _unit_id()
        now = dependencies.utcnow()
        positions = (
            dependencies.OperationalPosition.query
            .filter_by(unit_id=unit_id, is_active=True)
            .order_by(
                dependencies.OperationalPosition.display_order,
                dependencies.OperationalPosition.code,
            ).all()
        )
        state = []
        for position in positions:
            status_event = (
                dependencies.PositionStatusEvent.query
                .filter_by(unit_id=unit_id, position_id=position.id)
                .order_by(
                    dependencies.PositionStatusEvent.occurred_at.desc(),
                    dependencies.PositionStatusEvent.id.desc(),
                ).first()
            )
            session = (
                dependencies.PositionSession.query
                .filter_by(
                    unit_id=unit_id, position_id=position.id,
                    ended_at=None, is_void=False,
                ).first()
            )
            primary = (
                dependencies.db.session.get(
                    dependencies.Staff, session.primary_person_id
                ) if session else None
            )
            participants = []
            if session:
                for row in dependencies.PositionSessionParticipant.query.filter_by(
                    unit_id=unit_id, session_id=session.id, ended_at=None
                ).all():
                    person = dependencies.db.session.get(
                        dependencies.Staff, row.person_id
                    )
                    participants.append({
                        "id": row.id,
                        "person_name": person.name if person else "Unknown",
                        "role_id": row.role_id,
                        "started_at": row.started_at.isoformat() + "Z",
                    })
            physical_status = status_event.status if status_event else "closed"
            display_status = (
                "closed" if physical_status == "closed" else
                session.session_type if session else "vacant"
            )
            state.append({
                "id": position.id, "code": position.code,
                "label": position.label, "physical_status": physical_status,
                "display_status": display_status,
                "primary": ({
                    "id": primary.id, "name": primary.name,
                    "started_at": session.started_at.isoformat() + "Z",
                    "due_off_at": (
                        session.due_off_at.isoformat() + "Z"
                        if session.due_off_at else None
                    ),
                } if primary and session else None),
                "participants": participants,
            })
        return jsonify({
            "server_time": now.isoformat() + "Z", "positions": state,
        })

    return blueprint
