"""HTTP boundary for the Live Position Monitoring module."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import (
    Blueprint,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash

from live_position_service import (
    LivePositionConflict,
    LivePositionModels,
    LivePositionService,
    LivePositionValidationError,
)


@dataclass(frozen=True)
class LivePositionDependencies:
    db: Any
    Unit: Any
    OperationalPosition: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    PositionParticipantRole: Any
    PositionSessionAudit: Any
    ControllerKioskCredential: Any
    PositionEndorsement: Any
    Staff: Any
    utcnow: Callable[[], Any]
    is_admin_user: Callable[[Any], bool]
    consume_rate_limit: Callable[..., bool]
    reset_rate_limit: Callable[..., None]
    security_event: Callable[..., None]


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

    def _service() -> LivePositionService:
        return LivePositionService(
            dependencies.db,
            LivePositionModels(
                dependencies.OperationalPosition,
                dependencies.PositionStatusEvent,
                dependencies.PositionSession,
                dependencies.PositionSessionParticipant,
                dependencies.PositionSessionAudit,
            ),
            dependencies.utcnow,
        )

    def _payload() -> dict[str, Any]:
        return request.get_json(silent=True) or request.form.to_dict()

    def _request_key(data: dict[str, Any]) -> str:
        return str(
            data.get("request_key") or request.headers.get("Idempotency-Key", "")
        )

    def _audit_identity(
        action: str,
        actor_id: int,
        requested_person_id: int,
    ) -> None:
        dependencies.db.session.add(
            dependencies.PositionSessionAudit(
                unit_id=_unit_id(),
                actor_id=actor_id,
                action=action,
                occurred_at=dependencies.utcnow(),
                new_value_json=(
                    '{"requested_person_id":' + str(requested_person_id) + "}"
                ),
                transaction_key=LivePositionService.transaction_key(),
            )
        )

    def _verify_pin(person_id: int, pin: str) -> Any:
        unit_id = _unit_id()
        now = dependencies.utcnow()
        if not dependencies.consume_rate_limit(
            "controller-kiosk-pin",
            f"{unit_id}:{person_id}",
            limit=5,
            window=timedelta(minutes=15),
            fail_closed=True,
        ):
            dependencies.security_event(
                "controller_kiosk_pin_rate_limited", unit_id=unit_id
            )
            abort(429, "Too many PIN attempts. Try again later.")
        person = dependencies.Staff.query.filter_by(
            id=person_id, unit_id=unit_id, membership_status="active"
        ).first()
        credential = dependencies.ControllerKioskCredential.query.filter_by(
            unit_id=unit_id, person_id=person_id, enabled=True
        ).first()
        valid = bool(
            person
            and credential
            and (not credential.locked_until or credential.locked_until <= now)
            and check_password_hash(credential.pin_hash, pin)
        )
        if not valid:
            if credential:
                credential.failed_attempts = int(credential.failed_attempts or 0) + 1
                if credential.failed_attempts >= 5:
                    credential.locked_until = now + timedelta(minutes=15)
            _audit_identity("identity_verification_failed", current_user.id, person_id)
            dependencies.db.session.commit()
            dependencies.security_event("controller_kiosk_pin_failed", unit_id=unit_id)
            abort(403, "Identity verification failed.")
        credential.failed_attempts = 0
        credential.locked_until = None
        _audit_identity("identity_verified", person.id, person_id)
        dependencies.db.session.commit()
        dependencies.reset_rate_limit("controller-kiosk-pin", f"{unit_id}:{person_id}")
        return person

    def _validate_primary(person: Any) -> None:
        today = dependencies.utcnow().date()
        if not person.is_operational:
            raise LivePositionValidationError(
                "This person is not currently operational."
            )
        if not person.medical_expiry or person.medical_expiry < today:
            raise LivePositionValidationError(
                "A current medical is required to log on."
            )
        legacy_ue = any(
            expiry and expiry >= today
            for expiry in (
                person.tower_ue_expiry,
                person.radar_ue_expiry,
                person.met_ue_expiry,
            )
        )
        endorsement = dependencies.PositionEndorsement.query.filter(
            dependencies.PositionEndorsement.unit_id == _unit_id(),
            dependencies.PositionEndorsement.person_id == person.id,
            dependencies.PositionEndorsement.status == "valid",
            dependencies.PositionEndorsement.valid_from <= today,
            dependencies.db.or_(
                dependencies.PositionEndorsement.valid_until.is_(None),
                dependencies.PositionEndorsement.valid_until >= today,
            ),
        ).first()
        if not legacy_ue and not endorsement:
            raise LivePositionValidationError(
                "A current unit endorsement is required to log on."
            )

    def _validate_support(person: Any, role: Any) -> None:
        _validate_primary(person)
        if role.code == "ojti" and not person.has_ojti:
            raise LivePositionValidationError("A current OJTI is required.")
        if role.code in {"assessor", "examiner"} and not person.has_assessor:
            raise LivePositionValidationError(
                "A current assessor authority is required."
            )

    def _mutation_error(error: Exception):
        status = 409 if isinstance(error, LivePositionConflict) else 422
        return jsonify({"ok": False, "error": str(error)}), status

    def _ensure_participant_roles() -> None:
        defaults = (
            ("primary", "Primary controller", True, True),
            ("ojti", "OJTI", False, False),
            ("assessor", "Assessor", False, False),
            ("secondary", "Secondary controller", False, False),
            ("examiner", "Examiner", False, False),
            ("safety_controller", "Safety controller", False, False),
            ("observer", "Observer", False, False),
        )
        existing = {
            row.code
            for row in dependencies.PositionParticipantRole.query.filter_by(
                unit_id=_unit_id()
            ).all()
        }
        missing = [
            dependencies.PositionParticipantRole(
                unit_id=_unit_id(),
                code=code,
                label=label,
                is_primary=is_primary,
                counts_for_currency=counts,
                is_active=True,
            )
            for code, label, is_primary, counts in defaults
            if code not in existing
        ]
        if missing:
            dependencies.db.session.add_all(missing)
            dependencies.db.session.commit()

    @blueprint.get("/")
    @login_required
    def admin_home():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        _ensure_participant_roles()
        return render_template("live_position/admin_home.html")

    @blueprint.route("/admin/controller-pins", methods=["GET", "POST"])
    @login_required
    def controller_pins():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if request.method == "POST":
            data = _payload()
            person_id = int(data.get("person_id") or 0)
            pin = str(data.get("pin") or "")
            person = dependencies.Staff.query.filter_by(
                id=person_id, unit_id=_unit_id(), membership_status="active"
            ).first_or_404()
            if not pin.isdigit() or not 4 <= len(pin) <= 8:
                flash("PINs must contain 4 to 8 digits.", "error")
            else:
                credential = dependencies.ControllerKioskCredential.query.filter_by(
                    unit_id=_unit_id(), person_id=person.id
                ).first()
                if not credential:
                    credential = dependencies.ControllerKioskCredential(
                        unit_id=_unit_id(),
                        person_id=person.id,
                        pin_hash="",
                        changed_at=dependencies.utcnow(),
                    )
                    dependencies.db.session.add(credential)
                credential.pin_hash = generate_password_hash(pin)
                credential.enabled = True
                credential.failed_attempts = 0
                credential.locked_until = None
                credential.changed_at = dependencies.utcnow()
                _audit_identity("controller_pin_changed", current_user.id, person.id)
                dependencies.db.session.commit()
                flash(f"Kiosk PIN updated for {person.name}.", "ok")
                return redirect(url_for("live_position.controller_pins"))
        people = (
            dependencies.Staff.query.filter_by(
                unit_id=_unit_id(),
                membership_status="active",
                is_operational=True,
            )
            .order_by(dependencies.Staff.name)
            .all()
        )
        configured = {
            row.person_id
            for row in dependencies.ControllerKioskCredential.query.filter_by(
                unit_id=_unit_id(), enabled=True
            ).all()
        }
        return render_template(
            "live_position/controller_pins.html",
            people=people,
            configured=configured,
        )

    @blueprint.get("/kiosk")
    @login_required
    def kiosk_hmi():
        _require_kiosk_or_admin()
        unit = dependencies.db.session.get(dependencies.Unit, _unit_id())
        return render_template("live_position/kiosk.html", unit=unit)

    @blueprint.get("/api/controllers")
    @login_required
    def controllers():
        _require_kiosk_or_admin()
        _ensure_participant_roles()
        configured_ids = {
            row.person_id
            for row in dependencies.ControllerKioskCredential.query.filter_by(
                unit_id=_unit_id(), enabled=True
            ).all()
        }
        people = (
            dependencies.Staff.query.filter_by(
                unit_id=_unit_id(),
                membership_status="active",
                is_operational=True,
            )
            .order_by(dependencies.Staff.name)
            .all()
        )
        roles = (
            dependencies.PositionParticipantRole.query.filter_by(
                unit_id=_unit_id(), is_active=True, is_primary=False
            )
            .order_by(dependencies.PositionParticipantRole.label)
            .all()
        )
        return jsonify(
            {
                "controllers": [
                    {"id": person.id, "name": person.name}
                    for person in people
                    if person.id in configured_ids
                ],
                "roles": [
                    {"id": role.id, "code": role.code, "label": role.label}
                    for role in roles
                ],
            }
        )

    @blueprint.post("/api/positions/<int:position_id>/open")
    @login_required
    def open_position(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        actor = _verify_pin(int(data.get("person_id") or 0), str(data.get("pin") or ""))
        try:
            _service().set_position_open(
                unit_id=_unit_id(),
                position_id=position_id,
                actor_id=actor.id,
                open_position=True,
                reason=str(data.get("reason") or ""),
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/close")
    @login_required
    def close_position(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        actor = _verify_pin(int(data.get("person_id") or 0), str(data.get("pin") or ""))
        try:
            _service().set_position_open(
                unit_id=_unit_id(),
                position_id=position_id,
                actor_id=actor.id,
                open_position=False,
                reason=str(data.get("reason") or ""),
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/logon")
    @login_required
    def logon(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        person = _verify_pin(
            int(data.get("person_id") or 0), str(data.get("pin") or "")
        )
        try:
            _validate_primary(person)
            _service().start_session(
                unit_id=_unit_id(),
                position_id=position_id,
                person_id=person.id,
                actor_id=person.id,
                session_type=str(data.get("session_type") or "operational"),
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/logoff")
    @login_required
    def logoff(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        person = _verify_pin(
            int(data.get("person_id") or 0), str(data.get("pin") or "")
        )
        active = dependencies.PositionSession.query.filter_by(
            unit_id=_unit_id(),
            position_id=position_id,
            ended_at=None,
            is_void=False,
        ).first()
        if not active or active.primary_person_id != person.id:
            abort(403, "Only the active primary controller may log off.")
        try:
            if str(data.get("close_position") or "").lower() in {"1", "true", "yes"}:
                _service().set_position_open(
                    unit_id=_unit_id(),
                    position_id=position_id,
                    actor_id=person.id,
                    open_position=False,
                    reason=str(data.get("reason") or ""),
                    request_key=_request_key(data),
                )
            else:
                _service().end_session(
                    unit_id=_unit_id(),
                    position_id=position_id,
                    actor_id=person.id,
                    request_key=_request_key(data),
                )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/handover")
    @login_required
    def handover(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        incoming = _verify_pin(
            int(data.get("person_id") or 0), str(data.get("pin") or "")
        )
        try:
            _validate_primary(incoming)
            _service().handover(
                unit_id=_unit_id(),
                position_id=position_id,
                incoming_person_id=incoming.id,
                actor_id=incoming.id,
                session_type=str(data.get("session_type") or "operational"),
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/participants")
    @login_required
    def add_participant(position_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        person = _verify_pin(
            int(data.get("person_id") or 0), str(data.get("pin") or "")
        )
        role = dependencies.PositionParticipantRole.query.filter_by(
            id=int(data.get("role_id") or 0),
            unit_id=_unit_id(),
            is_active=True,
            is_primary=False,
        ).first_or_404()
        try:
            _validate_support(person, role)
            row = _service().add_participant(
                unit_id=_unit_id(),
                position_id=position_id,
                person_id=person.id,
                role_id=role.id,
                actor_id=person.id,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True, "participant_id": row.id})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post(
        "/api/positions/<int:position_id>/participants/<int:participant_id>/logoff"
    )
    @login_required
    def remove_participant(position_id: int, participant_id: int):
        _require_kiosk_or_admin()
        data = _payload()
        person = _verify_pin(
            int(data.get("person_id") or 0), str(data.get("pin") or "")
        )
        participant = dependencies.PositionSessionParticipant.query.filter_by(
            id=participant_id, unit_id=_unit_id(), ended_at=None
        ).first()
        if not participant or participant.person_id != person.id:
            abort(403, "Only the active participant may log off.")
        try:
            _service().end_participant(
                unit_id=_unit_id(),
                position_id=position_id,
                participant_id=participant_id,
                actor_id=person.id,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    def _state_payload() -> dict[str, Any]:
        unit_id = _unit_id()
        now = dependencies.utcnow()
        positions = (
            dependencies.OperationalPosition.query.filter_by(
                unit_id=unit_id, is_active=True
            )
            .order_by(
                dependencies.OperationalPosition.display_order,
                dependencies.OperationalPosition.code,
            )
            .all()
        )
        state = []
        for position in positions:
            status_event = (
                dependencies.PositionStatusEvent.query.filter_by(
                    unit_id=unit_id, position_id=position.id
                )
                .order_by(
                    dependencies.PositionStatusEvent.occurred_at.desc(),
                    dependencies.PositionStatusEvent.id.desc(),
                )
                .first()
            )
            session = dependencies.PositionSession.query.filter_by(
                unit_id=unit_id,
                position_id=position.id,
                ended_at=None,
                is_void=False,
            ).first()
            primary = (
                dependencies.db.session.get(
                    dependencies.Staff, session.primary_person_id
                )
                if session
                else None
            )
            participants = []
            if session:
                for row in dependencies.PositionSessionParticipant.query.filter_by(
                    unit_id=unit_id, session_id=session.id, ended_at=None
                ).all():
                    person = dependencies.db.session.get(
                        dependencies.Staff, row.person_id
                    )
                    role = dependencies.db.session.get(
                        dependencies.PositionParticipantRole, row.role_id
                    )
                    participants.append(
                        {
                            "id": row.id,
                            "person_name": person.name if person else "Unknown",
                            "role_id": row.role_id,
                            "role_label": role.label if role else "Supporting",
                            "started_at": row.started_at.isoformat() + "Z",
                        }
                    )
            physical_status = status_event.status if status_event else "closed"
            display_status = (
                "closed"
                if physical_status == "closed"
                else session.session_type
                if session
                else "vacant"
            )
            state.append(
                {
                    "id": position.id,
                    "code": position.code,
                    "label": position.label,
                    "physical_status": physical_status,
                    "display_status": display_status,
                    "primary": (
                        {
                            "id": primary.id,
                            "name": primary.name,
                            "started_at": session.started_at.isoformat() + "Z",
                            "due_off_at": (
                                session.due_off_at.isoformat() + "Z"
                                if session.due_off_at
                                else None
                            ),
                        }
                        if primary and session
                        else None
                    ),
                    "participants": participants,
                }
            )
        return {"server_time": now.isoformat() + "Z", "positions": state}

    @blueprint.get("/api/state")
    @login_required
    def live_state():
        _require_kiosk_or_admin()
        return jsonify(_state_payload())

    @blueprint.get("/api/events")
    @login_required
    def live_events():
        _require_kiosk_or_admin()

        @stream_with_context
        def events():
            # Database state is authoritative. A short heartbeat makes this
            # work consistently across multiple web processes; the browser's
            # normal polling remains the fallback when SSE is unavailable.
            for _index in range(120):
                yield f"event: state\ndata: {json.dumps(_state_payload())}\n\n"
                time.sleep(5)

        return Response(
            events(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return blueprint
