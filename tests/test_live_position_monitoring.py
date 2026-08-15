from datetime import date, datetime, timedelta
import re

import pytest

import app
from conftest import finish_operational_login
from live_position_service import (
    LivePositionModels,
    LivePositionService,
    LivePositionValidationError,
)


@pytest.fixture()
def live_position_data():
    with app.app.app_context():
        app.db.drop_all()
        app.db.create_all()
        unit = app.Unit(id=1, code="LIVE", name="Live Test Airport")
        kiosk = app.Staff(
            unit_id=1,
            username="position-screen",
            name="Position screen",
            staff_no="KIOSK-1",
            role="position_monitor",
            is_operational=False,
        )
        controller = app.Staff(
            unit_id=1,
            username="controller",
            name="Alex Controller",
            staff_no="ATCO-1",
            role="user",
            is_operational=True,
            medical_expiry=date(2027, 7, 31),
            tower_ue_expiry=date(2027, 7, 31),
        )
        supporter = app.Staff(
            unit_id=1,
            username="ojti",
            name="Sam Instructor",
            staff_no="ATCO-2",
            role="user",
            is_operational=True,
            medical_expiry=date(2027, 7, 31),
            tower_ue_expiry=date(2027, 7, 31),
            has_ojti=True,
            has_assessor=True,
        )
        trainee = app.Staff(
            unit_id=1,
            username="trainee",
            name="Taylor Trainee",
            staff_no="ATCO-3",
            role="user",
            is_operational=True,
            is_trainee=True,
            medical_expiry=date(2027, 7, 31),
            tower_ue_expiry=date(2027, 7, 31),
        )
        admin = app.Staff(
            unit_id=1,
            username="live-admin",
            name="Live Administrator",
            staff_no="ADMIN-1",
            role="admin",
            is_operational=False,
        )
        kiosk.set_password("secure-kiosk-password")
        controller.set_password("controller-password")
        supporter.set_password("supporter-password")
        trainee.set_password("trainee-password")
        admin.set_password("admin-password")
        position_group = app.OperationalPositionGroup(
            unit_id=1, name="Tower", display_order=10, is_active=True
        )
        position = app.OperationalPosition(
            unit_id=1,
            code="AIR",
            label="Aerodrome Control",
            group_name="Tower",
        )
        role = app.PositionParticipantRole(
            unit_id=1,
            code="ojti",
            label="OJTI",
            is_primary=False,
        )
        app.db.session.add_all(
            [unit, kiosk, controller, supporter, trainee, admin, position_group, position, role]
        )
        app.db.session.flush()
        app.db.session.add(
            app.FeatureFlag(unit_id=1, key="live_position_monitoring", enabled=True)
        )
        identity = app.PlatformIdentity(
            public_id="test-position-screen",
            username=kiosk.username,
            password_hash=kiosk.password_hash,
        )
        app.db.session.add(identity)
        app.db.session.flush()
        admin_identity = app.PlatformIdentity(
            public_id="test-live-admin",
            username=admin.username,
            password_hash=admin.password_hash,
        )
        app.db.session.add(admin_identity)
        app.db.session.flush()
        app.db.session.add(
            app.UnitMembership(
                identity_id=identity.id,
                unit_id=1,
                person_id=kiosk.id,
                role="StaffUser",
                status="active",
            )
        )
        app.db.session.add(
            app.UnitMembership(
                identity_id=admin_identity.id,
                unit_id=1,
                person_id=admin.id,
                role="UnitAdmin",
                status="active",
            )
        )
        app.db.session.commit()
        yield {
            "kiosk_id": kiosk.id,
            "admin_id": admin.id,
            "controller_id": controller.id,
            "supporter_id": supporter.id,
            "trainee_id": trainee.id,
            "position_id": position.id,
            "role_id": role.id,
        }
        app.db.session.remove()
        app.db.drop_all()


def _service(now):
    return LivePositionService(
        app.db,
        LivePositionModels(
            app.Staff,
            app.OperationalPosition,
            app.PositionStatusEvent,
            app.PositionSession,
            app.PositionSessionParticipant,
            app.PositionParticipantRole,
            app.PositionSessionAudit,
        ),
        lambda: now,
    )


def test_atomic_position_lifecycle_is_audited_and_idempotent(live_position_data):
    moment = datetime(2026, 7, 31, 8, 30)
    with app.app.app_context():
        service = _service(moment)
        opened = service.set_position_open(
            unit_id=1,
            position_id=live_position_data["position_id"],
            actor_id=live_position_data["kiosk_id"],
            open_position=True,
            request_key="open-once",
        )
        assert (
            service.set_position_open(
                unit_id=1,
                position_id=live_position_data["position_id"],
                actor_id=live_position_data["kiosk_id"],
                open_position=True,
                request_key="open-once",
            ).id
            == opened.id
        )
        session = service.start_session(
            unit_id=1,
            position_id=live_position_data["position_id"],
            person_id=live_position_data["controller_id"],
            actor_id=live_position_data["kiosk_id"],
            request_key="logon-once",
        )
        assert session.started_at == moment
        ended = service.end_session(
            unit_id=1,
            position_id=live_position_data["position_id"],
            actor_id=live_position_data["kiosk_id"],
            request_key="logoff-once",
        )
        assert ended.ended_at == moment
        assert app.PositionSessionAudit.query.count() == 3


def test_service_rejects_unverified_actor_and_cross_unit_participant_role(
    live_position_data,
):
    moment = datetime(2026, 7, 31, 8, 30)
    with app.app.app_context():
        service = _service(moment)
        with pytest.raises(
            LivePositionValidationError,
            match="active kiosk account",
        ):
            service.set_position_open(
                unit_id=1,
                position_id=live_position_data["position_id"],
                actor_id=live_position_data["admin_id"],
                open_position=True,
            )

        other_unit = app.Unit(id=2, code="OTHER", name="Other Airport")
        other_role = app.PositionParticipantRole(
            unit_id=2,
            code="ojti",
            label="Other OJTI",
            is_primary=False,
        )
        app.db.session.add_all([other_unit, other_role])
        app.db.session.commit()

        with pytest.raises(
            LivePositionValidationError,
            match="supporting participant role",
        ):
            service.start_session(
                unit_id=1,
                position_id=live_position_data["position_id"],
                person_id=live_position_data["controller_id"],
                actor_id=live_position_data["kiosk_id"],
                participants=[
                    {
                        "person_id": live_position_data["supporter_id"],
                        "role_id": other_role.id,
                    }
                ],
            )


def test_kiosk_password_login_bypasses_only_mfa_and_is_endpoint_limited(
    live_position_data,
):
    client = app.app.test_client()
    client.get("/login")
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    response = client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "position-screen",
            "password": "secure-kiosk-password",
        },
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/live-positions/kiosk")
    kiosk_page = client.get("/live-positions/kiosk")
    assert kiosk_page.status_code == 200
    assert b"Start kiosk display" in kiosk_page.data
    assert b"requestFullscreen" in kiosk_page.data
    assert b"live-position-board-viewport" in kiosk_page.data
    assert b'id="live-position-theme-toggle"' in kiosk_page.data
    assert b"atcroster-live-position-theme" in kiosk_page.data
    assert b"Switch to light mode" in kiosk_page.data
    assert b"ResizeObserver" in kiosk_page.data
    assert b"operationalGroupPriority" in kiosk_page.data
    assert b"startsWith('tower')" in kiosk_page.data
    assert b"startsWith('radar')" in kiosk_page.data
    assert b'data-primary-id="${occupied ? occupied.id' in kiosk_page.data
    assert b"primarySelect.disabled = operation === 'participant'" in kiosk_page.data
    assert b"operation === 'logoff' || operation === 'logoff_close'" in kiosk_page.data
    assert b"if (operation === 'close')" in kiosk_page.data
    assert b"The position could not be closed." in kiosk_page.data
    assert b"close_position: operation === 'logoff_close'" in kiosk_page.data
    assert b"Log off all controllers" in kiosk_page.data
    assert b"data-logoff-choice" in kiosk_page.data
    assert b"position?.participants.length" in kiosk_page.data
    assert b"data-accrued-seconds" in kiosk_page.data
    assert b"data-remaining-accrued-seconds" in kiosk_page.data
    assert b"Reset break" in kiosk_page.data
    assert b"group.positions.length} position" not in kiosk_page.data
    assert b'id="live-position-kiosk-logout"' in kiosk_page.data
    assert b"kioskLogout.requestSubmit()" in kiosk_page.data
    assert b"Log out</button>" not in kiosk_page.data
    state = client.get("/live-positions/api/state")
    assert state.status_code == 200
    assert state.get_json()["positions"][0]["display_status"] == "closed"
    blocked = client.get("/roster/2026-07")
    assert blocked.status_code == 302
    assert blocked.headers["Location"].endswith("/live-positions/kiosk")


def test_live_state_includes_configured_display_group(live_position_data):
    client = app.app.test_client()
    _login_kiosk(client)
    state = client.get("/live-positions/api/state").get_json()["positions"][0]
    assert state["group_name"] == "Tower"
    assert state["group_order"] == 10


def test_live_event_stream_retains_authenticated_tenant_context(live_position_data):
    client = app.app.test_client()
    _login_kiosk(client)

    response = client.get("/live-positions/api/events", buffered=False)
    try:
        first_event = next(response.response)
    finally:
        response.close()

    assert response.status_code == 200
    assert first_event.startswith(b"event: state\ndata: ")
    assert b'"group_name": "Tower"' in first_event


def _login_kiosk(client):
    client.get("/login")
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    response = client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "position-screen",
            "password": "secure-kiosk-password",
        },
    )
    assert response.status_code == 302
    assert client.get("/live-positions/kiosk").status_code == 200
    with client.session_transaction() as session:
        return session["_csrf_token"]


def _action(client, csrf, path, payload):
    return client.post(
        path,
        json=payload,
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": payload["request_key"]},
    )


def test_kiosk_controller_selection_runs_full_live_position_workflow(
    live_position_data,
):
    admin_client = app.app.test_client()
    admin_client.get("/login")
    with admin_client.session_transaction() as browser_session:
        admin_csrf = browser_session["_csrf_token"]
    admin_client.post(
        "/login",
        data={
            "_csrf_token": admin_csrf,
            "username": "live-admin",
            "password": "admin-password",
        },
    )
    finish_operational_login(admin_client)
    with admin_client.session_transaction() as browser_session:
        admin_csrf = browser_session["_csrf_token"]
    matrix_update = {
        "_csrf_token": admin_csrf,
        "action": "update_position",
        "position_id": str(live_position_data["position_id"]),
        "code": "AIR",
        "label": "Aerodrome Control",
        "display_order": "100",
        "maximum_session_duration_minutes": "120",
        "supporting_participants_allowed": "on",
        "multiple_supporting_participants_allowed": "on",
        "training_supported": "on",
        "assessment_supported": "on",
        "is_safety_critical": "on",
        "is_active": "on",
    }
    matrix_update.update(
        {
            f"allowance_{weekday}_{hour}": "75"
            for weekday in range(7)
            for hour in range(24)
        }
    )
    assert (
        admin_client.post(
            "/live-positions/admin/positions", data=matrix_update
        ).status_code
        == 302
    )
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    position = live_position_data["position_id"]
    logged_on = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "support_person_id": live_position_data["supporter_id"],
            "support_role": "ojti",
            "request_key": "logon-workflow",
        },
    )
    assert logged_on.status_code == 200
    state = client.get("/live-positions/api/state").get_json()["positions"][0]
    assert state["display_status"] == "training"
    assert state["physical_status"] == "open"
    assert state["primary"]["maximum_duration_seconds"] == 4500
    assert state["participants"][0]["maximum_duration_seconds"] == 4500
    assert state["primary"]["name"] == "Alex Controller"
    assert state["participants"][0]["role_label"] == "OJTI"
    assert "+00:00Z" not in state["primary"]["started_at"]
    participant_id = state["participants"][0]["id"]
    with app.app.app_context():
        opened = app.PositionStatusEvent.query.filter_by(
            position_id=position, status="open"
        ).one()
        assert opened.reason == "Opened automatically on controller logon"
    removed = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants/{participant_id}/logoff",
        {
            "request_key": "support-logoff-workflow",
        },
    )
    assert removed.status_code == 200
    primary_only_state = client.get("/live-positions/api/state").get_json()[
        "positions"
    ][0]
    assert primary_only_state["primary"]["name"] == "Alex Controller"
    assert primary_only_state["participants"] == []
    assert primary_only_state["display_status"] == "operational"
    assessed = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants",
        {
            "support_person_id": live_position_data["supporter_id"],
            "support_role": "assessor",
            "request_key": "assessor-workflow",
        },
    )
    assert assessed.status_code == 200
    assessed_state = client.get("/live-positions/api/state").get_json()["positions"][0]
    assert assessed_state["display_status"] == "assessment"
    assert assessed_state["participants"][0]["role_label"] == "Assessor"
    assert (
        _action(
            client,
            csrf,
            f"/live-positions/api/positions/{position}/participants/{assessed.get_json()['participant_id']}/logoff",
            {"request_key": "assessor-logoff-workflow"},
        ).status_code
        == 200
    )
    handed_over = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/handover",
        {
            "person_id": live_position_data["supporter_id"],
            "request_key": "handover-workflow",
        },
    )
    assert handed_over.status_code == 200
    handover_state = client.get("/live-positions/api/state").get_json()["positions"][0][
        "primary"
    ]
    assert handover_state["name"] == "Sam Instructor"
    assert handover_state["maximum_duration_seconds"] == 4500
    ended = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logoff",
        {
            "close_position": True,
            "request_key": "close-workflow",
        },
    )
    assert ended.status_code == 200
    assert (
        client.get("/live-positions/api/state").get_json()["positions"][0][
            "display_status"
        ]
        == "closed"
    )


def test_kiosk_actions_do_not_request_or_require_controller_pins(
    live_position_data,
):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    page = client.get("/live-positions/kiosk")
    assert b"Controller PIN" not in page.data
    assert b"Reason or note" not in page.data
    response = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{live_position_data['position_id']}/open",
        {
            "request_key": "no-pin-required",
        },
    )
    assert response.status_code == 200


def test_live_state_carries_position_time_across_a_short_break(live_position_data):
    with app.app.app_context():
        now = app.utcnow().replace(tzinfo=None)
        app.db.session.add_all(
            [
                app.PositionStatusEvent(
                    unit_id=1,
                    position_id=live_position_data["position_id"],
                    status="open",
                    occurred_at=now - timedelta(hours=2),
                    actor_id=live_position_data["kiosk_id"],
                    transaction_key="accrual-open",
                ),
                app.PositionSession(
                    unit_id=1,
                    position_id=live_position_data["position_id"],
                    primary_person_id=live_position_data["controller_id"],
                    started_at=now - timedelta(minutes=120),
                    ended_at=now - timedelta(minutes=30),
                    created_by_id=live_position_data["kiosk_id"],
                    transaction_key="accrual-prior",
                ),
                app.PositionSession(
                    unit_id=1,
                    position_id=live_position_data["position_id"],
                    primary_person_id=live_position_data["controller_id"],
                    started_at=now - timedelta(minutes=10),
                    created_by_id=live_position_data["kiosk_id"],
                    transaction_key="accrual-current",
                ),
            ]
        )
        app.db.session.commit()

    client = app.app.test_client()
    _login_kiosk(client)
    primary = client.get("/live-positions/api/state").get_json()["positions"][0][
        "primary"
    ]
    assert 5990 <= primary["accrued_seconds"] <= 6020
    assert primary["required_break_seconds"] == 1800


def test_controller_cannot_be_logged_on_to_two_positions(live_position_data):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    with app.app.app_context():
        second = app.OperationalPosition(
            unit_id=1, code="GMC", label="Ground", display_order=2
        )
        app.db.session.add(second)
        app.db.session.commit()
        second_id = second.id
    for position_id, request_key in (
        (live_position_data["position_id"], "open-first"),
        (second_id, "open-second"),
    ):
        assert (
            _action(
                client,
                csrf,
                f"/live-positions/api/positions/{position_id}/open",
                {"request_key": request_key},
            ).status_code
            == 200
        )
    first = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{live_position_data['position_id']}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "request_key": "first-logon",
        },
    )
    assert first.status_code == 200
    duplicate = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{second_id}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "request_key": "duplicate-logon",
        },
    )
    assert duplicate.status_code == 409
    assert "already logged on" in duplicate.get_json()["error"]


def test_secondary_role_requires_the_matching_qualification(live_position_data):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    position = live_position_data["position_id"]
    assert (
        _action(
            client,
            csrf,
            f"/live-positions/api/positions/{position}/open",
            {"request_key": "open-qualification-check"},
        ).status_code
        == 200
    )
    rejected = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logon",
        {
            "person_id": live_position_data["supporter_id"],
            "support_person_id": live_position_data["controller_id"],
            "support_role": "ojti",
            "request_key": "unqualified-ojti",
        },
    )
    assert rejected.status_code == 422
    assert "OJTI" in rejected.get_json()["error"]


def test_configured_position_requires_its_own_current_endorsement(
    live_position_data,
):
    position_id = live_position_data["position_id"]
    with app.app.app_context():
        app.db.session.add(
            app.PositionEndorsement(
                unit_id=1,
                person_id=live_position_data["supporter_id"],
                position_id=position_id,
                valid_from=date(2020, 1, 1),
                status="valid",
            )
        )
        app.db.session.commit()
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    denied = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position_id}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "request_key": "missing-position-endorsement",
        },
    )
    assert denied.status_code == 422
    assert "endorsement for this position" in denied.get_json()["error"]
    with app.app.app_context():
        app.db.session.add(
            app.PositionEndorsement(
                unit_id=1,
                person_id=live_position_data["controller_id"],
                position_id=position_id,
                valid_from=date(2020, 1, 1),
                status="valid",
            )
        )
        app.db.session.commit()
    accepted = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position_id}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "request_key": "current-position-endorsement",
        },
    )
    assert accepted.status_code == 200


def test_trainee_requires_ojti_for_live_position_logon(live_position_data):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    position = live_position_data["position_id"]
    rejected = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logon",
        {
            "person_id": live_position_data["trainee_id"],
            "request_key": "trainee-without-ojti",
        },
    )
    assert rejected.status_code == 422
    assert "trainee requires a current OJTI" in rejected.get_json()["error"]

    supervised = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logon",
        {
            "person_id": live_position_data["trainee_id"],
            "support_person_id": live_position_data["supporter_id"],
            "support_role": "ojti",
            "request_key": "trainee-with-ojti",
        },
    )
    assert supervised.status_code == 200
    state = client.get("/live-positions/api/state").get_json()["positions"][0]
    ojti_participant_id = state["participants"][0]["id"]
    cannot_remove_ojti = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants/{ojti_participant_id}/logoff",
        {"request_key": "remove-required-ojti"},
    )
    assert cannot_remove_ojti.status_code == 422
    assert "OJTI cannot be removed" in cannot_remove_ojti.get_json()["error"]
    with app.app.app_context():
        expired_controller = app.Staff(
            unit_id=1,
            username="expired-controller",
            name="Expired Controller",
            staff_no="ATCO-EXPIRED",
            role="user",
            is_operational=True,
            medical_expiry=date(2020, 1, 1),
            tower_ue_expiry=date(2027, 7, 31),
        )
        expired_controller.set_password("expired-controller-password")
        alert_position = app.OperationalPosition(
            unit_id=1, code="GMC", label="Ground", display_order=2
        )
        app.db.session.add_all([expired_controller, alert_position])
        app.db.session.flush()
        app.db.session.add_all(
            [
                app.PositionSession(
                    unit_id=1,
                    position_id=alert_position.id,
                    primary_person_id=expired_controller.id,
                    session_type="operational",
                    started_at=datetime(2026, 7, 30, 8, 0),
                    created_by_id=live_position_data["kiosk_id"],
                    transaction_key="expired-controller-session",
                ),
                app.PositionStatusEvent(
                    unit_id=1,
                    position_id=alert_position.id,
                    status="open",
                    occurred_at=datetime(2026, 7, 30, 8, 0),
                    actor_id=live_position_data["kiosk_id"],
                    transaction_key="expired-controller-position-open",
                ),
            ]
        )
        app.db.session.commit()
    positions = client.get("/live-positions/api/state").get_json()["positions"]
    state = next(item for item in positions if item["id"] == alert_position.id)
    assert "A current medical is required to log on." in state["eligibility_warnings"]


def test_operational_activity_reports_split_solo_and_ojti_time(live_position_data):
    with app.app.app_context():
        session_row = app.PositionSession(
            unit_id=1,
            position_id=live_position_data["position_id"],
            primary_person_id=live_position_data["controller_id"],
            session_type="training",
            started_at=datetime(2026, 7, 30, 8, 0),
            ended_at=datetime(2026, 7, 30, 10, 0),
            created_by_id=live_position_data["kiosk_id"],
            transaction_key="activity-report-session",
        )
        app.db.session.add(session_row)
        app.db.session.flush()
        session_id = session_row.id
        app.db.session.add(
            app.PositionSessionParticipant(
                unit_id=1,
                session_id=session_row.id,
                person_id=live_position_data["supporter_id"],
                role_id=live_position_data["role_id"],
                started_at=datetime(2026, 7, 30, 9, 0),
                ended_at=datetime(2026, 7, 30, 9, 30),
                transaction_key="activity-report-participant",
            )
        )
        app.db.session.commit()

    client = app.app.test_client()
    client.get("/login")
    with client.session_transaction() as browser_session:
        csrf = browser_session["_csrf_token"]
    client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "live-admin",
            "password": "admin-password",
        },
    )
    finish_operational_login(client)
    query = (
        "start=2026-07-30&end=2026-07-30&person_id="
        f"{live_position_data['controller_id']}"
    )
    individual = client.get(f"/reports/operational-activity?{query}")
    assert individual.status_code == 200
    assert b"Alex Controller" in individual.data
    assert b"Operational activity total" not in individual.data
    assert b"All controllers" not in individual.data
    assert b"Position occupancy by day" in individual.data
    assert b"On position" in individual.data
    assert b"Off position" in individual.data
    assert b"Primary controller" in individual.data
    assert b"09:00" in individual.data
    assert b"11:00" in individual.data
    assert b"Breakdown by position" not in individual.data
    assert b"01:30" in individual.data
    assert b"00:30" in individual.data
    assert b"02:00" in individual.data
    assert b"<th>Sam Instructor</th>" not in individual.data
    assert b"Position screen" not in individual.data
    assert b"75.0%" in individual.data

    chooser = client.get(
        "/reports/operational-activity?"
        "start=2026-07-30&end=2026-07-30"
    )
    assert chooser.status_code == 200
    assert b"Entire unit" in chooser.data
    assert b"Alex Controller" in chooser.data
    assert b"Sam Instructor" in chooser.data

    position_only = client.get(
        "/reports/operational-activity?"
        "start=2026-07-30&end=2026-07-30&scope=unit&position_id="
        f"{live_position_data['position_id']}"
    )
    assert position_only.status_code == 200
    assert b"Aerodrome Control" in position_only.data
    assert b"10:00" in position_only.data
    assert b"10:30" in position_only.data

    legacy = client.get(
        f"/live-positions/reports/operational-activity?{query}",
        follow_redirects=False,
    )
    assert legacy.status_code == 308
    assert legacy.headers["Location"].endswith(f"/reports/operational-activity?{query}")

    # PostgreSQL DateTime columns return naive values while the application
    # clock is timezone-aware. An open session must still render safely.
    with app.app.app_context():
        open_session = app.db.session.get(app.PositionSession, session_id)
        open_session.ended_at = None
        app.db.session.commit()
    open_report = client.get(f"/reports/operational-activity?{query}")
    assert open_report.status_code == 200


def test_admin_can_configure_currency_category_and_position(live_position_data):
    client = app.app.test_client()
    client.get("/login")
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    response = client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "live-admin",
            "password": "admin-password",
        },
    )
    assert response.status_code == 302
    finish_operational_login(client)
    home = client.get("/modules")
    assert home.status_code == 200
    assert b"Administration" in home.data
    administration = client.get("/administration")
    assert administration.status_code == 200
    assert b"/live-positions/admin/positions" in administration.data
    assert client.get("/live-positions/kiosk").status_code == 403
    page = client.get("/live-positions/admin/positions")
    assert page.status_code == 200
    assert b"Maximum-time weekly matrix" in page.data
    assert b"Cumulative controller-time recovery" in page.data
    assert b"airport\xe2\x80\x99s local timezone" in page.data
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    policy = client.post(
        "/live-positions/admin/positions",
        data={
            "_csrf_token": csrf,
            "action": "update_recovery_policy",
            "base_break_minutes": "25",
            "escalation_after_minutes": "90",
            "extra_break_minutes": "10",
            "escalation_interval_minutes": "45",
            "escalation_cap_minutes": "180",
        },
    )
    assert policy.status_code == 302
    with app.app.app_context():
        saved_policy = app.LivePositionRecoveryPolicy.query.filter_by(unit_id=1).one()
        assert saved_policy.base_break_minutes == 25
        assert saved_policy.escalation_cap_minutes == 180
        assert app.PositionSessionAudit.query.filter_by(
            unit_id=1, action="recovery_policy_updated"
        ).one()
    group = client.post(
        "/live-positions/admin/positions",
        data={
            "_csrf_token": csrf,
            "action": "create_group",
            "group_name": "Radar",
            "group_display_order": "10",
        },
    )
    assert group.status_code == 302
    category = client.post(
        "/live-positions/admin/positions",
        data={
            "_csrf_token": csrf,
            "action": "create_category",
            "category_code": "TWR",
            "category_label": "Tower",
        },
    )
    assert category.status_code == 302
    with app.app.app_context():
        category_id = app.PositionCurrencyCategory.query.filter_by(code="TWR").one().id
        group_id = app.OperationalPositionGroup.query.filter_by(name="Radar").one().id
    position = client.post(
        "/live-positions/admin/positions",
        data={
            "_csrf_token": csrf,
            "action": "create_position",
            "code": "GMC",
            "label": "Ground Movement Control",
            "display_order": "10",
            "maximum_session_duration_minutes": "90",
            "allowance_0_8": "75",
            "position_group_id": str(group_id),
            "currency_category_id": str(category_id),
            "supporting_participants_allowed": "on",
            "multiple_supporting_participants_allowed": "on",
            "training_supported": "on",
            "assessment_supported": "on",
            "is_safety_critical": "on",
            "is_active": "on",
        },
    )
    assert position.status_code == 302
    with app.app.app_context():
        configured = app.OperationalPosition.query.filter_by(code="GMC").one()
        assert configured.label == "Ground Movement Control"
        assert configured.group_name == "Radar"
        assert configured.maximum_session_duration_minutes == 90
        allowance = app.OperationalPositionTimeAllowance.query.filter_by(
            unit_id=1, position_id=configured.id
        ).one()
        assert (allowance.weekday, allowance.start_hour) == (0, 8)
        assert allowance.maximum_duration_minutes == 75
        assert configured.currency_category_id == category_id
    page = client.get("/live-positions/admin/positions")
    assert b'<option value="%d" selected>Radar</option>' % group_id in page.data


def test_live_position_module_must_be_enabled(live_position_data):
    with app.app.app_context():
        app.FeatureFlag.query.filter_by(
            unit_id=1, key="live_position_monitoring"
        ).delete()
        app.db.session.commit()
    client = app.app.test_client()
    client.get("/login")
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
    client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "position-screen",
            "password": "secure-kiosk-password",
        },
    )
    assert client.get("/live-positions/kiosk").status_code == 404


def test_unit_admin_can_provision_a_dedicated_kiosk_account(live_position_data):
    admin_client = app.app.test_client()
    admin_client.get("/login")
    with admin_client.session_transaction() as session:
        csrf = session["_csrf_token"]
    admin_client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "live-admin",
            "password": "admin-password",
        },
    )
    finish_operational_login(admin_client)
    page = admin_client.get("/administration/kiosk-accounts")
    assert page.status_code == 200
    with admin_client.session_transaction() as session:
        csrf = session["_csrf_token"]
    created = admin_client.post(
        "/administration/kiosk-accounts",
        data={"_csrf_token": csrf, "action": "create_invitation"},
        follow_redirects=True,
    )
    token_match = re.search(rb"/invite/([A-Za-z0-9_-]+)", created.data)
    assert token_match
    invitation_path = f"/invite/{token_match.group(1).decode()}"

    kiosk_client = app.app.test_client()
    setup = kiosk_client.get(invitation_path)
    assert setup.status_code == 200
    with kiosk_client.session_transaction() as session:
        csrf = session["_csrf_token"]
    accepted = kiosk_client.post(
        invitation_path,
        data={
            "_csrf_token": csrf,
            "name": "Tower display",
            "username": "tower-display",
            "email": "kiosk@example.test",
            "password": "secure-kiosk-password",
        },
    )
    assert accepted.status_code == 302
    with app.app.app_context():
        kiosk = app.Staff.query.filter_by(username="tower-display").one()
        membership = app.UnitMembership.query.filter_by(person_id=kiosk.id).one()
        assert kiosk.role == "position_monitor"
        assert not kiosk.is_operational
        assert membership.role == "PositionMonitor"
        assert membership.status == "active"

    kiosk_client.get("/login")
    with kiosk_client.session_transaction() as session:
        csrf = session["_csrf_token"]
    login = kiosk_client.post(
        "/login",
        data={
            "_csrf_token": csrf,
            "username": "tower-display",
            "password": "secure-kiosk-password",
        },
    )
    assert login.status_code == 302
    assert login.headers["Location"].endswith("/live-positions/kiosk")
