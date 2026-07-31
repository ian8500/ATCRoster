from datetime import date, datetime

import pytest

import app
from conftest import finish_operational_login
from live_position_service import LivePositionModels, LivePositionService
from werkzeug.security import generate_password_hash


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
        admin.set_password("admin-password")
        position = app.OperationalPosition(
            unit_id=1,
            code="AIR",
            label="Aerodrome Control",
        )
        role = app.PositionParticipantRole(
            unit_id=1,
            code="ojti",
            label="OJTI",
            is_primary=False,
        )
        app.db.session.add_all(
            [unit, kiosk, controller, supporter, admin, position, role]
        )
        app.db.session.flush()
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
        app.db.session.add_all(
            [
                app.ControllerKioskCredential(
                    unit_id=1,
                    person_id=controller.id,
                    pin_hash=generate_password_hash("1234"),
                    changed_at=datetime.now(),
                ),
                app.ControllerKioskCredential(
                    unit_id=1,
                    person_id=supporter.id,
                    pin_hash=generate_password_hash("5678"),
                    changed_at=datetime.now(),
                ),
            ]
        )
        app.db.session.commit()
        yield {
            "kiosk_id": kiosk.id,
            "controller_id": controller.id,
            "supporter_id": supporter.id,
            "position_id": position.id,
            "role_id": role.id,
        }
        app.db.session.remove()
        app.db.drop_all()


def _service(now):
    return LivePositionService(
        app.db,
        LivePositionModels(
            app.OperationalPosition,
            app.PositionStatusEvent,
            app.PositionSession,
            app.PositionSessionParticipant,
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
    assert client.get("/live-positions/kiosk").status_code == 200
    state = client.get("/live-positions/api/state")
    assert state.status_code == 200
    assert state.get_json()["positions"][0]["display_status"] == "closed"
    blocked = client.get("/roster/2026-07")
    assert blocked.status_code == 302
    assert blocked.headers["Location"].endswith("/live-positions/kiosk")


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


def test_kiosk_pin_authorises_full_live_position_workflow(live_position_data):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    position = live_position_data["position_id"]
    opened = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/open",
        {
            "person_id": live_position_data["controller_id"],
            "pin": "1234",
            "request_key": "open-workflow",
        },
    )
    assert opened.status_code == 200
    logged_on = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logon",
        {
            "person_id": live_position_data["controller_id"],
            "pin": "1234",
            "session_type": "training",
            "request_key": "logon-workflow",
        },
    )
    assert logged_on.status_code == 200
    supported = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants",
        {
            "person_id": live_position_data["supporter_id"],
            "pin": "5678",
            "role_id": live_position_data["role_id"],
            "request_key": "support-workflow",
        },
    )
    assert supported.status_code == 200
    state = client.get("/live-positions/api/state").get_json()["positions"][0]
    assert state["display_status"] == "training"
    assert state["primary"]["name"] == "Alex Controller"
    assert state["participants"][0]["role_label"] == "OJTI"
    participant_id = supported.get_json()["participant_id"]
    removed = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants/{participant_id}/logoff",
        {
            "person_id": live_position_data["supporter_id"],
            "pin": "5678",
            "request_key": "support-logoff-workflow",
        },
    )
    assert removed.status_code == 200
    handed_over = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/handover",
        {
            "person_id": live_position_data["supporter_id"],
            "pin": "5678",
            "session_type": "operational",
            "request_key": "handover-workflow",
        },
    )
    assert handed_over.status_code == 200
    assert (
        client.get("/live-positions/api/state").get_json()["positions"][0]["primary"][
            "name"
        ]
        == "Sam Instructor"
    )
    ended = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/logoff",
        {
            "person_id": live_position_data["supporter_id"],
            "pin": "5678",
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


def test_wrong_pin_is_generic_and_audited(live_position_data):
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    response = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{live_position_data['position_id']}/open",
        {
            "person_id": live_position_data["controller_id"],
            "pin": "9999",
            "request_key": "wrong-pin",
        },
    )
    assert response.status_code == 403
    with app.app.app_context():
        audit = app.PositionSessionAudit.query.filter_by(
            action="identity_verification_failed"
        ).one()
        assert "requested_person_id" in audit.new_value_json


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
    module = client.get("/live-positions/")
    assert module.status_code == 200
    assert b"/live-positions/admin/positions" in module.data
    assert b"Coming in the next build stage" in module.data
    page = client.get("/live-positions/admin/positions")
    assert page.status_code == 200
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
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
    position = client.post(
        "/live-positions/admin/positions",
        data={
            "_csrf_token": csrf,
            "action": "create_position",
            "code": "GMC",
            "label": "Ground Movement Control",
            "display_order": "10",
            "group_name": "Tower",
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
        assert configured.currency_category_id == category_id
