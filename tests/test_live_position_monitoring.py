from datetime import date, datetime
import re

import pytest

import app
from conftest import finish_operational_login
from live_position_service import LivePositionModels, LivePositionService


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
            [unit, kiosk, controller, supporter, admin, position_group, position, role]
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
    client = app.app.test_client()
    csrf = _login_kiosk(client)
    position = live_position_data["position_id"]
    opened = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/open",
        {
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
            "support_person_id": live_position_data["supporter_id"],
            "support_role": "ojti",
            "request_key": "logon-workflow",
        },
    )
    assert logged_on.status_code == 200
    state = client.get("/live-positions/api/state").get_json()["positions"][0]
    assert state["display_status"] == "training"
    assert state["primary"]["name"] == "Alex Controller"
    assert state["participants"][0]["role_label"] == "OJTI"
    assert "+00:00Z" not in state["primary"]["started_at"]
    participant_id = state["participants"][0]["id"]
    removed = _action(
        client,
        csrf,
        f"/live-positions/api/positions/{position}/participants/{participant_id}/logoff",
        {
            "request_key": "support-logoff-workflow",
        },
    )
    assert removed.status_code == 200
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
    with client.session_transaction() as session:
        csrf = session["_csrf_token"]
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
