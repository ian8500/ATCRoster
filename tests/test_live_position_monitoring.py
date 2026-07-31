from datetime import datetime

import pytest

import app
from live_position_service import LivePositionModels, LivePositionService


@pytest.fixture()
def live_position_data():
    with app.app.app_context():
        app.db.drop_all()
        app.db.create_all()
        unit = app.Unit(id=1, code="LIVE", name="Live Test Airport")
        kiosk = app.Staff(
            unit_id=1, username="position-screen", name="Position screen",
            staff_no="KIOSK-1", role="position_monitor", is_operational=False,
        )
        controller = app.Staff(
            unit_id=1, username="controller", name="Alex Controller",
            staff_no="ATCO-1", role="user", is_operational=True,
        )
        kiosk.set_password("secure-kiosk-password")
        controller.set_password("controller-password")
        position = app.OperationalPosition(
            unit_id=1, code="AIR", label="Aerodrome Control",
        )
        app.db.session.add_all([unit, kiosk, controller, position])
        app.db.session.flush()
        identity = app.PlatformIdentity(
            public_id="test-position-screen", username=kiosk.username,
            password_hash=kiosk.password_hash,
        )
        app.db.session.add(identity)
        app.db.session.flush()
        app.db.session.add(app.UnitMembership(
            identity_id=identity.id, unit_id=1, person_id=kiosk.id,
            role="StaffUser", status="active",
        ))
        app.db.session.commit()
        yield {
            "kiosk_id": kiosk.id, "controller_id": controller.id,
            "position_id": position.id,
        }
        app.db.session.remove()
        app.db.drop_all()


def _service(now):
    return LivePositionService(
        app.db,
        LivePositionModels(
            app.OperationalPosition, app.PositionStatusEvent,
            app.PositionSession, app.PositionSessionParticipant,
            app.PositionSessionAudit,
        ),
        lambda: now,
    )


def test_atomic_position_lifecycle_is_audited_and_idempotent(live_position_data):
    moment = datetime(2026, 7, 31, 8, 30)
    with app.app.app_context():
        service = _service(moment)
        opened = service.set_position_open(
            unit_id=1, position_id=live_position_data["position_id"],
            actor_id=live_position_data["kiosk_id"], open_position=True,
            request_key="open-once",
        )
        assert service.set_position_open(
            unit_id=1, position_id=live_position_data["position_id"],
            actor_id=live_position_data["kiosk_id"], open_position=True,
            request_key="open-once",
        ).id == opened.id
        session = service.start_session(
            unit_id=1, position_id=live_position_data["position_id"],
            person_id=live_position_data["controller_id"],
            actor_id=live_position_data["kiosk_id"], request_key="logon-once",
        )
        assert session.started_at == moment
        ended = service.end_session(
            unit_id=1, position_id=live_position_data["position_id"],
            actor_id=live_position_data["kiosk_id"], request_key="logoff-once",
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
    response = client.post("/login", data={
        "_csrf_token": csrf, "username": "position-screen",
        "password": "secure-kiosk-password",
    })
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/live-positions/kiosk")
    assert client.get("/live-positions/kiosk").status_code == 200
    state = client.get("/live-positions/api/state")
    assert state.status_code == 200
    assert state.get_json()["positions"][0]["display_status"] == "closed"
    blocked = client.get("/roster/2026-07")
    assert blocked.status_code == 302
    assert blocked.headers["Location"].endswith("/live-positions/kiosk")
