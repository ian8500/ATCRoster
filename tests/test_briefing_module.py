import io
import json
from datetime import datetime, timedelta
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app
from app import (
    Assignment, BriefingAudit, BriefingDelivery, BriefingItem, FeatureFlag,
    RosterPublication, ShiftType, Staff, Unit, Watch, db,
)


def test_briefing_tables_are_routed_to_the_operational_database():
    assert {
        "briefing_item",
        "briefing_delivery",
        "briefing_audit",
        "briefing_assurance_run",
    }.issubset(app.OPERATIONAL_TABLE_NAMES)


@pytest.fixture()
def briefing_client():
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        unit = Unit(
            id=1, code="BRF", name="Briefing Test Airport",
            active_user_limit=10, onboarding_step=100,
        )
        watch = Watch(unit_id=1, name="Red Watch", order_index=1)
        db.session.add_all([unit, watch])
        db.session.flush()
        admin = Staff(
            unit_id=1, username="brief_admin", name="Brief Admin",
            staff_no="BA-1", role="admin", watch_id=watch.id,
            is_operational=False,
        )
        admin.set_password("password123")
        user = Staff(
            unit_id=1, username="brief_user", name="Brief User",
            staff_no="BU-1", role="user", watch_id=watch.id,
            is_operational=True,
        )
        user.set_password("password123")
        db.session.add_all([
            admin, user,
            FeatureFlag(unit_id=1, key="briefing_module", enabled=True),
            ShiftType(
                unit_id=1, code="M", name="Morning",
                start_time=datetime.strptime("07:00", "%H:%M").time(),
                end_time=datetime.strptime("15:00", "%H:%M").time(),
                is_working=True,
            ),
            ShiftType(
                unit_id=1, code="OFF", name="Off",
                start_time=None, end_time=None, is_working=False,
            ),
        ])
        db.session.commit()
    yield app.app.test_client()
    with app.app.app_context():
        db.session.remove()
        db.drop_all()


def _login(client, username):
    response = client.post(
        "/login",
        data={"username": username, "password": "password123"},
        follow_redirects=True,
    )
    assert response.status_code == 200


def _csrf(client):
    client.get("/modules")
    with client.session_transaction() as session:
        return session["_csrf_token"]


def test_feature_flag_exposes_module_selector(briefing_client):
    _login(briefing_client, "brief_user")
    response = briefing_client.get("/modules")
    assert response.status_code == 200
    assert b"Roster" in response.data
    assert b"Briefing" in response.data


def test_admin_publishes_instruction_and_user_acknowledges(briefing_client):
    _login(briefing_client, "brief_admin")
    now = datetime.now()
    response = briefing_client.post(
        "/briefing/admin",
        data={
            "_csrf_token": _csrf(briefing_client),
            "kind": "instruction",
            "title": "Runway inspection procedure",
            "effective_at": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"),
            "expires_at": (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M"),
            "target_scope": "operational",
            "priority": "important",
            "mandatory": "yes",
            "action": "publish",
            "document": (io.BytesIO(b"%PDF-1.4 test document"), "instruction.pdf"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"published to 1 users" in response.data
    with app.app.app_context():
        item_id = BriefingItem.query.one().id
        assert BriefingDelivery.query.one().recipient_name == "Brief User"

    briefing_client.get("/logout")
    _login(briefing_client, "brief_user")
    page = briefing_client.get(f"/briefing/item/{item_id}")
    assert page.status_code == 200
    assert b"Runway inspection procedure" in page.data
    document = briefing_client.get(f"/briefing/item/{item_id}/document")
    assert document.status_code == 200
    assert document.data.startswith(b"%PDF-")
    response = briefing_client.post(
        f"/briefing/item/{item_id}/acknowledge",
        data={"_csrf_token": _csrf(briefing_client), "confirmation": "yes"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    with app.app.app_context():
        assert BriefingDelivery.query.one().acknowledged_at is not None
        assert BriefingAudit.query.filter_by(event_type="acknowledged").count() == 1


def test_assurance_ignores_non_working_assignments(briefing_client):
    _login(briefing_client, "brief_admin")
    with app.app.app_context():
        user = Staff.query.filter_by(username="brief_user").one()
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=user.id, day=datetime.now().date()
        ).first()
        if assignment:
            assignment.code = "OFF"
        else:
            db.session.add(Assignment(
                unit_id=1, staff_id=user.id, day=datetime.now().date(),
                code="OFF", source="manual",
            ))
        db.session.commit()
        today = datetime.now().date()
        db.session.add(RosterPublication(
            unit_id=1, year=today.year, month=today.month, version=1,
            state="published",
            snapshot_json=json.dumps(app._roster_snapshot(
                today.year, today.month
            )),
            published_at=datetime.now(),
        ))
        db.session.commit()
    response = briefing_client.post(
        "/briefing/admin/assurance",
        data={
            "_csrf_token": _csrf(briefing_client),
            "date": datetime.now().date().isoformat(),
        },
    )
    assert response.status_code == 200
    assert b"No working duties found for this date" in response.data
