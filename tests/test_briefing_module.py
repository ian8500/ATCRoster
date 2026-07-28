import io
import json
from datetime import datetime, timedelta
import os
import sys
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app
from briefing_module import briefing_local_now
from app import (
    Assignment, BriefingAssuranceRun, BriefingAudit, BriefingDelivery,
    BriefingItem, BriefingMessageType, FeatureFlag, RosterPublication,
    ShiftType, Staff, Unit, Watch, db,
)


def test_briefing_tables_are_routed_to_the_operational_database():
    assert {
        "briefing_item",
        "briefing_delivery",
        "briefing_audit",
        "briefing_assurance_run",
        "briefing_message_type",
    }.issubset(app.OPERATIONAL_TABLE_NAMES)


def test_briefing_uses_airport_local_time():
    with app.app.app_context():
        unit = Unit(
            id=99,
            code="TZT",
            name="Timezone Test",
            timezone="Europe/London",
        )
        db.session.add(unit)
        db.session.flush()
        expected = datetime.now(ZoneInfo("Europe/London")).replace(tzinfo=None)
        actual = briefing_local_now(unit.id)
        assert abs((actual - expected).total_seconds()) < 2
        db.session.rollback()


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
            BriefingMessageType(
                unit_id=1, name="Safety instruction", display_order=10,
            ),
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
    client.get("/briefing/admin")
    with client.session_transaction() as session:
        return session["_csrf_token"]


def test_feature_flag_exposes_module_selector(briefing_client):
    response = briefing_client.post(
        "/login",
        data={"username": "brief_user", "password": "password123"},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/modules")
    response = briefing_client.get("/modules")
    assert response.status_code == 200
    assert b"Roster" in response.data
    assert b"Briefing" in response.data
    assert response.data.count(b"<svg") >= 2


def test_module_navigation_keeps_roster_and_briefing_separate(
    briefing_client,
):
    _login(briefing_client, "brief_user")
    today = datetime.now()

    roster = briefing_client.get(
        f"/roster/{today.year}-{today.month:02d}"
    )
    assert roster.status_code == 200
    assert b'href="/modules"' in roster.data
    assert b'href="/briefing/"' not in roster.data

    briefing = briefing_client.get("/briefing/")
    assert briefing.status_code == 200
    assert b'href="/modules"' in briefing.data
    assert b'href="/briefing/"' in briefing.data
    assert b'href="/roster/' not in briefing.data


def test_admin_configures_instruction_message_types(briefing_client):
    _login(briefing_client, "brief_admin")
    publish_page = briefing_client.get("/briefing/admin")
    assert publish_page.status_code == 200
    assert b"Instruction message types</div>" not in publish_page.data
    assert b'href="/briefing/admin/reports"' in publish_page.data
    assert b"briefing-nav__admin-start" in publish_page.data
    assert b"data-briefing-upload-progress" in publish_page.data
    assert b"Uploading briefing" in publish_page.data
    assert b"NOTAM / operational notice" not in publish_page.data
    assert b'name="priority"' not in publish_page.data
    assert publish_page.data.index(b"My briefing") < publish_page.data.index(
        b"Publish"
    )

    legacy = briefing_client.get("/briefing/admin/assurance")
    assert legacy.status_code == 308
    assert legacy.headers["Location"].endswith("/briefing/admin/reports")

    settings_page = briefing_client.get("/briefing/admin/settings")
    assert settings_page.status_code == 200
    assert b"Instruction message types" in settings_page.data

    response = briefing_client.post(
        "/briefing/admin/message-types/configure",
        data={
            "_csrf_token": _csrf(briefing_client),
            "message_types": (
                "Safety instruction\nTechnical instruction\n"
                "Operational notice"
            ),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Technical instruction" in response.data
    assert b"<th>Available</th>" not in response.data
    assert b'name="message_types"' in response.data
    with app.app.app_context():
        active_names = {
            row.name for row in BriefingMessageType.query.filter_by(
                active=True
            ).all()
        }
        assert active_names == {
            "Safety instruction",
            "Technical instruction",
            "Operational notice",
        }


def test_admin_publishes_instruction_and_user_acknowledges(briefing_client):
    _login(briefing_client, "brief_admin")
    now = datetime.now()
    response = briefing_client.post(
        "/briefing/admin",
        data={
            "_csrf_token": _csrf(briefing_client),
            "kind": "instruction",
            "message_type_id": "1",
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
    briefing_home = briefing_client.get("/briefing/")
    assert b"data-briefing-card" in briefing_home.data
    assert b"briefing-card--compact" in briefing_home.data
    assert b'role="link"' in briefing_home.data
    assert b"Open briefing" not in briefing_home.data
    page = briefing_client.get(f"/briefing/item/{item_id}")
    assert page.status_code == 200
    assert b"Runway inspection procedure" in page.data
    assert b"Full screen" in page.data
    assert b"Exit full screen" in page.data
    assert b"webkitExitFullscreen" in page.data
    assert b"displayedSeconds += 1" in page.data
    assert b"navigator.sendBeacon" in page.data
    assert b"Pop out" in page.data
    assert b"Download" in page.data
    assert b"data-pdf-frame" in page.data
    document = briefing_client.get(f"/briefing/item/{item_id}/document")
    assert document.status_code == 200
    assert document.data.startswith(b"%PDF-")
    assert document.mimetype == "application/pdf"
    assert document.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in document.headers["Content-Security-Policy"]
    assert "sandbox" not in document.headers["Content-Security-Policy"]
    assert "inline" in document.headers["Content-Disposition"]
    download = briefing_client.get(
        f"/briefing/item/{item_id}/document?download=1"
    )
    assert download.status_code == 200
    assert "attachment" in download.headers["Content-Disposition"]
    response = briefing_client.post(
        f"/briefing/item/{item_id}/acknowledge",
        data={"_csrf_token": _csrf(briefing_client), "confirmation": "yes"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    with app.app.app_context():
        assert BriefingDelivery.query.one().acknowledged_at is not None
        assert BriefingAudit.query.filter_by(event_type="acknowledged").count() == 1

    response = briefing_client.post(
        f"/briefing/item/{item_id}/archive",
        data={"_csrf_token": _csrf(briefing_client)},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Runway inspection procedure" not in response.data
    archive = briefing_client.get("/briefing/archive")
    assert archive.status_code == 200
    assert b"Safety instruction" in archive.data
    assert b"Runway inspection procedure" in archive.data

    response = briefing_client.post(
        f"/briefing/item/{item_id}/delete",
        data={"_csrf_token": _csrf(briefing_client)},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Runway inspection procedure" not in response.data
    with app.app.app_context():
        delivery = BriefingDelivery.query.one()
        assert delivery.archived_at is not None
        assert delivery.deleted_at is not None
        assert BriefingAudit.query.filter_by(
            event_type="recipient_archived"
        ).count() == 1
        assert BriefingAudit.query.filter_by(
            event_type="recipient_deleted"
        ).count() == 1


def test_brief_of_day_is_displayed_without_open_or_acknowledgement(
    briefing_client,
):
    _login(briefing_client, "brief_admin")
    now = datetime.now()
    response = briefing_client.post(
        "/briefing/admin",
        data={
            "_csrf_token": _csrf(briefing_client),
            "kind": "daily",
            "title": "Today at Glasgow",
            "body": "Operational note " * 80,
            "effective_at": (
                now - timedelta(hours=1)
            ).strftime("%Y-%m-%dT%H:%M"),
            "expires_at": (
                now + timedelta(days=1)
            ).strftime("%Y-%m-%dT%H:%M"),
            "target_scope": "all",
            "priority": "routine",
            "mandatory": "yes",
            "action": "publish",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    with app.app.app_context():
        item = BriefingItem.query.one()
        assert item.kind == "daily"
        assert item.mandatory is False

    briefing_client.get("/logout")
    _login(briefing_client, "brief_user")
    page = briefing_client.get("/briefing/")
    assert b"Today at Glasgow" in page.data
    assert b"Briefs of the Day" in page.data
    assert b"Mandatory Messages" in page.data
    assert b"Other Messages" in page.data
    assert b"data-daily-expand" in page.data
    assert b'role="link"' not in page.data
    assert b"Mandatory</span>" not in page.data
    assert b">Open briefing" not in page.data
    assert b"nav-attention-count" not in page.data


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
        "/briefing/admin/reports",
        data={
            "_csrf_token": _csrf(briefing_client),
            "date": datetime.now().date().isoformat(),
        },
    )
    assert response.status_code == 200
    assert b"Login and roster activity" in response.data
    assert b"On duty with unread mandatory messages" in response.data
    assert b"Unread instructions by user" in response.data
    assert b"Read instructions and active reading time" in response.data
    assert b"No on-duty users have unread mandatory messages" in response.data
    with app.app.app_context():
        saved = json.loads(BriefingAssuranceRun.query.one().result_json)
        assert set(saved) == {
            "login_roster",
            "on_duty_mandatory",
            "read_profiles",
            "unread_profiles",
        }


def test_assurance_reports_on_duty_and_profile_unread_items(
    briefing_client,
):
    _login(briefing_client, "brief_admin")
    today = datetime.now().date()
    with app.app.app_context():
        user = Staff.query.filter_by(username="brief_user").one()
        db.session.add(Assignment(
            unit_id=1,
            staff_id=user.id,
            day=today,
            code="M",
            source="manual",
        ))
        db.session.flush()
        db.session.add(RosterPublication(
            unit_id=1,
            year=today.year,
            month=today.month,
            version=1,
            state="published",
            snapshot_json=json.dumps(app._roster_snapshot(
                today.year, today.month
            )),
            published_at=datetime.now(),
        ))
        item = BriefingItem(
            unit_id=1,
            kind="instruction",
            title="Mandatory safety update",
            message_type_name="Safety instruction",
            effective_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(days=1),
            mandatory=True,
            priority="critical",
            status="published",
            created_by_id=1,
            created_by_name="Brief Admin",
        )
        db.session.add(item)
        db.session.flush()
        db.session.add(BriefingDelivery(
            unit_id=1,
            briefing_id=item.id,
            recipient_id=user.id,
            recipient_name=user.name,
        ))
        db.session.commit()

    response = briefing_client.post(
        "/briefing/admin/reports",
        data={
            "_csrf_token": _csrf(briefing_client),
            "date": today.isoformat(),
        },
    )
    assert response.status_code == 200
    assert b"Mandatory safety update" in response.data
    assert b"1 unread instruction" in response.data
    assert b"Different" in response.data
