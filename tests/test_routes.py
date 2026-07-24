import os
import sys
import tempfile
from datetime import date, time

import pytest
import pyotp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "atc_roster_test.db")
# Ensure a clean database path before importing the app module
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import app  # noqa: E402
from app import (
    Watch,
    Staff,
    ShiftType,
    StaffWatchHistory,
    Unit,
    ensure_month_requirement,
    generate_month,
    refresh_shift_cache,
    db,
)


ADMIN_CREDENTIALS = {"username": "admin_test", "password": "password123"}


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    with app.app.app_context():
        db.drop_all()
        db.create_all()

        db.session.add(Unit(
            id=1, code="TST", name="Test Airport", active_user_limit=20
        ))
        watch_a = Watch(name="Watch A", order_index=1)
        watch_b = Watch(name="Watch B", order_index=2)
        db.session.add_all([watch_a, watch_b])

        shifts = [
            ShiftType(code="M", name="Morning", start_time=time(7, 0), end_time=time(15, 0), is_working=True),
            ShiftType(code="D", name="Day", start_time=time(9, 0), end_time=time(17, 0), is_working=True),
            ShiftType(code="A", name="Afternoon", start_time=time(13, 0), end_time=time(21, 0), is_working=True),
            ShiftType(code="N", name="Night", start_time=time(21, 0), end_time=time(5, 0), is_working=True),
            ShiftType(code="OFF", name="Off", start_time=None, end_time=None, is_working=False),
        ]
        db.session.add_all(shifts)
        db.session.commit()
        refresh_shift_cache()

        admin = Staff(
            username=ADMIN_CREDENTIALS["username"],
            name="Admin Test",
            staff_no="ADM-001",
            role="admin",
            watch=watch_a,
            pattern_csv="M,OFF",
        )
        admin.set_password(ADMIN_CREDENTIALS["password"])
        db.session.add(admin)
        role_users = [
            Staff(
                unit_id=1, username="editor_test", name="Editor Test",
                staff_no="ED-001", role="editor", watch=watch_a,
                is_operational=True,
            ),
            Staff(
                unit_id=1, username="watch_manager_test", name="Watch Manager Test",
                staff_no="WM-001", role="user", watch=watch_a,
                is_wm=True, is_operational=True,
            ),
            Staff(
                unit_id=1, username="duty_watch_manager_test",
                name="Duty Watch Manager Test", staff_no="DWM-001",
                role="user", watch=watch_b, is_dwm=True,
                is_operational=True,
            ),
            Staff(
                unit_id=1, username="staff_test", name="Staff Test",
                staff_no="USR-001", role="user", watch=watch_b,
                is_operational=True,
            ),
        ]
        for user in role_users:
            user.set_password("password123")
        db.session.add_all(role_users)

        control = Unit(
            id=2, code="CTRL", name="Platform Control",
            status="platform_control", active_user_limit=5,
        )
        other_unit = Unit(
            id=3, code="OTH", name="Other Airport", active_user_limit=5,
        )
        db.session.add_all([control, other_unit])
        db.session.flush()
        platform_user = Staff(
            unit_id=control.id, username="platform_test",
            name="Platform Test", staff_no="CTRL-001",
            role="superadmin", is_operational=False,
        )
        platform_user.set_password("password123")
        other_user = Staff(
            unit_id=other_unit.id, username="other_staff_test",
            name="Other Airport Staff", staff_no="OTH-001",
            role="user", is_operational=True,
        )
        other_user.set_password("password123")
        db.session.add_all([platform_user, other_user])
        db.session.flush()
        db.session.add(app.PlatformIdentity(
            public_id="platform-role-test",
            username=platform_user.username,
            password_hash=platform_user.password_hash,
        ))
        db.session.commit()

        ensure_month_requirement(2025, 4)
        generate_month(2025, 4)

    yield

    with app.app.app_context():
        db.session.remove()
        db.drop_all()

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture()
def client():
    return app.app.test_client()


def login(client):
    response = client.post(
        "/login",
        data={"username": ADMIN_CREDENTIALS["username"], "password": ADMIN_CREDENTIALS["password"]},
        follow_redirects=True,
    )
    assert response.status_code == 200
    return response


def csrf(client):
    client.get("/publications/2025-04")
    with client.session_transaction() as sess:
        return sess["_csrf_token"]


def test_login_page_loads(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Login" in resp.data
    assert b"Skip to main content" in resp.data
    assert b'class="nav-toggle"' in resp.data
    assert b'data-password-toggle="login-password"' in resp.data


def test_friendly_error_pages_preserve_status_codes(client):
    missing = client.get("/this-page-does-not-exist")
    assert missing.status_code == 404
    assert b"That page or record was not found" in missing.data

    login(client)
    expired_form = client.post(
        "/leave",
        data={
            "form": "leave_add",
            "staff_id": "1",
            "leave_type": "AL",
            "start": "2025-04-01",
            "end": "2025-04-01",
        },
    )
    assert expired_form.status_code == 400
    assert b"page or form has expired" in expired_form.data


def test_roster_has_persistent_zoom_presets(client):
    login(client)
    response = client.get("/roster/2025-04")
    assert response.status_code == 200
    assert b'data-roster-zoom="0.75"' in response.data
    assert b'data-roster-zoom="0.90"' in response.data
    assert b'data-roster-zoom="1"' in response.data
    assert b'data-roster-zoom="fit"' in response.data
    assert b"code-input code-len-3" in response.data
    assert b"shift on 01 April 2025" in response.data
    assert b"Active unit" in response.data
    assert b"data-operational-clock" in response.data
    assert b"Secure session" in response.data

    stylesheet = client.get("/static/styles.css")
    assert stylesheet.status_code == 200
    assert b".roster .cell select.code-input.off" in stylesheet.data
    assert b"background: var(--off-blue)" in stylesheet.data
    assert b"select.code-input.code-len-5" in stylesheet.data
    assert b"-webkit-appearance:none" in stylesheet.data


def test_favicon_is_served(client):
    resp = client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"


def test_compliance_centre_and_evidence_export(client):
    login(client)
    page = client.get("/compliance-centre?ym=2025-04")
    assert page.status_code == 200
    assert b"Fatigue &amp; Compliance Centre" in page.data
    export = client.get("/compliance-centre/export?ym=2025-04")
    assert export.status_code == 200
    assert export.mimetype == "text/csv"
    assert b"Airport,Month,ATCO" in export.data


def test_roster_publication_and_acknowledgement(client):
    login(client)
    token = csrf(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        position = app.OperationalPosition(
            unit_id=admin.unit_id, code="PUB", label="Publication test position"
        )
        db.session.add(position)
        db.session.flush()
        db.session.add(app.PositionRequirement(
            unit_id=admin.unit_id, day=date(2025, 4, 1), shift_code="M",
            position_id=position.id, required_count=0, contingency_count=0,
        ))
        db.session.add(app.BreakPlan(
            unit_id=admin.unit_id, day=date(2025, 4, 1),
            person_id=admin.id, start_time=time(10, 0), end_time=time(10, 30),
            recorded_by_id=admin.id,
        ))
        db.session.add(app.RosterRuleVersion(
            unit_id=admin.unit_id, version=1, name="Approved test rules",
            rules_json="{}", state="approved", effective_from=date(2025, 1, 1),
            change_reference="TEST-001",
            consultation_summary="Approved test consultation evidence.",
            approved_by_id=admin.id, approved_at=app.utcnow(),
        ))
        db.session.commit()
    rejected = client.post(
        "/publications/2025-04",
        data={"_csrf_token": token, "action": "publish"},
        follow_redirects=True,
    )
    assert b"release declaration" in rejected.data
    published = client.post(
        "/publications/2025-04",
        data={
            "_csrf_token": token,
            "action": "publish",
            "release_declaration": "yes",
            "exception_reason": (
                "Operational manager reviewed staffing and fatigue exceptions "
                "with mitigations in place."
            ),
        },
        follow_redirects=True,
    )
    assert published.status_code == 200
    assert b"Version 1" in published.data
    with app.app.app_context():
        publication = app.RosterPublication.query.filter_by(
            year=2025, month=4, state="published"
        ).first()
        assert publication is not None
        snapshot = app.json.loads(publication.snapshot_json)
        assert snapshot["release_assurance"]["declared_by_id"]
        assert snapshot["release_assurance"]["exception_reason"]
        publication_id = publication.id
    acknowledged = client.post(
        "/publications/2025-04",
        data={
            "_csrf_token": token,
            "action": "acknowledge",
            "publication_id": publication_id,
        },
        follow_redirects=True,
    )
    assert acknowledged.status_code == 200
    assert b"You acknowledged this version" in acknowledged.data


def test_security_headers_are_present(client):
    response = client.get("/login")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_role_permission_matrix_and_cross_airport_isolation():
    credentials = {
        "superadmin": "platform_test",
        "admin": ADMIN_CREDENTIALS["username"],
        "editor": "editor_test",
        "watch_manager": "watch_manager_test",
        "duty_watch_manager": "duty_watch_manager_test",
        "staff": "staff_test",
    }
    common = {
        "roster": "/roster/2025-04",
        "requests": "/requests",
        "published": "/publications/2025-04",
        "fatigue": "/fatigue/report",
        "overtime": "/overtime",
        "leave": "/leave",
        "reports": "/reports",
        "metrics": "/metrics",
        "qualification": "/compliance",
        "compliance": "/compliance-centre?ym=2025-04",
        "operations": "/operations/2025-04",
        "coverage": "/planning/coverage/2025-04",
        "scenarios": "/planning/scenarios",
        "accounts": "/unit/accounts",
        "onboarding": "/unit/onboarding",
        "admin": "/admin",
        "reference": "/admin/reference",
        "platform": "/platform/admin",
    }
    expected = {
        "superadmin": {
            **{name: 403 for name in common},
            "platform": 200,
        },
        "admin": {
            **{name: 200 for name in common},
            "platform": 403,
        },
        "editor": {
            "roster": 200, "requests": 200, "published": 200,
            "fatigue": 200, "overtime": 200, "leave": 200,
            "reports": 302, "metrics": 200, "qualification": 200,
            "compliance": 403, "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "watch_manager": {
            "roster": 200, "requests": 200, "published": 200,
            "fatigue": 200, "overtime": 403, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "compliance": 403, "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "duty_watch_manager": {
            "roster": 200, "requests": 200, "published": 200,
            "fatigue": 200, "overtime": 403, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "compliance": 403, "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "staff": {
            "roster": 200, "requests": 200, "published": 200,
            "fatigue": 200, "overtime": 403, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "compliance": 403, "operations": 403, "coverage": 403,
            "scenarios": 403, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
    }

    clients = {}
    for role, username in credentials.items():
        role_client = app.app.test_client()
        response = role_client.post(
            "/login",
            data={"username": username, "password": "password123"},
        )
        assert response.status_code == 302
        clients[role] = role_client
        for capability, path in common.items():
            actual = role_client.get(path).status_code
            assert actual == expected[role][capability], (
                role, capability, actual, expected[role][capability]
            )

    assert clients["superadmin"].get("/").headers["Location"].endswith(
        "/platform/admin"
    )
    platform_denial = clients["superadmin"].get("/roster/2025-04")
    assert b"Return to Platform Administration" in platform_denial.data
    assert b"ask your Unit Administrator" not in platform_denial.data

    with app.app.app_context():
        admin = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        same_unit_other = Staff.query.filter_by(username="staff_test").one()
        other_airport = db.session.query(Staff).execution_options(
            skip_tenant_scope=True
        ).filter_by(username="other_staff_test").one()

    assert clients["staff"].get(
        f"/staff/{same_unit_other.id}"
    ).status_code == 200
    assert clients["staff"].get(f"/staff/{admin.id}").status_code == 403
    assert clients["admin"].get(f"/staff/{same_unit_other.id}").status_code == 200
    assert clients["admin"].get(f"/staff/{other_airport.id}").status_code == 404

    target_day = "2025-04-02"
    assert clients["staff"].post(
        f"/assign/{admin.id}/2025-04/{target_day}",
        data={"_csrf_token": csrf(clients["staff"]), "code": "A"},
    ).status_code == 403
    assert clients["watch_manager"].post(
        f"/assign/{admin.id}/2025-04/{target_day}",
        data={
            "_csrf_token": csrf(clients["watch_manager"]),
            "code": "A",
        },
    ).status_code == 302
    assert clients["duty_watch_manager"].post(
        f"/assign/{admin.id}/2025-04/{target_day}",
        data={
            "_csrf_token": csrf(clients["duty_watch_manager"]),
            "code": "N",
        },
    ).status_code == 302
    assert clients["editor"].post(
        f"/assign/{admin.id}/2025-04/{target_day}",
        data={"_csrf_token": csrf(clients["editor"]), "code": "M"},
    ).status_code == 302


def test_health_endpoints_report_ready(client):
    assert client.get("/health/live").get_json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.get_json()["status"] == "ready"


def test_overtime_finder_reports_an_empty_search_instead_of_looking_broken(client):
    login(client)
    token = csrf(client)
    response = client.post(
        "/overtime",
        data={
            "_csrf_token": token,
            "action": "find",
            "date": "2025-04-01",
            "shift_code": "M",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Eligibility result" in response.data
    assert b"Nobody is eligible for overtime for this date and shift" in response.data


def test_production_operations_and_fatigue_workflows(client):
    login(client)
    token = csrf(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        admin_id = admin.id

    position_response = client.post(
        "/operations/2025-04",
        data={
            "_csrf_token": token, "action": "create_position",
            "code": "TWR", "label": "Tower Controller",
            "description": "Aerodrome control position",
            "is_safety_critical": "on",
        },
        follow_redirects=True,
    )
    assert position_response.status_code == 200
    with app.app.app_context():
        position = app.OperationalPosition.query.filter_by(code="TWR").first()
        assert position is not None
        position_id = position.id

    endorsement_response = client.post(
        "/operations/2025-04",
        data={
            "_csrf_token": token, "action": "grant_endorsement",
            "person_id": admin_id, "position_id": position_id,
            "valid_from": "2025-01-01", "valid_until": "2026-12-31",
            "restrictions": "",
        },
        follow_redirects=True,
    )
    assert b"Operational assurance record saved" in endorsement_response.data

    requirement_response = client.post(
        "/operations/2025-04",
        data={
            "_csrf_token": token, "action": "set_position_requirement",
            "day": "2025-04-01", "shift_code": "M",
            "position_id": position_id, "required_count": "1",
            "contingency_count": "1",
        },
        follow_redirects=True,
    )
    assert requirement_response.status_code == 200

    break_response = client.post(
        "/operations/2025-04",
        data={
            "_csrf_token": token, "action": "add_break",
            "day": "2025-04-01", "person_id": admin_id,
            "position_id": position_id, "start_time": "10:00",
            "end_time": "10:30", "kind": "break",
        },
        follow_redirects=True,
    )
    assert break_response.status_code == 200

    fatigue_response = client.post(
        "/fatigue/report",
        data={
            "_csrf_token": token, "duty_day": "2025-04-01",
            "severity": "medium",
            "summary": "Reduced sleep before the planned morning duty.",
        },
        follow_redirects=True,
    )
    assert b"Fatigue report submitted" in fatigue_response.data
    with app.app.app_context():
        assert app.PositionEndorsement.query.count() == 1
        assert app.PositionRequirement.query.filter_by(
            position_id=position_id
        ).count() == 1
        assert app.BreakPlan.query.filter_by(position_id=position_id).count() == 1
        assert app.FatigueReport.query.count() == 1


def test_index_redirects_to_roster(client):
    login(client)
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/roster/" in resp.headers["Location"]


def test_roster_routes_render(client):
    login(client)
    month = "2025-04"
    roster_resp = client.get(f"/roster/{month}")
    assert roster_resp.status_code == 200
    export_resp = client.get(f"/roster/{month}/export")
    assert export_resp.status_code == 200
    assert export_resp.mimetype == "text/csv"


def test_admin_pages_accessible(client):
    login(client)
    endpoints = [
        "/admin",
        "/leave",
        "/metrics",
        "/reports",
        "/requests",
        "/admin/toil/new",
        "/metrics/export",
    ]
    for url in endpoints:
        resp = client.get(url)
        assert resp.status_code == 200, f"Endpoint {url} returned {resp.status_code}"


def test_admin_can_configure_requestable_shift(client):
    login(client)
    with app.app.app_context():
        shift = ShiftType.query.filter_by(code="M").one()
        shift_id = shift.id

    response = client.post(
        "/admin",
        data={
            "form": "shift_edit",
            "shift_id": shift_id,
            "name": "Morning",
            "start": "07:00",
            "end": "15:00",
            "is_working": "on",
            "is_active": "on",
            "is_requestable": "on",
            "required_qualification": "medical",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Shift updated" in response.data
    assert b"Requestable" in response.data
    assert b"Required qualification" in response.data

    with app.app.app_context():
        shift = db.session.get(ShiftType, shift_id)
        assert shift.is_active is True
        assert shift.is_requestable is True
        assert shift.required_qualification == "medical"


def test_non_working_shift_cannot_be_requestable(client):
    login(client)
    with app.app.app_context():
        shift = ShiftType.query.filter_by(code="OFF").one()
        shift_id = shift.id

    response = client.post(
        "/admin",
        data={
            "form": "shift_edit",
            "shift_id": shift_id,
            "name": "Off",
            "is_active": "on",
            "is_requestable": "on",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Only active working shifts can be requestable" in response.data

    with app.app.app_context():
        assert db.session.get(ShiftType, shift_id).is_requestable is False


def test_admin_staff_edit_handles_missing_watch_history(client):
    with app.app.app_context():
        staff = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        hist = StaffWatchHistory(
            staff_id=staff.id,
            watch_id=9999,  # invalid watch id to simulate stale pending move
            effective_date=date(2025, 6, 1),
        )
        db.session.add(hist)
        db.session.commit()
        hist_id = hist.id

    login(client)
    resp = client.get(f"/admin/staff/{staff.id}")
    assert resp.status_code == 200
    assert b"Unknown watch" in resp.data

    with app.app.app_context():
        StaffWatchHistory.query.filter_by(id=hist_id).delete()
        db.session.commit()


def test_admin_watch_move_flow(client):
    login(client)

    with app.app.app_context():
        staff = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        watch_a = Watch.query.filter_by(name="Watch A").first()
        watch_b = Watch.query.filter_by(name="Watch B").first()

    # Create a new watch move and ensure redirect ends on staff edit page
    create_resp = client.post(
        f"/admin/staff/{staff.id}/watch-move",
        data={"watch_id": watch_b.id, "effective_date": "2025-06-01"},
        follow_redirects=True,
    )
    assert create_resp.status_code == 200
    assert b"Watch move recorded" in create_resp.data

    with app.app.app_context():
        history = (
            StaffWatchHistory.query.filter_by(staff_id=staff.id)
            .order_by(StaffWatchHistory.effective_date.desc())
            .all()
        )
        assert history, "Watch move history should exist after creating a move"
        entry = history[0]
        assert entry.watch_id == watch_b.id

    # Update the existing watch move to a different watch and effective date
    update_resp = client.post(
        f"/admin/staff/watch-move/{entry.id}/edit",
        data={"watch_id": watch_a.id, "effective_date": "2025-07-01"},
        follow_redirects=True,
    )
    assert update_resp.status_code == 200
    assert b"Watch move updated" in update_resp.data

    with app.app.app_context():
        updated_entry = db.session.get(StaffWatchHistory, entry.id)
        assert updated_entry is not None
        assert updated_entry.watch_id == watch_a.id
        assert str(updated_entry.effective_date) == "2025-07-01"

    # Delete the watch move and ensure other admin links still load
    delete_resp = client.post(
        f"/admin/staff/watch-move/{entry.id}/delete",
        follow_redirects=True,
    )
    assert delete_resp.status_code == 200
    assert b"Watch move deleted" in delete_resp.data

    with app.app.app_context():
        assert db.session.get(StaffWatchHistory, entry.id) is None

    # Sanity check that the main admin dashboard still renders after the flow
    admin_resp = client.get("/admin")
    assert admin_resp.status_code == 200


def test_mfa_challenge_completes_login(client):
    client.get("/logout")
    secret = pyotp.random_base32()
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        app.MfaCredential.query.filter_by(person_id=admin.id).delete()
        db.session.add(app.MfaCredential(
            unit_id=admin.unit_id,
            person_id=admin.id,
            encrypted_secret=app._field_cipher().encrypt(secret.encode()).decode(),
            enabled=True,
            enrolled_at=app.utcnow(),
        ))
        db.session.commit()
    password_step = client.post(
        "/login", data=ADMIN_CREDENTIALS, follow_redirects=False
    )
    assert password_step.status_code == 302
    assert "/login/mfa" in password_step.headers["Location"]
    challenge = client.get("/login/mfa")
    assert challenge.status_code == 200
    with client.session_transaction() as sess:
        token = sess["_csrf_token"]
    verified = client.post(
        "/login/mfa",
        data={"_csrf_token": token, "code": pyotp.TOTP(secret).now()},
        follow_redirects=False,
    )
    assert verified.status_code == 302
    assert "/roster/" in client.get("/", follow_redirects=False).headers["Location"]
