import io
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pyotp
import pytest
from sqlalchemy.exc import IntegrityError
from conftest import finish_operational_login

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "atc_roster_test.db")
# Ensure a clean database path before importing the app module
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import app
import fatigue_compliance
import fatigue_engine
from app import (
    AnnotationAudit,
    AnnotationType,
    Assignment,
    ChangeLog,
    Leave,
    ShiftType,
    Staff,
    StaffWatchHistory,
    QualificationType,
    Unit,
    Watch,
    db,
    ensure_month_requirement,
    generate_month,
    refresh_shift_cache,
)

ADMIN_CREDENTIALS = {"username": "admin_test", "password": "password123"}


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/privacy", b"ATCRoster privacy notice"),
        ("/cookies", b"Cookie and local-storage notice"),
        ("/terms", b"ATCRoster user terms"),
        ("/subprocessors", b"ATCRoster subprocessors"),
    ],
)
def test_public_legal_pages(client, path, expected):
    response = client.get(path)
    assert response.status_code == 200
    assert expected in response.data


def test_privacy_notice_identifies_operator_and_contact(client):
    response = client.get("/privacy")
    assert response.status_code == 200
    assert b"Ian John Dickson trading as IDAviation" in response.data
    assert b"Flat 0/2, 24 Caird Drive" in response.data
    assert b"privacy@atcroster.com" in response.data


def test_public_shell_uses_local_professional_branding(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert b'<span class="brand-mark" aria-hidden="true"><span></span>' in response.data
    assert b"fonts.googleapis.com" not in response.data


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    with app.app.app_context():
        db.drop_all()
        db.create_all()

        db.session.add(Unit(
            id=1, code="TST", name="Test Airport", active_user_limit=20,
            onboarding_step=100,
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
        db.session.add(QualificationType(
            unit_id=1, code="MEDICAL", label="Medical",
            expiry_required=True, is_active=True,
        ))
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
                permissions_json=json.dumps({
                    "edit_roster": True, "apply_annotations": True,
                }),
            ),
            Staff(
                unit_id=1, username="duty_watch_manager_test",
                name="Duty Watch Manager Test", staff_no="DWM-001",
                role="user", watch=watch_b, is_dwm=True,
                is_operational=True,
                permissions_json=json.dumps({
                    "edit_roster": True, "apply_annotations": True,
                }),
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
        db.session.flush()
        for user in (admin, *role_users):
            identity = app.PlatformIdentity(
                public_id=f"test-{user.username}",
                username=user.username,
                password_hash=user.password_hash,
            )
            db.session.add(identity)
            db.session.flush()
            db.session.add(app.UnitMembership(
                identity_id=identity.id,
                unit_id=user.unit_id,
                person_id=user.id,
                role={
                    "admin": "UnitAdmin",
                    "editor": "RosterEditor",
                }.get(user.role, "StaffUser"),
                status="active",
            ))

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
    return login_as(
        client,
        ADMIN_CREDENTIALS["username"],
        ADMIN_CREDENTIALS["password"],
        follow_redirects=True,
    )


def login_as(client, username, password="password123", **kwargs):
    client.get("/login")
    with client.session_transaction() as sess:
        token = sess["_csrf_token"]
    response = client.post(
        "/login",
        data={
            "_csrf_token": token,
            "username": username,
            "password": password,
        },
        **kwargs,
    )
    expected_status = 200 if kwargs.get("follow_redirects") else 302
    assert response.status_code == expected_status
    if username != "platform_test":
        finish_operational_login(client)
    return response


def csrf(client):
    client.get("/login")
    with client.session_transaction() as sess:
        return sess["_csrf_token"]


def acknowledge_reports(client):
    warning = client.get("/reports")
    assert warning.status_code == 200
    assert b"Sensitive information ahead" in warning.data
    token = csrf(client)
    response = client.post(
        "/reports",
        data={"_csrf_token": token},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/reports")


def copy_authenticated_session(source_client, target_client):
    with source_client.session_transaction() as source_session:
        values = dict(source_session)
    with target_client.session_transaction() as target_session:
        target_session.clear()
        target_session.update(values)


def _clear_flexible_patterns(unit_id=1):
    app.StaffRule.query.filter_by(unit_id=unit_id).delete()
    app.StaffPatternAssignment.query.filter_by(unit_id=unit_id).delete()
    day_ids = [
        row.id for row in app.WorkPatternDay.query.filter_by(unit_id=unit_id).all()
    ]
    if day_ids:
        app.WorkPatternDayAllowedShift.query.filter(
            app.WorkPatternDayAllowedShift.work_pattern_day_id.in_(day_ids)
        ).delete(synchronize_session=False)
    app.WorkPatternDay.query.filter_by(unit_id=unit_id).delete()
    app.WorkPattern.query.filter_by(unit_id=unit_id).delete()
    db.session.commit()


def test_reports_require_sensitive_data_acknowledgement(client):
    login(client)
    warning = client.get("/reports")
    assert warning.status_code == 200
    assert b"Sensitive information ahead" in warning.data
    assert b"Check your surroundings" in warning.data
    assert b"OK, open reports" in warning.data
    assert b"Leave-Year Summary" not in warning.data

    direct_report = client.get("/reports/leave-year")
    assert direct_report.status_code == 302
    assert direct_report.headers["Location"].endswith("/reports")

    acknowledge_reports(client)
    hub = client.get("/reports")
    assert hub.status_code == 200
    assert b"Leave-Year Summary" in hub.data
    opened = client.get("/reports/leave-year")
    assert opened.status_code == 200
    assert b"Leave year" in opened.data
    assert b'<body class="app-body report-page">' in opened.data
    assert b'data-command="print-report"' in opened.data
    assert b'data-command="save-report-pdf"' in opened.data
    assert b"choose <strong>Save as PDF</strong>" in opened.data

    # Returning to the Reports tab is a new entry and must require a fresh
    # privacy acknowledgement.
    returned = client.get("/reports")
    assert returned.status_code == 200
    assert b"Sensitive information ahead" in returned.data
    assert b"Leave-Year Summary" not in returned.data

    direct_after_return = client.get("/reports/leave-year")
    assert direct_after_return.status_code == 302
    assert direct_after_return.headers["Location"].endswith("/reports")


def test_leave_year_report_filters_by_watch(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        watch_a = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        watch_b = Watch.query.filter_by(unit_id=1, name="Watch B").one()

    all_users = client.get("/reports/leave-year")
    assert all_users.status_code == 200
    assert b"Entire unit" in all_users.data
    assert b"Apply filter" in all_users.data
    assert b"<td>Admin Test</td>" in all_users.data
    assert b"<td>Duty Watch Manager Test</td>" in all_users.data

    watch_a_only = client.get(f"/reports/leave-year?watch_id={watch_a.id}")
    assert watch_a_only.status_code == 200
    assert b"<td>Admin Test</td>" in watch_a_only.data
    assert b"<td>Duty Watch Manager Test</td>" not in watch_a_only.data

    watch_b_only = client.get(f"/reports/leave-year?watch_id={watch_b.id}")
    assert watch_b_only.status_code == 200
    assert b"<td>Duty Watch Manager Test</td>" in watch_b_only.data
    assert b"<td>Admin Test</td>" not in watch_b_only.data


def test_annotation_totals_report_and_export_filter_by_watch(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        watch_a = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        watch_b = Watch.query.filter_by(unit_id=1, name="Watch B").one()

    entire_unit = client.get("/metrics")
    assert entire_unit.status_code == 200
    assert b"Entire unit" in entire_unit.data
    assert b"Admin Test" in entire_unit.data
    assert b"Duty Watch Manager Test" in entire_unit.data

    watch_a_only = client.get(f"/metrics?watch_id={watch_a.id}")
    assert watch_a_only.status_code == 200
    assert b"Admin Test" in watch_a_only.data
    assert b"Duty Watch Manager Test" not in watch_a_only.data
    assert f"watch_id={watch_a.id}".encode() in watch_a_only.data

    watch_b_export = client.get(f"/metrics/export?watch_id={watch_b.id}")
    assert watch_b_export.status_code == 200
    csv_text = watch_b_export.data.decode()
    assert "Duty Watch Manager Test" in csv_text
    assert "Admin Test" not in csv_text
    assert f"watch-{watch_b.id}" in watch_b_export.headers["Content-Disposition"]


def test_leave_year_report_uses_selected_end_date_and_coloured_balances(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        person = Staff.query.filter_by(
            unit_id=1, username=ADMIN_CREDENTIALS["username"]
        ).one()
        original_values = (
            person.leave_year_start_month,
            person.leave_entitlement_days,
            person.leave_public_holidays,
            person.leave_carryover_days,
            person.toil_half_days,
        )
        person.leave_year_start_month = 4
        person.leave_entitlement_days = 20
        person.leave_public_holidays = 5
        person.leave_carryover_days = 2
        person.toil_half_days = 3
        assignments = [
            Assignment(
                unit_id=1,
                staff_id=person.id,
                day=date(2027, 5, day),
                code="AL",
                source="manual",
            )
            for day in (5, 20)
        ]
        db.session.add_all(assignments)
        db.session.commit()
        person_id = person.id
        assignment_ids = [row.id for row in assignments]

    early = client.get("/reports/leave-year?end_date=2027-05-10")
    later = client.get("/reports/leave-year?end_date=2027-05-25")

    assert early.status_code == 200
    assert b'input id="leave-year-end" type="date" name="end_date" value="2027-05-10"' in early.data
    assert b"Entitlement and balances as of 2027-05-10." in early.data
    assert f'data-staff-id="{person_id}" data-al-taken="1" data-leave-remaining="26"'.encode() in early.data
    assert f'data-staff-id="{person_id}" data-al-taken="2" data-leave-remaining="25"'.encode() in later.data
    assert b"leave-year-col--remaining balance-positive" in early.data
    assert b"leave-year-col--toil-balance balance-positive" in early.data
    assert b'leave-year-col--taken">AL taken</th><th></th>' not in early.data
    assert b'<td class="ta-right leave-year-col leave-year-col--taken">1</td>\n          <td></td>' not in early.data
    assert client.get("/reports/leave-year?end_date=not-a-date").status_code == 400

    with app.app.app_context():
        Assignment.query.filter(Assignment.id.in_(assignment_ids)).delete()
        person = db.session.get(Staff, person_id)
        (
            person.leave_year_start_month,
            person.leave_entitlement_days,
            person.leave_public_holidays,
            person.leave_carryover_days,
            person.toil_half_days,
        ) = original_values
        db.session.commit()


def test_sickness_report_filters_by_watch(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        watch_a = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        watch_b = Watch.query.filter_by(unit_id=1, name="Watch B").one()
        admin = Staff.query.filter_by(unit_id=1, username="admin_test").one()
        dwm = Staff.query.filter_by(
            unit_id=1, username="duty_watch_manager_test"
        ).one()
        today = date.today()
        for person in (admin, dwm):
            assignment = Assignment.query.filter_by(
                unit_id=1, staff_id=person.id, day=today
            ).first()
            if assignment is None:
                assignment = Assignment(
                    unit_id=1, staff_id=person.id, day=today, source="manual"
                )
                db.session.add(assignment)
            assignment.code = "SC"
        db.session.commit()

    all_users = client.get("/reports/sickness")
    assert all_users.status_code == 200
    assert b"Entire unit" in all_users.data
    assert b"Apply filter" in all_users.data
    assert b"<td>Admin Test</td>" in all_users.data
    assert b"<td>Duty Watch Manager Test</td>" in all_users.data

    watch_a_only = client.get(f"/reports/sickness?watch_id={watch_a.id}")
    assert watch_a_only.status_code == 200
    assert b"<td>Admin Test</td>" in watch_a_only.data
    assert b"<td>Duty Watch Manager Test</td>" not in watch_a_only.data

    watch_b_only = client.get(f"/reports/sickness?watch_id={watch_b.id}")
    assert watch_b_only.status_code == 200
    assert b"<td>Duty Watch Manager Test</td>" in watch_b_only.data
    assert b"<td>Admin Test</td>" not in watch_b_only.data


def test_sickness_days_are_grouped_into_continuous_instances():
    person = SimpleNamespace(name="Test ATCO")
    rows = [
        SimpleNamespace(
            staff_id=7, staff=person, day=date(2025, 4, 1), code="SC"
        ),
        SimpleNamespace(
            staff_id=7, staff=person, day=date(2025, 4, 2), code="SSC"
        ),
        SimpleNamespace(
            staff_id=7, staff=person, day=date(2025, 4, 4), code="SC"
        ),
    ]
    instances = app._group_sickness_instances(
        rows, date(2025, 4, 1), date(2025, 4, 30)
    )
    assert len(instances) == 2
    assert instances[0]["duration"] == 2
    assert instances[0]["codes"] == ["SC", "SSC"]
    assert instances[1]["start"] == date(2025, 4, 4)


def test_login_page_loads(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Login" in resp.data
    assert b"Skip to main content" in resp.data
    assert b'class="nav-toggle"' in resp.data
    assert b'data-password-toggle="login-password"' in resp.data
    assert b'name="_csrf_token"' in resp.data
    assert b'class="container container--xs py-5 login-page"' in resp.data


def test_login_rejects_missing_and_invalid_csrf_tokens(client):
    missing = client.post("/login", data=ADMIN_CREDENTIALS)
    assert missing.status_code == 400

    client.get("/login")
    invalid = client.post(
        "/login",
        data={"_csrf_token": "invalid", **ADMIN_CREDENTIALS},
    )
    assert invalid.status_code == 400


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("unknown-central-identity", "irrelevant"),
        (ADMIN_CREDENTIALS["username"], "wrong-password"),
    ],
)
def test_login_returns_same_generic_error_for_unknown_and_wrong_credentials(
    client, username, password,
):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    response = client.post(
        "/login",
        data={
            "_csrf_token": token,
            "username": username,
            "password": password,
        },
    )

    assert response.status_code == 200
    assert b"Invalid username or password." in response.data


def test_login_preserves_password_whitespace_and_rotates_session(client):
    with app.app.app_context():
        admin = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        admin.set_password(" password with spaces ")
        identity = app.PlatformIdentity.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        identity.password_hash = admin.password_hash
        db.session.commit()
    try:
        client.get("/login")
        with client.session_transaction() as session:
            session["attacker_supplied_marker"] = "must-not-survive"
            token = session["_csrf_token"]
        response = client.post(
            "/login",
            data={
                "_csrf_token": token,
                "username": ADMIN_CREDENTIALS["username"],
                "password": " password with spaces ",
            },
        )
        assert response.status_code == 302
        finish_operational_login(client)
        with client.session_transaction() as session:
            assert "attacker_supplied_marker" not in session
            assert "_user_id" in session
    finally:
        with app.app.app_context():
            admin = Staff.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            admin.set_password(ADMIN_CREDENTIALS["password"])
            identity = app.PlatformIdentity.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            identity.password_hash = admin.password_hash
            db.session.commit()


def test_login_and_logout_require_valid_csrf_tokens(client):
    signed_in = login_as(
        client,
        ADMIN_CREDENTIALS["username"],
        ADMIN_CREDENTIALS["password"],
        follow_redirects=False,
    )
    assert signed_in.status_code == 302

    assert client.get("/logout").status_code == 405
    assert client.post("/logout").status_code == 400
    assert client.post(
        "/logout", data={"_csrf_token": "invalid"}
    ).status_code == 400

    token = csrf(client)
    signed_out = client.post(
        "/logout", data={"_csrf_token": token}, follow_redirects=False
    )
    assert signed_out.status_code == 302
    assert signed_out.headers["Location"].endswith("/login")
    assert client.get("/").status_code == 302


@pytest.mark.parametrize(
    "path",
    [
        "/login",
        "/recover",
        "/recover/approve/not-a-token",
        "/recover/reset/not-a-token",
        "/invite/not-a-token",
        "/login/mfa",
        "/login/platform-mfa",
        "/login/platform-mfa/setup",
    ],
)
def test_anonymous_browser_posts_default_deny_missing_csrf(client, path):
    response = client.post(path)
    assert response.status_code == 400


def test_user_can_update_own_profile_contact_details(client):
    login(client)
    with app.app.app_context():
        staff_id = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one().id
    response = client.post(
        f"/staff/{staff_id}",
        data={
            "_csrf_token": csrf(client),
            "email": "admin.profile@example.test",
            "phone_number": "0044 7700 900123",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Contact details updated" in response.data
    assert b'data-profile-section="overview"' in response.data
    assert b'data-profile-section="contact"' in response.data
    assert b'data-profile-section="security"' in response.data
    assert b'data-profile-section="mfa"' in response.data
    assert b'action="/password"' in response.data
    assert b"Multi-factor authentication is enabled." in response.data
    assert b"Select a profile function" in response.data
    with app.app.app_context():
        staff = db.session.get(Staff, staff_id)
        assert staff.email == "admin.profile@example.test"
        assert staff.phone_number == "+447700900123"
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
    assert b"Active unit" not in response.data
    assert b"data-operational-clock" in response.data
    assert b"Secure session" in response.data
    assert b'<body class="app-body roster-page">' in response.data

    stylesheet = client.get("/static/styles.css")
    assert stylesheet.status_code == 200
    assert b".roster .cell select.code-input.off" in stylesheet.data
    assert b"background: var(--off-blue)" in stylesheet.data
    assert b"select.code-input.code-len-5" in stylesheet.data
    assert b"-webkit-appearance:none" in stylesheet.data
    assert b".roster-page .page-content" in stylesheet.data
    assert b"padding:.65rem 0" in stylesheet.data
    assert b"table.roster { zoom:var(--ui-scale); }" in stylesheet.data
    assert b"transform: scale(var(--ui-scale))" not in stylesheet.data
    assert b"remaining below every sticky heading" in stylesheet.data
    assert b"z-index:12" in stylesheet.data
    assert response.data.count(b'data-roster-auto-submit="true"') > 1
    assert b"roster-shift-dialog" not in response.data
    assert response.data.count(b'class="annot-select" data-annotation-select') == 1
    assert b">Annotate</button>" not in response.data
    assert b'<span aria-hidden="true">&nbsp;</span></button>' in response.data
    assert b"Saving\xe2\x80\xa6" in response.data
    assert b"atcroster:scroll:" in response.data
    assert b"updateRosterStatusFrameWidth" in response.data
    assert b"--roster-status-frame-width" in response.data
    assert b".roster-status-frame::before" in stylesheet.data
    assert b"annotationButton.dataset.version = payload.version" in response.data


def test_roster_renders_annual_leave_as_static_al_code(client):
    login(client)
    client.get("/roster/2025-06")
    duty_day = date(2025, 6, 3)
    with app.app.app_context():
        assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
        original = (assignment.code, assignment.source, assignment.note)
        assignment.code = "AL"
        assignment.source = "leave"
        assignment.note = "annual leave"
        db.session.commit()

    try:
        response = client.get("/roster/2025-06")
        assert response.status_code == 200
        assert b'class="code-input code-display code-len-2 al group-a"' in response.data
        assert b">AL</span>" in response.data
    finally:
        with app.app.app_context():
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            assignment.code, assignment.source, assignment.note = original
            db.session.commit()


def test_annual_leave_requires_soal_before_roster_shift_override(client):
    login(client)
    ym = "2025-06"
    duty_day = date(2025, 6, 4)
    client.get(f"/roster/{ym}")
    with app.app.app_context():
        assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
        original = (
            assignment.code,
            assignment.source,
            assignment.note,
            assignment.annotation,
            assignment.annotation_note,
            assignment.version,
        )
        assignment.code = "AL"
        assignment.source = "leave"
        assignment.note = "annual leave"
        assignment.annotation = ""
        assignment.annotation_note = ""
        soal_definition = AnnotationType.query.filter_by(
            unit_id=1, code="SOAL"
        ).first()
        created_soal_definition = soal_definition is None
        if soal_definition is None:
            soal_definition = AnnotationType(
                unit_id=1,
                code="SOAL",
                label="SOAL",
                category="Overtime",
                tags="ot,soal",
                is_active=True,
            )
            db.session.add(soal_definition)
        leave = Leave(
            unit_id=1,
            staff_id=1,
            leave_type="AL",
            start=duty_day,
            end=duty_day,
        )
        db.session.add(leave)
        db.session.commit()
        app.refresh_annotation_cache()
        leave_id = leave.id
        soal_definition_id = soal_definition.id
        version = assignment.version

    try:
        blocked = client.post(
            f"/assign/1/{ym}/{duty_day.isoformat()}",
            data={
                "_csrf_token": csrf(client),
                "assignment_version": version,
                "code": "D",
            },
            follow_redirects=True,
        )
        assert blocked.status_code == 200
        assert b"annual-leave cell is locked" in blocked.data
        with app.app.app_context():
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            assert assignment.code == "AL"
            version = assignment.version

        applied = client.post(
            f"/assign/1/{ym}/{duty_day.isoformat()}",
            data={
                "_csrf_token": csrf(client),
                "assignment_version": version,
                "annotation": "SOAL",
            },
            follow_redirects=True,
        )
        assert applied.status_code == 200
        with app.app.app_context():
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            assert assignment.annotation == "SOAL"
            version = assignment.version
        assert b"code-input code-len-2 al group-a" in applied.data
        assert b"SOAL" in applied.data

        shifted = client.post(
            f"/assign/1/{ym}/{duty_day.isoformat()}",
            data={
                "_csrf_token": csrf(client),
                "assignment_version": version,
                "code": "D",
            },
            follow_redirects=True,
        )
        assert shifted.status_code == 200
        with app.app.app_context():
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            assert assignment.code == "D"
            assert assignment.annotation == "SOAL"
            assert db.session.get(Leave, leave_id) is not None
            version = assignment.version

        removed = client.post(
            f"/assign/1/{ym}/{duty_day.isoformat()}",
            data={
                "_csrf_token": csrf(client),
                "assignment_version": version,
                "annotation": "__remove__",
            },
            follow_redirects=True,
        )
        assert removed.status_code == 200
        assert b'class="code-input code-display code-len-2 al group-a"' in removed.data
        with app.app.app_context():
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            assert assignment.code == "AL"
            assert not assignment.annotation
            assert db.session.get(Leave, leave_id) is not None
    finally:
        with app.app.app_context():
            leave = db.session.get(Leave, leave_id)
            if leave:
                db.session.delete(leave)
            if created_soal_definition:
                AnnotationAudit.query.filter_by(
                    annotation_type_id=soal_definition_id
                ).delete(synchronize_session=False)
                definition = db.session.get(AnnotationType, soal_definition_id)
                if definition:
                    db.session.delete(definition)
            assignment = Assignment.query.filter_by(staff_id=1, day=duty_day).one()
            (
                assignment.code,
                assignment.source,
                assignment.note,
                assignment.annotation,
                assignment.annotation_note,
                assignment.version,
            ) = original
            db.session.commit()
            app.refresh_annotation_cache()


def test_favicon_is_served(client):
    resp = client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"


def test_retired_compliance_page_redirects_to_roster(client):
    login(client)
    page = client.get("/compliance-centre?ym=2025-04")
    assert page.status_code == 302
    assert page.headers["Location"].endswith("/roster/2025-04")
    export = client.get("/compliance-centre/export?ym=2025-04")
    assert export.status_code == 302
    assert export.headers["Location"].endswith("/roster/2025-04")
    roster = client.get("/roster/2025-04")
    assert b"fatigue" in roster.data.lower()
    assert b'href="/compliance-centre"' not in roster.data


def test_roster_publication_is_managed_from_monthly_roster(client):
    login(client)
    draft = client.get("/roster/2025-04")
    assert b"Draft roster" in draft.data
    assert b"Publish roster" in draft.data
    assert b'class="daily-total"><strong>Total ' in draft.data
    assert b'class="rag-count' in draft.data
    stylesheet = client.get("/static/styles.css")
    assert b".rag-count--over" in stylesheet.data
    assert b"@keyframes roster-count-over-pulse" in stylesheet.data
    token = csrf(client)
    published = client.post(
        "/roster/2025-04/publish",
        data={"_csrf_token": token},
        follow_redirects=True,
    )
    assert published.status_code == 200
    assert b"Published roster" in published.data
    assert b"Published " in published.data
    assert b"Draft roster" not in published.data
    with app.app.app_context():
        publication = app.RosterPublication.query.filter_by(
            year=2025, month=4, state="published"
        ).first()
        assert publication is not None
        snapshot = app.json.loads(publication.snapshot_json)
        assert snapshot["published_by"]["name"] == "Admin Test"

    returned_to_draft = client.post(
        "/roster/2025-04/unpublish",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert returned_to_draft.status_code == 200
    assert b"Draft roster" in returned_to_draft.data
    assert b"returned to Draft" in returned_to_draft.data
    with app.app.app_context():
        publication = app.RosterPublication.query.filter_by(
            year=2025, month=4
        ).order_by(app.RosterPublication.version.desc()).first()
        assert publication.state == "withdrawn"

    client.post("/logout", data={"_csrf_token": csrf(client)})
    client.get("/login")
    with client.session_transaction() as sess:
        login_token = sess["_csrf_token"]
    client.post("/login", data={
        "_csrf_token": login_token,
        "username": "staff_test", "password": "password123",
    })
    denied = client.post(
        "/roster/2025-05/publish",
        data={"_csrf_token": csrf(client)},
    )
    assert denied.status_code == 403
    denied_undo = client.post(
        "/roster/2025-04/unpublish",
        data={"_csrf_token": csrf(client)},
    )
    assert denied_undo.status_code == 403

    for username, ym in (
        ("watch_manager_test", "2025-05"),
        ("duty_watch_manager_test", "2025-06"),
    ):
        client.post("/logout", data={"_csrf_token": csrf(client)})
        client.get("/login")
        with client.session_transaction() as sess:
            login_token = sess["_csrf_token"]
        client.post("/login", data={
            "_csrf_token": login_token,
            "username": username, "password": "password123",
        })
        response = client.post(
            f"/roster/{ym}/publish",
            data={"_csrf_token": csrf(client)},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Published roster" in response.data


def test_stale_roster_cell_version_is_rejected(client):
    login(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).one()
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=admin.id, day=date(2025, 4, 2)
        ).one()
        stale_version = assignment.version
        staff_id = admin.id
    first = client.post(
        f"/assign/{staff_id}/2025-04/2025-04-02",
        data={
            "_csrf_token": csrf(client),
            "code": "A",
            "assignment_version": stale_version,
        },
    )
    assert first.status_code == 302
    stale = client.post(
        f"/assign/{staff_id}/2025-04/2025-04-02",
        data={
            "_csrf_token": csrf(client),
            "code": "N",
            "assignment_version": stale_version,
        },
    )
    assert stale.status_code == 409
    with app.app.app_context():
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=staff_id, day=date(2025, 4, 2)
        ).one()
        assert assignment.code == "A"
        assert assignment.version == stale_version + 1


def test_roster_shift_can_be_saved_without_a_page_reload(client):
    login(client)
    duty_day = date(2025, 4, 3)
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).one()
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=admin.id, day=duty_day
        ).one()
        original = (assignment.code, assignment.source, assignment.version)
        staff_id = admin.id
        version = assignment.version
    try:
        response = client.post(
            f"/assign/{staff_id}/2025-04/{duty_day.isoformat()}",
            data={
                "_csrf_token": csrf(client),
                "code": "D",
                "assignment_version": version,
            },
            headers={
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["code"] == "D"
        assert payload["version"] == version + 1
        assert payload["is_training"] is False
        assert payload["day"] == duty_day.isoformat()
        assert set(payload["day_summary"]) == {
            "counts", "night_active", "rag", "required", "total",
        }
    finally:
        with app.app.app_context():
            assignment = Assignment.query.filter_by(
                unit_id=1, staff_id=staff_id, day=duty_day
            ).one()
            assignment.code, assignment.source, assignment.version = original
            db.session.commit()


def test_roster_publication_emails_every_registered_unit_user(
    client, monkeypatch,
):
    login(client)
    with app.app.app_context():
        Staff.query.filter_by(unit_id=1).update({"email": ""})
        admin = Staff.query.filter_by(
            unit_id=1, username=ADMIN_CREDENTIALS["username"]
        ).one()
        staff = Staff.query.filter_by(
            unit_id=1, username="staff_test"
        ).one()
        admin.email = "publishing.admin@example.test"
        staff.email = "registered.user@example.test"
        db.session.commit()

    delivered = []

    def capture_email(address, subject, body):
        delivered.append((address, subject, body))
        return True

    monkeypatch.setattr(app, "_send_account_email", capture_email)
    response = client.post(
        "/roster/2025-07/publish",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Email sent to 2 registered users." in response.data
    assert {row[0] for row in delivered} == {
        "publishing.admin@example.test",
        "registered.user@example.test",
    }
    assert all("July 2025 roster published" in row[1] for row in delivered)
    assert all("/roster/2025-07" in row[2] for row in delivered)


def test_security_headers_are_present(client):
    response = client.get("/login")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Content-Security-Policy"].startswith(
        "default-src 'self'"
    )
    policy = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'nonce-" in policy
    assert "script-src 'self' 'unsafe-inline'" not in policy
    assert "'unsafe-inline'" not in policy
    assert "style-src-attr 'none'" in policy
    assert "connect-src 'self'" in policy


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
        "overtime": "/overtime",
        "leave": "/leave",
        "reports": "/reports",
        "metrics": "/metrics",
        "qualification": "/compliance",
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
            "roster": 200, "requests": 200,
            "overtime": 200, "leave": 200,
            "reports": 302, "metrics": 200, "qualification": 200,
            "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "watch_manager": {
            "roster": 200, "requests": 200,
            "overtime": 200, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "duty_watch_manager": {
            "roster": 200, "requests": 200,
            "overtime": 200, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "operations": 403, "coverage": 200,
            "scenarios": 200, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
        "staff": {
            "roster": 200, "requests": 200,
            "overtime": 403, "leave": 403,
            "reports": 403, "metrics": 403, "qualification": 403,
            "operations": 403, "coverage": 403,
            "scenarios": 403, "accounts": 403, "onboarding": 403,
            "admin": 403, "reference": 403, "platform": 403,
        },
    }

    clients = {}
    for role, username in credentials.items():
        role_client = app.app.test_client()
        response = login_as(role_client, username, follow_redirects=False)
        assert response.status_code == 302
        if role == "superadmin":
            protected = role_client.get("/platform/admin")
            assert protected.status_code == 302
            assert "/login" in protected.headers["Location"]
            setup = role_client.get("/login/platform-mfa/setup")
            assert setup.status_code == 200
            with role_client.session_transaction() as session:
                secret = session["_pending_platform_mfa_secret"]
                token = session["_csrf_token"]
            role_client.post(
                "/login/platform-mfa/setup",
                data={
                    "_csrf_token": token,
                    "code": pyotp.TOTP(secret).now(),
                },
            )
            role_client.get("/login/platform-mfa")
            with role_client.session_transaction() as session:
                token = session["_csrf_token"]
            verified = role_client.post(
                "/login/platform-mfa",
                data={
                    "_csrf_token": token,
                    "code": pyotp.TOTP(secret).now(),
                },
            )
            assert verified.status_code == 302
        if role in ("admin", "editor"):
            acknowledge_reports(role_client)
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


def test_login_next_uses_canonical_allowlisted_internal_route(client):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    response = client.post(
        "/login?next=/requests%3Fview%3Dpending",
        data={"_csrf_token": token, **ADMIN_CREDENTIALS},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login/mfa")
    with client.session_transaction() as session:
        assert session["_mfa_next"] == "/requests"


@pytest.mark.parametrize(
    "target",
    [
        "https://attacker.example/collect",
        "//attacker.example/collect",
        "/unknown-path",
        "/staff/999999",
    ],
)
def test_login_next_rejects_external_or_unapproved_destinations(
    client, target,
):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    response = client.post(
        "/login",
        query_string={"next": target},
        data={"_csrf_token": token, **ADMIN_CREDENTIALS},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login/mfa")
    with client.session_transaction() as session:
        assert session["_mfa_next"] == "/modules"


def test_untrusted_host_returns_plain_400_instead_of_error_handler_500(client):
    original = app.app.config.get("TRUSTED_HOSTS")
    app.app.config["TRUSTED_HOSTS"] = ["expected.example"]
    try:
        response = client.get("/health/ready", headers={"Host": "unknown.example"})
    finally:
        app.app.config["TRUSTED_HOSTS"] = original
    assert response.status_code == 400
    assert response.content_type.startswith("text/plain")
    assert b"untrusted host" in response.data


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
    assert response.data.index(b"Eligibility result") < response.data.index(
        b"What if?"
    )


@pytest.mark.parametrize("rostered_code", ["OFF", "AL"])
def test_overtime_finder_offers_operational_staff_on_off_or_leave_days(
    client, rostered_code,
):
    login(client)
    chosen_day = date(2027, 1, 15)
    with app.app.app_context():
        watch = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        person = Staff.query.filter_by(
            unit_id=1, username="ian_overtime_test"
        ).first()
        if person is None:
            person = Staff(
                unit_id=1,
                username="ian_overtime_test",
                name="Ian Overtime Test",
                staff_no="IAN-OT",
                role="user",
                watch_id=watch.id,
                membership_status="no_login",
                is_operational=True,
                exclude_from_ot=False,
                tower_ue_expiry=date(2028, 1, 1),
                pattern_csv="OFF",
                pattern_override=True,
            )
            person.set_password("password123")
            db.session.add(person)
            db.session.flush()
        person.exclude_from_ot = False
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=person.id, day=chosen_day
        ).first()
        if assignment is None:
            assignment = Assignment(
                unit_id=1,
                staff_id=person.id,
                day=chosen_day,
                source="manual",
            )
            db.session.add(assignment)
        assignment.code = rostered_code
        db.session.commit()
        person_id = person.id

    token = csrf(client)
    response = client.post(
        "/overtime",
        data={
            "_csrf_token": token,
            "action": "find",
            "date": chosen_day.isoformat(),
            "shift_code": "M",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert f'data-eligible-staff="{person_id}"'.encode() in response.data
    assert b"<td>Ian Overtime Test</td>" in response.data
    if rostered_code == "AL":
        assert b"On AL that day" in response.data

    what_if = client.post(
        "/overtime",
        data={
            "_csrf_token": token,
            "action": "what_if",
            "what_if_staff_id": str(person_id),
            "date": chosen_day.isoformat(),
            "shift_code": "M",
        },
        follow_redirects=True,
    )
    assert what_if.status_code == 200
    assert b"Ian Overtime Test is eligible" in what_if.data
    assert b"No exclusion rules were triggered" in what_if.data or b"Advisory information" in what_if.data

    with app.app.app_context():
        person = db.session.get(Staff, person_id)
        person.exclude_from_ot = True
        db.session.commit()

    ineligible = client.post(
        "/overtime",
        data={
            "_csrf_token": token,
            "action": "what_if",
            "what_if_staff_id": str(person_id),
            "date": chosen_day.isoformat(),
            "shift_code": "M",
        },
        follow_redirects=True,
    )
    assert ineligible.status_code == 200
    assert b"Ian Overtime Test is not eligible" in ineligible.data
    assert b"Opted out of overtime" in ineligible.data


def test_production_operations_workflows(client):
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

    with app.app.app_context():
        assert app.PositionEndorsement.query.count() == 1
        assert app.PositionRequirement.query.filter_by(
            position_id=position_id
        ).count() == 1
        assert app.BreakPlan.query.filter_by(position_id=position_id).count() == 1


def test_standalone_fatigue_reporting_workflow_is_removed(client):
    login(client)
    roster = client.get("/roster/2025-04")
    assert b'href="/fatigue/report"' not in roster.data
    assert client.get("/fatigue/report").status_code == 404
    assert client.post(
        "/fatigue/report",
        data={"_csrf_token": csrf(client)},
    ).status_code == 404


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
    assert b'data-roster-sticky-shield' in roster_resp.data
    assert b'rosterStickyShield.style.height' in roster_resp.data


def test_position_monitor_account_is_hidden_from_roster_and_export(client):
    with app.app.app_context():
        kiosk = Staff(
            unit_id=1,
            username="roster_hidden_kiosk",
            name="Hidden Position Monitor",
            staff_no="KIOSK-HIDDEN",
            role="position_monitor",
            is_operational=True,
        )
        kiosk.set_password("not-used")
        db.session.add(kiosk)
        db.session.flush()
        db.session.add_all(
            [
                Assignment(
                    unit_id=1,
                    staff_id=kiosk.id,
                    day=date(2027, 1, 5),
                    code="AL",
                    annotation="AAVA",
                    source="manual",
                ),
                Assignment(
                    unit_id=1,
                    staff_id=kiosk.id,
                    day=date.today(),
                    code="SC",
                    source="manual",
                ),
            ]
        )
        db.session.commit()
        kiosk_id = kiosk.id

    login(client)
    roster = client.get("/roster/2027-01")
    export = client.get("/roster/2027-01/export")

    assert roster.status_code == 200
    assert export.status_code == 200
    assert b"Hidden Position Monitor" not in roster.data
    assert b"KIOSK-HIDDEN" not in roster.data
    assert "Hidden Position Monitor" not in export.data.decode()
    assert "KIOSK-HIDDEN" not in export.data.decode()

    acknowledge_reports(client)
    report_responses = [
        client.get("/metrics?start=2027-01-01&end=2027-01-31"),
        client.get("/metrics/export?start=2027-01-01&end=2027-01-31"),
        client.get("/reports/leave/2027-01"),
        client.get("/reports/leave.csv?ym=2027-01"),
        client.get("/reports/leave-year"),
        client.get("/reports/sickness"),
    ]
    for response in report_responses:
        assert response.status_code == 200
        assert b"Hidden Position Monitor" not in response.data
        assert b"KIOSK-HIDDEN" not in response.data

    with app.app.app_context():
        Assignment.query.filter_by(staff_id=kiosk_id).delete()
        Staff.query.filter_by(id=kiosk_id).delete()
        db.session.commit()


def test_csv_exports_neutralise_spreadsheet_formula_payloads(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        person = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        person.name = '=HYPERLINK("https://attacker.invalid")'
        person.staff_no = "+SUM(1,1)"
        db.session.commit()
    try:
        roster_csv = client.get("/roster/2025-04/export").data.decode()
        metrics_csv = client.get("/metrics/export").data.decode()
        leave_csv = client.get("/reports/leave.csv?ym=2025-04").data.decode()
        for exported in (roster_csv, metrics_csv, leave_csv):
            assert "'=HYPERLINK" in exported
            assert "'+SUM(1,1)" in exported
    finally:
        with app.app.app_context():
            person = Staff.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            person.name = "Admin Test"
            person.staff_no = "ADM-001"
            db.session.commit()


def test_admin_pages_accessible(client):
    login(client)
    acknowledge_reports(client)
    endpoints = [
        "/admin",
        "/admin/reference",
        "/leave",
        "/metrics",
        "/reports",
        "/requests",
        "/admin/toil/new",
        "/admin/fatigue-rules",
        "/metrics/export",
    ]
    for url in endpoints:
        resp = client.get(url)
        assert resp.status_code == 200, f"Endpoint {url} returned {resp.status_code}"
        if url == "/admin/reference":
            assert b"annotation-edit-list" in resp.data
            assert b"<table" not in resp.data
            assert b"Leave codes" not in resp.data
            assert b"Working shift codes" in resp.data
            assert b'name="values" value="M"' in resp.data
            assert b'placeholder="Comma or space separated codes"' not in resp.data
            assert b"System tags" not in resp.data
            assert b"Allowed suffixes" not in resp.data
            assert b"Sort order" not in resp.data
            assert b"Allow a suffix" not in resp.data


def test_roster_code_lists_only_accept_existing_shift_codes(client):
    login(client)
    token = csrf(client)
    rejected = client.post(
        "/admin/reference",
        data={
            "_csrf_token": token,
            "form": "settings_codes",
            "key": "working_codes",
            "values": ["M", "DOESNOTEXIST"],
        },
        follow_redirects=True,
    )
    assert b"do not exist: DOESNOTEXIST" in rejected.data
    with app.app.app_context():
        assert app.RosterSetting.query.filter_by(
            unit_id=1, key="working_codes"
        ).first() is None

    token = csrf(client)
    saved = client.post(
        "/admin/reference",
        data={
            "_csrf_token": token,
            "form": "settings_codes",
            "key": "working_codes",
            "values": ["M", "D"],
        },
        follow_redirects=True,
    )
    assert b"Reference list updated." in saved.data
    with app.app.app_context():
        assert app.get_working_codes() == {"M", "D"}


def test_shift_staffing_mapping_follows_shift_type_tool(client):
    login(client)
    response = client.get("/admin")
    assert response.status_code == 200
    assert b"Required for accurate daily totals and coverage warnings" not in response.data
    assert response.data.index(b"admin-shift-list") < response.data.index(
        b"Which shifts count toward staffing?"
    )


def test_operations_workspace_is_hidden_from_primary_navigation(client):
    login(client)
    roster = client.get("/roster/2025-04")
    assert roster.status_code == 200
    assert b'href="/operations/' not in roster.data
    # The underlying workspace is retained for a later re-enable.
    assert client.get("/operations/2025-04").status_code == 200


def test_annotation_totals_follow_unit_definitions_not_fixed_columns(client):
    login(client)
    acknowledge_reports(client)
    with app.app.app_context():
        person = Staff.query.filter_by(username="staff_test").one()
        db.session.add_all([
            AnnotationType(
                unit_id=1, code="CUSTOM", label="Custom Cover",
                category="Operations", is_active=True, sort_order=1,
            ),
            AnnotationType(
                unit_id=1, code="OLD", label="Retired Marker",
                category="Operations", is_active=False, sort_order=2,
                has_been_used=True,
            ),
            Assignment(
                unit_id=1, staff_id=person.id, day=date(2026, 1, 10),
                code="M", annotation="CUSTOM", source="manual",
            ),
            Assignment(
                unit_id=1, staff_id=person.id, day=date(2026, 1, 11),
                code="M", annotation="OLD", source="manual",
            ),
            Assignment(
                unit_id=1, staff_id=person.id, day=date(2026, 1, 12),
                code="M", annotation="INFO",
                annotation_note="Operational context only",
                source="manual",
            ),
        ])
        db.session.commit()
        app.refresh_annotation_cache()

    page = client.get(
        "/metrics?start=2026-01-10&end=2026-01-12"
    )
    assert page.status_code == 200
    assert b"Annotation Totals" in page.data
    assert b"Custom Cover" in page.data
    assert b"Retired Marker" in page.data
    assert b"historical" in page.data
    assert b"Ext Total" not in page.data
    assert b"AAVA Total" not in page.data
    assert b">Information</th>" not in page.data
    assert b"Operational context only" not in page.data

    exported = client.get(
        "/metrics/export?start=2026-01-10&end=2026-01-12"
    )
    assert exported.status_code == 200
    csv_text = exported.data.decode()
    assert "Custom Cover (CUSTOM)" in csv_text
    assert "Retired Marker (OLD)" in csv_text
    assert "Information (INFO)" not in csv_text
    assert "Operational context only" not in csv_text
    assert "Ext Total" not in csv_text


def test_admin_can_add_and_manage_custom_fatigue_rules(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(username="staff_test").one()
        target_day = date(2026, 2, 1)
        db.session.add(Assignment(
            unit_id=1, staff_id=person.id, day=target_day,
            code="M", source="manual",
        ))
        db.session.commit()
        person_id = person.id

    added = client.post(
        "/admin/fatigue-rules",
        data={
            "_csrf_token": csrf(client),
            "action": "add_custom",
            "name": "Local seven-hour duty limit",
            "rule_type": "max_duty_hours",
            "threshold": "7",
            "window_days": "1",
            "severity": "critical",
            "enabled": "on",
        },
        follow_redirects=True,
    )
    assert added.status_code == 200
    assert b"fatigue rule saved" in added.data
    assert b"Local seven-hour duty limit" in added.data

    with app.app.app_context():
        person = db.session.get(Staff, person_id)
        findings = app.fatigue_flags_for_range(
            person, [target_day]
        )
        custom_messages = [
            message for message in findings[target_day]
            if "Local seven-hour duty limit" in message
        ]
        assert custom_messages
        config = app._fatigue_rule_config(1)
        custom_code = config["custom"][0]["code"]

    paused = client.post(
        "/admin/fatigue-rules",
        data={
            "_csrf_token": csrf(client),
            "action": "update_custom",
            "code": custom_code,
            "name": "Local seven-hour duty limit",
            "rule_type": "max_duty_hours",
            "threshold": "7",
            "window_days": "1",
            "severity": "critical",
        },
        follow_redirects=True,
    )
    assert paused.status_code == 200
    with app.app.app_context():
        person = db.session.get(Staff, person_id)
        findings = app.fatigue_flags_for_range(
            person, [target_day]
        )
        assert not any(
            custom_code in message
            for message in findings.get(target_day, [])
        )


def test_d24_requires_a_complete_observation_window(client):
    start = datetime(2026, 1, 1, 8)
    segments = []
    for offset in range(31):
        duty_start = start + timedelta(days=offset)
        segments.append({
            "day": duty_start.date(),
            "start": duty_start,
            "end": duty_start + timedelta(hours=8),
            "mins": 8 * 60,
            "night": False,
            "early": False,
            "early_pre0600": False,
            "morning": True,
        })

    with app.app.app_context():
        without_history = app._analyze_segments(segments)
        with_history = app._analyze_segments(
            segments,
            observation_start=start - timedelta(days=30),
        )
    assert app._analyze_segments is fatigue_engine._analyze_segments
    assert app._compliance_month is fatigue_compliance.compliance_month
    assert (
        app._fatigue_rule_config
        == app._fatigue_rule_config_service.load
    )
    assert not any(
        message.startswith("D24:")
        for messages in without_history.values()
        for message in messages
    )
    assert any(
        message.startswith("D24:")
        for messages in with_history.values()
        for message in messages
    )


def test_system_fatigue_threshold_changes_are_airport_specific(client):
    login(client)
    updated = client.post(
        "/admin/fatigue-rules",
        data={
            "_csrf_token": csrf(client),
            "action": "update_system",
            "code": "D21",
            "name": "Local duty duration",
            "severity": "warning",
            "enabled": "on",
            "parameter_max_duty_hours": "7",
        },
        follow_redirects=True,
    )
    assert updated.status_code == 200
    assert b"D21 fatigue rule updated" in updated.data
    assert b"Test Airport" in updated.data
    with app.app.app_context():
        unit_one = app._fatigue_rule_config(1)
        other_airport = app._fatigue_rule_config(3)
        assert (
            unit_one["system"]["D21"]["parameters"]
            ["max_duty_hours"]["value"]
        ) == 7
        assert (
            other_airport["system"]["D21"]["parameters"]
            ["max_duty_hours"]["value"]
        ) == 10
        assert unit_one["system"]["D21"]["name"] == "Local duty duration"
        assert other_airport["system"]["D21"]["name"] != "Local duty duration"

    # Restore the test airport default for subsequent tests.
    restored = client.post(
        "/admin/fatigue-rules",
        data={
            "_csrf_token": csrf(client),
            "action": "update_system",
            "code": "D21",
            "name": "Duty duration and rolling hours",
            "severity": "critical",
            "enabled": "on",
            "parameter_max_duty_hours": "10",
        },
    )
    assert restored.status_code == 302


def test_admin_can_configure_requestable_shift(client):
    login(client)
    with app.app.app_context():
        shift = ShiftType.query.filter_by(code="M").one()
        shift_id = shift.id

    response = client.post(
        "/admin",
        data={
            "form": "shift_edit",
            "_csrf_token": csrf(client),
            "shift_id": shift_id,
            "name": "Morning",
            "start": "07:00",
            "end": "15:00",
            "is_working": "on",
            "is_active": "on",
            "is_requestable": "on",
            "required_qualification": "MEDICAL",
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
        assert shift.required_qualification == "MEDICAL"


def test_unit_admin_edits_and_previews_qualification_import(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(
            username="staff_test", unit_id=1
        ).one()
        medical = QualificationType.query.filter_by(
            unit_id=1, code="MEDICAL"
        ).one()
        person_id = person.id
        medical_id = medical.id
    token = csrf(client)
    assigned = client.post("/compliance", data={
        "_csrf_token": token,
        "action": "save_person",
        "person_id": person_id,
        "type_id": medical_id,
        "issued_on": "2026-01-01",
        "valid_from": "2026-01-01",
        "expires_on": "2027-01-01",
        "status": "valid",
    }, follow_redirects=True)
    assert assigned.status_code == 200
    assert b"Person qualification saved" in assigned.data
    created = client.post("/compliance", data={
        "_csrf_token": csrf(client),
        "action": "create_type",
        "code": "UCA",
        "label": "Unit Competence Assessor",
        "warning_days_csv": "180,90,60,30",
        "expiry_required": "no",
        "is_active": "yes",
    }, follow_redirects=True)
    assert created.status_code == 200
    assert b"Qualification type saved" in created.data
    preview = client.post(
        "/compliance",
        data={
            "_csrf_token": csrf(client),
            "action": "import_preview",
            "csv_file": (
                io.BytesIO(
                    b"staff_no,type_code,status,issued_on,valid_from,expires_on\n"
                    b"USR-001,UCA,valid,2026-01-01,2026-01-01,\n"
                ),
                "qualifications.csv",
            ),
        },
        content_type="multipart/form-data",
    )
    assert preview.status_code == 200
    assert b"1 validated records" in preview.data


def test_onboarding_branding_rules_and_csv_preview(client):
    login(client)
    token = csrf(client)
    identity = client.post(
        "/unit/onboarding",
        data={
            "_csrf_token": token,
            "action": "identity",
            "name": "Test Airport",
            "code": "TST",
            "timezone": "Europe/London",
            "locale": "en-GB",
            "date_format": "%d/%m/%Y",
            "display_name": "TST Roster Control",
            "primary_colour": "#123456",
            "accent_colour": "#abcdef",
        },
        follow_redirects=True,
    )
    assert identity.status_code == 200
    assert b"TST Roster Control" in identity.data
    rules = client.post(
        "/unit/onboarding",
        data={
            "_csrf_token": token,
            "action": "request_rules",
            "request_months_ahead": "6",
            "request_lock_day": "18",
        },
        follow_redirects=True,
    )
    assert rules.status_code == 200
    preview = client.post(
        "/unit/onboarding",
        data={
            "_csrf_token": token,
            "action": "csv_preview",
            "csv_file": (
                io.BytesIO(
                    b"name,staff_no,watch\nImported Person,IMP-001,Watch A\n"
                ),
                "people.csv",
            ),
        },
        content_type="multipart/form-data",
    )
    assert preview.status_code == 200
    assert b"Imported Person" in preview.data
    with client.session_transaction() as session:
        nonce = session["_onboarding_csv_preview"]["nonce"]
    applied = client.post(
        "/unit/onboarding",
        data={
            "_csrf_token": token,
            "action": "csv_apply",
            "nonce": nonce,
        },
        follow_redirects=True,
    )
    assert b"Validated staff records imported" in applied.data
    with app.app.app_context():
        imported = Staff.query.filter_by(staff_no="IMP-001").one()
        assert imported.membership_status == "no_login"
        unit = app.db.session.get(app.Unit, 1)
        assert unit.request_months_ahead == 6
        assert unit.request_lock_day == 18


def test_unit_admin_is_guided_until_onboarding_is_completed(client):
    login(client)
    with app.app.app_context():
        unit = db.session.get(Unit, 1)
        unit.onboarding_step = 0
        db.session.commit()

    home = client.get("/", follow_redirects=False)
    assert home.status_code == 302
    assert "/unit/onboarding" in home.headers["Location"]
    guided = client.get("/unit/onboarding")
    assert b"Guided first-time setup" in guided.data

    completed = client.post(
        "/unit/onboarding",
        data={
            "_csrf_token": csrf(client),
            "action": "complete_setup",
            "confirm_complete": "yes",
        },
        follow_redirects=False,
    )
    assert completed.status_code == 302
    assert completed.headers["Location"].endswith("/")
    dashboard = client.get("/", follow_redirects=False)
    assert "/roster/" in dashboard.headers["Location"]
    with app.app.app_context():
        assert db.session.get(Unit, 1).onboarding_step == 100


def test_admin_can_map_created_shifts_to_roster_counts(client):
    login(client)
    with app.app.app_context():
        shifts = ShiftType.query.filter_by(unit_id=1).all()
        morning = next(shift for shift in shifts if shift.code == "M")
        form = {
            "_csrf_token": csrf(client),
            "form": "counter_mapping",
        }
        for shift in shifts:
            form[f"counter_group_{shift.id}"] = ""
        form[f"counter_group_{morning.id}"] = "D"

    saved = client.post("/admin", data=form, follow_redirects=True)
    assert saved.status_code == 200
    assert b"Shift counter mapping saved" in saved.data
    assert b"Which shifts count toward staffing?" in saved.data
    with app.app.app_context():
        app.refresh_roster_settings_cache()
        assert app.shift_counter_group("M", 1) == "D"
        assert app.shift_counter_group("OFF", 1) == ""

        row = app.RosterSetting.query.filter_by(
            unit_id=1, key="shift_counter_map"
        ).one()
        mapping = json.loads(row.value)
        mapping["M"] = "M"
        row.value = json.dumps(mapping)
        db.session.commit()
        app.refresh_roster_settings_cache()


def test_staffing_requirements_rows_offer_copy_below(client):
    login(client)
    page = client.get("/admin")
    assert page.status_code == 200
    assert page.data.count(b"<tr data-requirement-row>") == 24
    assert page.data.count(b'aria-label="Copy ') == 24
    assert b"Copy below" in page.data
    assert b"data-requirements-copy-status" in page.data
    assert b"requirementRows.slice(rowIndex + 1)" in page.data


def test_weekend_and_special_date_requirements_are_available(client):
    login(client)
    page = client.get("/admin")
    assert page.status_code == 200
    assert b"Monday\xe2\x80\x93Friday" in page.data
    assert b"Saturday" in page.data
    assert b"Sunday" in page.data
    assert b"requirements-day-group--weekday" in page.data
    assert b"requirements-day-group--saturday" in page.data
    assert b"requirements-day-group--sunday" in page.data
    assert b"requirements-day-cell--start" in page.data
    assert page.data.count(b'name="req_sat_m"') == 24
    assert page.data.count(b'name="req_sun_n"') == 24
    assert b"Special date requirements" in page.data
    assert b'name="special_day"' in page.data

    requirement_values = {
        "req_m": "4", "req_d": "3", "req_a": "4", "req_n": "2",
        "req_sat_m": "2", "req_sat_d": "1",
        "req_sat_a": "2", "req_sat_n": "0",
        "req_sun_m": "1", "req_sun_d": "1",
        "req_sun_a": "1", "req_sun_n": "0",
    }
    saved_defaults = client.post(
        "/admin",
        data={
            "_csrf_token": csrf(client),
            "form": "req",
            "ym": "2026-12",
            **requirement_values,
        },
        follow_redirects=True,
    )
    assert saved_defaults.status_code == 200
    with app.app.app_context():
        monthly = app.Requirement.query.filter_by(
            unit_id=1, year=2026, month=12
        ).one()
        assert monthly.req_sat_m == 2
        assert monthly.req_sun_a == 1

    saved = client.post(
        "/admin",
        data={
            "_csrf_token": csrf(client),
            "form": "special_requirement",
            "special_day": "2026-12-25",
            "special_label": "Christmas Day",
            "special_req_m": "2",
            "special_req_d": "1",
            "special_req_a": "2",
            "special_req_n": "0",
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert b"Special requirements saved for 25 December 2026" in saved.data

    roster = client.get("/roster/2026-12")
    assert roster.status_code == 200
    assert b"Special staffing requirements" in roster.data
    assert b"Christmas Day" in roster.data
    assert b"Friday 25 December" in roster.data

    with app.app.app_context():
        special = app.SpecialRequirement.query.filter_by(
            unit_id=1, day=date(2026, 12, 25)
        ).one()
        assert app.requirements_for_day(
            None, special.day, special
        ) == {"M": 2, "D": 1, "A": 2, "N": 0}


def test_effective_requirements_use_weekend_defaults_and_date_override():
    monthly = app.Requirement(
        req_m=4, req_d=3, req_a=4, req_n=2,
        req_sat_m=2, req_sat_d=1, req_sat_a=2, req_sat_n=0,
        req_sun_m=1, req_sun_d=1, req_sun_a=1, req_sun_n=0,
    )
    assert app.requirements_for_day(
        monthly, date(2026, 12, 21)
    ) == {"M": 4, "D": 3, "A": 4, "N": 2}
    assert app.requirements_for_day(
        monthly, date(2026, 12, 26)
    ) == {"M": 2, "D": 1, "A": 2, "N": 0}
    assert app.requirements_for_day(
        monthly, date(2026, 12, 27)
    ) == {"M": 1, "D": 1, "A": 1, "N": 0}


def test_counter_requires_created_shift_and_respects_closed_nights(client):
    monday = date(2026, 7, 27)
    tuesday = date(2026, 7, 28)
    with app.app.app_context():
        night_setting = app.RosterSetting.query.filter_by(
            unit_id=1, key="night_active_weekdays"
        ).first()
        if not night_setting:
            night_setting = app.RosterSetting(
                unit_id=1, key="night_active_weekdays"
            )
            db.session.add(night_setting)
        night_setting.value = "0"
        mapping_setting = app.RosterSetting.query.filter_by(
            unit_id=1, key="shift_counter_map"
        ).first()
        if not mapping_setting:
            mapping_setting = app.RosterSetting(
                unit_id=1, key="shift_counter_map"
            )
            db.session.add(mapping_setting)
        mapping = json.loads(mapping_setting.value or "{}")
        mapping["N"] = "N"
        mapping_setting.value = json.dumps(mapping)
        db.session.commit()
        app.refresh_roster_settings_cache()

        assert app.shift_counter_group_for_day("N", monday, 1) == "N"
        assert app.shift_counter_group_for_day("N", tuesday, 1) == ""
        assert app.shift_counter_group("UNCREATED", 1) == ""


def test_roster_counter_requires_current_medical_and_unit_endorsement():
    roster_day = date(2026, 7, 31)
    person = Staff(
        medical_expiry=roster_day,
        tower_ue_expiry=roster_day,
    )
    assert app.staff_is_countable_on(person, roster_day)

    person.medical_expiry = roster_day - timedelta(days=1)
    assert not app.staff_is_countable_on(person, roster_day)

    person.medical_expiry = roster_day
    person.tower_ue_expiry = roster_day - timedelta(days=1)
    person.radar_ue_expiry = None
    person.met_ue_expiry = None
    assert not app.staff_is_countable_on(person, roster_day)

    person.met_ue_expiry = roster_day
    assert not app.staff_is_countable_on(person, roster_day)

    person.radar_ue_expiry = roster_day
    assert app.staff_is_countable_on(person, roster_day)


def test_under_training_flags_do_not_replace_an_in_date_endorsement():
    roster_day = date(2026, 7, 31)
    person = Staff(
        medical_expiry=roster_day,
        tower_ut=True,
        radar_ut=True,
        met_ut=True,
    )
    assert not app.staff_is_countable_on(person, roster_day)


def test_roster_never_shows_fatigue_warning_on_off_shift(
    client, monkeypatch
):
    warning_day = date(2025, 4, 1)
    with app.app.app_context():
        person = Staff.query.filter_by(unit_id=1).first()
        assert person is not None
        monkeypatch.setattr(
            app,
            "fatigue_flags_for_range",
            lambda *_args, **_kwargs: {
                warning_day: ["stale warning must not appear"]
            },
        )

        visible = app.roster_fatigue_flags_for_range(
            person, [warning_day], {warning_day: "OFF"}, 1
        )
        assert visible == {}

        visible_working = app.roster_fatigue_flags_for_range(
            person, [warning_day], {warning_day: "M"}, 1
        )
        assert visible_working == {
            warning_day: ["stale warning must not appear"]
        }


def test_manual_toil_form_submits_and_can_add_or_deduct(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(username="staff_test").one()
        person.toil_half_days = 2
        person_id = person.id
        db.session.commit()

    added = client.post(
        "/admin/toil/new",
        data={
            "_csrf_token": csrf(client),
            "staff_id": person_id,
            "direction": "add",
            "amount": "1",
            "unit": "days",
            "note": "Regression test",
        },
        follow_redirects=True,
    )
    assert added.status_code == 200
    assert b"1 days added to Staff Test" in added.data
    deducted = client.post(
        "/admin/toil/new",
        data={
            "_csrf_token": csrf(client),
            "staff_id": person_id,
            "direction": "subtract",
            "amount": "4",
            "unit": "hours",
            "note": "Regression test",
        },
        follow_redirects=True,
    )
    assert deducted.status_code == 200
    with app.app.app_context():
        assert db.session.get(Staff, person_id).toil_half_days == 3


def test_roster_scenario_uses_guided_fields_without_json(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(username="staff_test").one()
        person_id = person.id
    saved = client.post(
        "/planning/scenarios",
        data={
            "_csrf_token": csrf(client),
            "name": "Guided cover check",
            "staff_id": person_id,
            "day": "2026-07-27",
            "code": "M",
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert b"live roster" in saved.data
    with app.app.app_context():
        scenario = app.Scenario.query.filter_by(
            name="Guided cover check"
        ).one()
        changes = json.loads(scenario.changes_json)
        assert changes[0]["staff_id"] == str(person_id)
        assert changes[0]["code"] == "M"


def test_non_working_shift_cannot_be_requestable(client):
    login(client)
    with app.app.app_context():
        shift = ShiftType.query.filter_by(code="OFF").one()
        shift_id = shift.id

    response = client.post(
        "/admin",
        data={
            "form": "shift_edit",
            "_csrf_token": csrf(client),
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
        data={
            "_csrf_token": csrf(client),
            "watch_id": watch_b.id, "effective_date": "2025-06-01",
        },
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
        data={
            "_csrf_token": csrf(client),
            "watch_id": watch_a.id, "effective_date": "2025-07-01",
        },
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
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert delete_resp.status_code == 200
    assert b"Watch move deleted" in delete_resp.data

    with app.app.app_context():
        assert db.session.get(StaffWatchHistory, entry.id) is None

    # Sanity check that the main admin dashboard still renders after the flow
    admin_resp = client.get("/admin")
    assert admin_resp.status_code == 200


def test_unit_watch_and_personal_pattern_inheritance(client):
    anchor = date(2026, 7, 27)  # Monday
    with app.app.app_context():
        watch_a = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        watch_b = Watch.query.filter_by(unit_id=1, name="Watch B").one()
        watch_a.pattern_csv = "N,N"
        watch_a.pattern_anchor = anchor
        watch_b.pattern_csv = "A,A"
        watch_b.pattern_anchor = anchor
        person = Staff(
            unit_id=1, username="pattern_test", name="Pattern Test",
            staff_no="PAT-001", role="user", watch=watch_a,
            pattern_override=False,
        )
        person.set_password("password123")
        db.session.add(person)
        night_setting = app.RosterSetting.query.filter_by(
            unit_id=1, key="night_active_weekdays"
        ).first()
        if not night_setting:
            night_setting = app.RosterSetting(
                unit_id=1, key="night_active_weekdays"
            )
            db.session.add(night_setting)
        night_setting.value = "0"
        db.session.commit()
        app.refresh_roster_settings_cache()

        assert app.code_from_pattern(person, anchor) == "N"
        assert app.code_from_pattern(
            person, date(2026, 7, 28)
        ) == "OFF"

        db.session.add(StaffWatchHistory(
            unit_id=1, staff_id=person.id, watch_id=watch_b.id,
            effective_date=date(2026, 7, 28),
        ))
        db.session.commit()
        assert app.code_from_pattern(
            person, date(2026, 7, 28)
        ) == "A"

        watch_a.pattern_csv = ""
        watch_a.pattern_anchor = anchor
        for key, value in (
            ("base_pattern_csv", "M,A,OFF"),
            ("base_pattern_anchor", "2026-07-26"),
        ):
            setting = app.RosterSetting.query.filter_by(
                unit_id=1, key=key
            ).first()
            if not setting:
                setting = app.RosterSetting(unit_id=1, key=key)
                db.session.add(setting)
            setting.value = value
        db.session.commit()
        app.refresh_roster_settings_cache()
        assert app.code_from_pattern(person, anchor) == "M"
        assert app.code_from_pattern(
            person, date(2026, 7, 28)
        ) == "A"

        person.pattern_override = True
        person.pattern_csv = "D,OFF"
        person.pattern_anchor = anchor
        db.session.commit()
        assert app.code_from_pattern(person, anchor) == "D"

    login(client)
    admin_page = client.get("/admin")
    assert admin_page.status_code == 200
    assert b"Roster cycle and watches" in admin_page.data
    assert b"Nights when the airport is open" in admin_page.data


def test_mfa_challenge_completes_login(client):
    client.post("/logout", data={"_csrf_token": csrf(client)})
    secret = pyotp.random_base32()
    with app.app.app_context():
        admin = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).first()
        app.MfaCredential.query.filter_by(person_id=admin.id).delete()
        db.session.add(app.MfaCredential(
            unit_id=admin.unit_id,
            person_id=admin.id,
            encrypted_secret=app._encrypt_field(secret),
            enabled=True,
            enrolled_at=app.utcnow(),
        ))
        db.session.commit()
    password_step = client.post(
        "/login",
        data={"_csrf_token": csrf(client), **ADMIN_CREDENTIALS},
        follow_redirects=False,
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


def test_continuous_activity_cannot_extend_absolute_session(client):
    login(client)
    with client.session_transaction() as session:
        session["_last_seen_epoch"] = int(app.utcnow().timestamp())
        session["_session_started_at"] = (
            app.utcnow() - app.timedelta(minutes=721)
        ).isoformat()
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    with client.session_transaction() as session:
        assert "_user_id" not in session


def test_privilege_change_forces_existing_session_invalidation(client):
    login(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        admin.role = "user"
        db.session.commit()
    try:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]
        with client.session_transaction() as session:
            assert "_user_id" not in session
    finally:
        with app.app.app_context():
            admin = Staff.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            admin.role = "admin"
            db.session.commit()


def test_password_change_revokes_another_existing_session(client):
    other_client = app.app.test_client()
    login(client)
    copy_authenticated_session(client, other_client)
    response = client.post(
        "/password",
        data={
            "_csrf_token": csrf(client),
            "current_password": ADMIN_CREDENTIALS["password"],
            "new_password": "replacement-password-2026",
            "confirm_password": "replacement-password-2026",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    try:
        revoked = other_client.get("/", follow_redirects=False)
        assert revoked.status_code == 302
        assert "/login" in revoked.headers["Location"]
        with other_client.session_transaction() as browser_session:
            assert "_user_id" not in browser_session
    finally:
        with app.app.app_context():
            person = Staff.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            person.set_password(ADMIN_CREDENTIALS["password"])
            identity = app.PlatformIdentity.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            identity.password_hash = person.password_hash
            db.session.commit()


def test_airport_mfa_change_revokes_another_existing_session(client):
    other_client = app.app.test_client()
    login(client)
    copy_authenticated_session(client, other_client)
    with app.app.app_context():
        person = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        credential = app.MfaCredential.query.filter_by(
            person_id=person.id
        ).one()
        original_secret = credential.encrypted_secret
        credential.encrypted_secret = app._encrypt_field(pyotp.random_base32())
        credential.enrolled_at = app.utcnow()
        db.session.commit()
    try:
        revoked = other_client.get("/", follow_redirects=False)
        assert revoked.status_code == 302
        assert "/login" in revoked.headers["Location"]
        with other_client.session_transaction() as browser_session:
            assert "_user_id" not in browser_session
    finally:
        with app.app.app_context():
            person = Staff.query.filter_by(
                username=ADMIN_CREDENTIALS["username"]
            ).one()
            credential = app.MfaCredential.query.filter_by(
                person_id=person.id
            ).one()
            credential.encrypted_secret = original_secret
            db.session.commit()


def test_membership_deactivation_stops_an_existing_session_immediately(client):
    login(client)
    with app.app.app_context():
        identity = app.PlatformIdentity.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        membership = app.UnitMembership.query.filter_by(
            identity_id=identity.id, unit_id=1
        ).one()
        membership.status = "suspended"
        db.session.commit()
        membership_id = membership.id
    try:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]
        with client.session_transaction() as browser_session:
            assert "_user_id" not in browser_session
    finally:
        with app.app.app_context():
            membership = db.session.get(app.UnitMembership, membership_id)
            membership.status = "active"
            db.session.commit()


def test_audit_evidence_cannot_be_modified_or_deleted_through_the_orm(client):
    login(client)
    with app.app.app_context():
        audit = ChangeLog(
            unit_id=1,
            who_user_id=1,
            entity_type="Assurance",
            entity_id=1,
            field="state",
            old_value="before",
            new_value="after",
        )
        db.session.add(audit)
        db.session.commit()
        audit_id = audit.id
        audit.new_value = "tampered"
        with pytest.raises(PermissionError, match="append-only"):
            db.session.commit()
        db.session.rollback()
        audit = db.session.get(ChangeLog, audit_id)
        assert audit.new_value == "after"
        db.session.delete(audit)
        with pytest.raises(PermissionError, match="append-only"):
            db.session.commit()
        db.session.rollback()
        assert db.session.get(ChangeLog, audit_id) is not None


def test_business_change_and_audit_evidence_roll_back_atomically(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(
            username=ADMIN_CREDENTIALS["username"]
        ).one()
        day = date(2031, 1, 7)
        assignment = Assignment(
            unit_id=1, staff_id=person.id, day=day, code="M", source="manual"
        )
        invalid_audit = AnnotationAudit(
            unit_id=1,
            assignment_id=None,
            actor_id=person.id,
            action=None,
        )
        db.session.add_all([assignment, invalid_audit])
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()
        assert Assignment.query.filter_by(staff_id=person.id, day=day).first() is None


def test_existing_leave_can_filter_by_watch_or_whole_unit(client):
    login(client)
    with app.app.app_context():
        watch_a = Watch.query.filter_by(unit_id=1, name="Watch A").one()
        watch_b = Watch.query.filter_by(unit_id=1, name="Watch B").one()
        person_a = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).one()
        person_b = Staff.query.filter_by(username="staff_test").one()
        original_watch_ids = (person_a.watch_id, person_b.watch_id)
        person_a.watch_id = watch_a.id
        person_b.watch_id = watch_b.id
        leave_a = Leave(
            unit_id=1,
            staff_id=person_a.id,
            leave_type="AL",
            start=date(2029, 7, 2),
            end=date(2029, 7, 3),
        )
        leave_b = Leave(
            unit_id=1,
            staff_id=person_b.id,
            leave_type="AL",
            start=date(2029, 7, 4),
            end=date(2029, 7, 5),
        )
        db.session.add_all([leave_a, leave_b])
        db.session.commit()
        leave_a_id, leave_b_id = leave_a.id, leave_b.id
        watch_a_id, watch_b_id = watch_a.id, watch_b.id

    whole_unit = client.get("/leave?ym=2029-07")
    watch_a_page = client.get(f"/leave?ym=2029-07&watch_id={watch_a_id}")
    watch_b_page = client.get(f"/leave?ym=2029-07&watch_id={watch_b_id}")

    assert whole_unit.status_code == 200
    assert b">Whole unit</option>" in whole_unit.data
    assert f'data-leave-id="{leave_a_id}"'.encode() in whole_unit.data
    assert f'data-leave-id="{leave_b_id}"'.encode() in whole_unit.data
    assert f'data-leave-id="{leave_a_id}"'.encode() in watch_a_page.data
    assert f'data-leave-id="{leave_b_id}"'.encode() not in watch_a_page.data
    assert f'data-leave-id="{leave_b_id}"'.encode() in watch_b_page.data
    assert f'data-leave-id="{leave_a_id}"'.encode() not in watch_b_page.data

    with app.app.app_context():
        Leave.query.filter(Leave.id.in_([leave_a_id, leave_b_id])).delete()
        person_a = Staff.query.filter_by(username=ADMIN_CREDENTIALS["username"]).one()
        person_b = Staff.query.filter_by(username="staff_test").one()
        person_a.watch_id, person_b.watch_id = original_watch_ids
        db.session.commit()


def test_airport_absence_catalogue_and_calendar_token(client):
    with app.app.app_context():
        app.MfaCredential.query.delete()
        db.session.commit()
    client.post("/logout", data={"_csrf_token": csrf(client)})
    login(client)
    added = client.post(
        "/leave?ym=2026-07",
        data={
            "_csrf_token": csrf(client),
            "form": "absence_type_add",
            "code": "CL",
            "label": "Compassionate leave",
            "category": "leave",
        },
        follow_redirects=True,
    )
    assert added.status_code == 200
    assert b"Compassionate leave (CL)" in added.data

    with app.app.app_context():
        person = Staff.query.filter_by(username="staff_test").one()
        person.calendar_token = None
        db.session.commit()
        person_id = person.id
    generated = client.post(
        f"/staff/{person_id}/calendar-token",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert generated.status_code == 200
    with app.app.app_context():
        person = db.session.get(Staff, person_id)
        token = person.calendar_token
        assert token
    feed = app.app.test_client().get(f"/calendar/{person_id}/{token}.ics")
    assert feed.status_code == 200
    assert b"BEGIN:VCALENDAR" in feed.data


def test_unit_messages_permission_boundary(client):
    client.post("/logout", data={"_csrf_token": csrf(client)})
    login(client)
    assert client.get("/messages").status_code == 200

    staff_client = app.app.test_client()
    login_as(staff_client, "staff_test")
    assert staff_client.get("/messages").status_code == 403

    wm_client = app.app.test_client()
    login_as(wm_client, "watch_manager_test")
    assert wm_client.get("/messages").status_code == 200


def test_unit_messages_recipient_order_and_default(client):
    login(client)
    page = client.get("/messages")
    assert page.status_code == 200
    content = page.data
    whole_unit = content.index(b'value="all" selected')
    watch = content.index(b'value="watch"')
    individual = content.index(b'value="individual"')
    assert whole_unit < watch < individual
    assert b">Whole unit</option>" in content
    assert b">Watch</option>" in content
    assert b">Individual</option>" in content
    assert b'data-recipient-detail="watch" hidden' in content
    assert b'data-recipient-detail="individual" hidden' in content
    assert b'data-recipient-detail="operational" hidden' in content
    assert b"updateRecipientDetails" in content


def test_admin_configures_airport_sms_numbers(client, monkeypatch):
    login(client)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+447700900999")

    response = client.post(
        "/admin",
        data={
            "_csrf_token": csrf(client),
            "form": "sms_settings",
            "sms_sender_numbers": (
                "Operations | +44 7700 900111\nBackup | +447700900112"
            ),
            "sms_operational_numbers": (
                "Duty desk | +44 141 555 0100\nSupervisor | +447700900113"
            ),
            "sms_default_sender": "+447700900112",
            "sms_default_operational_number": "+441415550100",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"SMS numbers saved for this airport." in response.data

    page = client.get("/messages")
    assert page.status_code == 200
    assert b"Operations" in page.data
    assert b"Duty desk" in page.data
    assert b'value="+447700900112" selected' in page.data
    assert b'value="+441415550100" selected' in page.data


def test_messages_rejects_unapproved_sender_and_sends_to_operational_number(
    client, monkeypatch
):
    login(client)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+447700900999")
    sent = []

    def fake_send(to_number, body, creds=None, from_number=None):
        sent.append((to_number, body, from_number))
        return True, "SMtest"

    monkeypatch.setattr(app, "_send_sms_via_twilio", fake_send)
    rejected = client.post(
        "/messages",
        data={
            "_csrf_token": csrf(client),
            "scope": "operational",
            "sender_number": "+447700900777",
            "operational_number": "+441415550100",
            "template": "custom",
            "message": "Test message",
        },
    )
    assert rejected.status_code == 400
    assert not sent

    response = client.post(
        "/messages",
        data={
            "_csrf_token": csrf(client),
            "scope": "operational",
            "sender_number": "+447700900112",
            "operational_number": "+441415550100",
            "template": "custom",
            "message": "Operational test",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert sent == [
        ("+441415550100", "Operational test", "+447700900112")
    ]
    assert b"SMS sent to 1 recipient." in response.data
    with app.app.app_context():
        audit = app.SmsAudit.query.order_by(app.SmsAudit.id.desc()).first()
        assert audit.sent_by_name == "Admin Test"
        assert audit.sender_number == "+447700900112"
        assert audit.recipient_number == "+441415550100"
        assert audit.recipient_label == "Duty desk"
        assert audit.message_type == "operational"
        assert audit.message_content == "Operational test"
        assert audit.provider_message_id == "SMtest"


def test_sms_audit_is_unit_admin_only(client):
    login(client)
    page = client.get("/admin/sms-audit")
    assert page.status_code == 200
    assert b"Operational test" in page.data
    assert b"Admin Test" in page.data

    wm_client = app.app.test_client()
    login_as(wm_client, "watch_manager_test")
    assert wm_client.get("/messages").status_code == 200
    assert wm_client.get("/admin/sms-audit").status_code == 403


def test_users_can_delete_only_their_own_read_notifications(client):
    login(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(username="admin_test").one()
        other = Staff.query.filter_by(username="staff_test").one()
        unread = app.Notification(
            unit_id=1, recipient_id=admin.id,
            kind="test", message="Unread notification",
        )
        read = app.Notification(
            unit_id=1, recipient_id=admin.id,
            kind="test", message="Read notification",
            read_at=app.utcnow(),
        )
        someone_else = app.Notification(
            unit_id=1, recipient_id=other.id,
            kind="test", message="Private notification",
            read_at=app.utcnow(),
        )
        db.session.add_all([unread, read, someone_else])
        db.session.commit()
        unread_id, read_id, other_id = unread.id, read.id, someone_else.id

    blocked = client.post(
        f"/notifications/{unread_id}/delete",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert b"Mark the notification as read before deleting it." in blocked.data
    with app.app.app_context():
        assert db.session.get(app.Notification, unread_id) is not None

    deleted = client.post(
        f"/notifications/{read_id}/delete",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert b"Notification deleted." in deleted.data
    with app.app.app_context():
        assert db.session.get(app.Notification, read_id) is None

    forbidden = client.post(
        f"/notifications/{other_id}/delete",
        data={"_csrf_token": csrf(client)},
    )
    assert forbidden.status_code == 404
    with app.app.app_context():
        assert db.session.get(app.Notification, other_id) is not None


def test_users_can_mark_their_notifications_read_individually(client):
    login(client)
    with app.app.app_context():
        admin = Staff.query.filter_by(username="admin_test").one()
        other = Staff.query.filter_by(username="staff_test").one()
        app.Notification.query.filter_by(
            unit_id=1, recipient_id=admin.id
        ).delete()
        first = app.Notification(
            unit_id=1, recipient_id=admin.id,
            kind="test", message="First unread notification",
        )
        second = app.Notification(
            unit_id=1, recipient_id=admin.id,
            kind="test", message="Second unread notification",
        )
        private = app.Notification(
            unit_id=1, recipient_id=other.id,
            kind="test", message="Another user's notification",
        )
        db.session.add_all([first, second, private])
        db.session.commit()
        first_id, second_id, private_id = first.id, second.id, private.id

    marked = client.post(
        f"/notifications/{first_id}/read",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert marked.status_code == 200
    assert b"Notification marked as read." in marked.data
    assert b"1 new" in marked.data
    with app.app.app_context():
        assert db.session.get(app.Notification, first_id).read_at is not None
        assert db.session.get(app.Notification, second_id).read_at is None

    forbidden = client.post(
        f"/notifications/{private_id}/read",
        data={"_csrf_token": csrf(client)},
    )
    assert forbidden.status_code == 404
    with app.app.app_context():
        assert db.session.get(app.Notification, private_id).read_at is None


def test_primary_navigation_matches_role_permissions():
    editor_client = app.app.test_client()
    login_as(editor_client, "editor_test")
    editor_page = editor_client.get("/")
    assert editor_page.status_code == 302
    editor_page = editor_client.get(editor_page.headers["Location"])
    assert b'href="/compliance-centre"' not in editor_page.data
    assert editor_client.get("/compliance-centre").status_code == 403

    wm_client = app.app.test_client()
    login_as(wm_client, "watch_manager_test")
    wm_page = wm_client.get("/roster/2025-04")
    assert wm_page.status_code == 200
    assert b"Secure session \xc2\xb7 Watch Manager" in wm_page.data

    dwm_client = app.app.test_client()
    login_as(dwm_client, "duty_watch_manager_test")
    dwm_page = dwm_client.get("/roster/2025-04")
    assert dwm_page.status_code == 200
    assert b"Secure session \xc2\xb7 Duty Watch Manager" in dwm_page.data


def test_unit_admin_seeds_standard_patterns_idempotently(client):
    with app.app.app_context():
        _clear_flexible_patterns()
    login(client)
    first = client.post(
        "/administration/work-patterns",
        data={"_csrf_token": csrf(client), "action": "seed"},
        follow_redirects=True,
    )
    assert first.status_code == 200
    assert b"Added 2 standard pattern(s)." in first.data
    second = client.post(
        "/administration/work-patterns",
        data={"_csrf_token": csrf(client), "action": "seed"},
        follow_redirects=True,
    )
    assert b"Standard patterns already exist." in second.data
    with app.app.app_context():
        patterns = app.WorkPattern.query.filter_by(unit_id=1).all()
        assert {row.name for row in patterns} == {
            "Standard 6-on/4-off", "Part-time 4-on/6-off",
        }
        assert all(row.cycle_length_days == 10 for row in patterns)
        six_on = next(row for row in patterns if row.name.startswith("Standard"))
        days = app.WorkPatternDay.query.filter_by(
            unit_id=1, work_pattern_id=six_on.id
        ).order_by(app.WorkPatternDay.day_index).all()
        assert [row.day_type for row in days] == ["FIXED_SHIFT"] * 6 + ["OFF"] * 4


def test_pattern_admin_creates_and_configures_a_custom_cycle(client):
    login(client)
    response = client.post(
        "/administration/work-patterns",
        data={
            "_csrf_token": csrf(client), "action": "create",
            "name": "Three-day flexible test", "cycle_length_days": "3",
            "contracted_minutes_per_cycle": "960",
            "description": "Test pattern",
        },
        follow_redirects=False,
    )
    assert response.status_code == 302
    with app.app.app_context():
        pattern = app.WorkPattern.query.filter_by(
            unit_id=1, name="Three-day flexible test"
        ).one()
        morning = ShiftType.query.filter_by(unit_id=1, code="M").one()
        afternoon = ShiftType.query.filter_by(unit_id=1, code="A").one()
        pattern_id, morning_id, afternoon_id = pattern.id, morning.id, afternoon.id
    saved = client.post(
        f"/administration/work-patterns/{pattern_id}",
        data={
            "_csrf_token": csrf(client), "name": "Three-day flexible test",
            "cycle_length_days": "3", "contracted_minutes_per_cycle": "960",
            "description": "Configured test pattern", "is_active": "on",
            "day_type_0": "FIXED_SHIFT", "fixed_shift_type_id_0": str(morning_id),
            "required_work_0": "on", "notes_0": "Morning duty",
            "day_type_1": "WORK_ALLOWED_SET",
            "allowed_shift_type_ids_1": [str(morning_id), str(afternoon_id)],
            "required_work_1": "on", "notes_1": "Flexible duty",
            "day_type_2": "OFF", "notes_2": "Protected rest",
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert b"Pattern saved." in saved.data
    assert b"28-day preview" in saved.data
    with app.app.app_context():
        days = app.WorkPatternDay.query.filter_by(
            unit_id=1, work_pattern_id=pattern_id
        ).order_by(app.WorkPatternDay.day_index).all()
        assert [row.day_type for row in days] == [
            "FIXED_SHIFT", "WORK_ALLOWED_SET", "OFF",
        ]
        allowed = app.WorkPatternDayAllowedShift.query.filter_by(
            unit_id=1, work_pattern_day_id=days[1].id
        ).all()
        assert {row.shift_type_id for row in allowed} == {morning_id, afternoon_id}


def test_admin_assigns_dated_pattern_and_hard_staff_rule(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(unit_id=1, username="staff_test").one()
        pattern = app.WorkPattern.query.filter_by(
            unit_id=1, name="Standard 6-on/4-off"
        ).one()
        person_id, pattern_id = person.id, pattern.id
    assigned = client.post(
        f"/administration/staff/{person_id}/work-rules",
        data={
            "_csrf_token": csrf(client), "action": "assign_pattern",
            "work_pattern_id": str(pattern_id), "effective_from": "2026-09-01",
            "anchor_date": "2026-09-01", "anchor_day_index": "0",
            "notes": "Permanent cycle",
        },
        follow_redirects=True,
    )
    assert assigned.status_code == 200
    assert b"Effective-dated pattern assignment added." in assigned.data
    rule = client.post(
        f"/administration/staff/{person_id}/work-rules",
        data={
            "_csrf_token": csrf(client), "action": "add_rule",
            "rule_type": "NO_NIGHT", "hardness": "HARD",
            "effective_from": "2026-09-01", "penalty_weight": "1",
            "reason": "Medical restriction",
        },
        follow_redirects=True,
    )
    assert rule.status_code == 200
    assert b"Staff rule added." in rule.data
    with app.app.app_context():
        assert app.StaffPatternAssignment.query.filter_by(
            unit_id=1, staff_id=person_id, work_pattern_id=pattern_id
        ).count() == 1
        stored_rule = app.StaffRule.query.filter_by(
            unit_id=1, staff_id=person_id, rule_type="NO_NIGHT"
        ).one()
        assert stored_rule.hardness == "HARD"
    locked = client.post(
        f"/administration/work-patterns/{pattern_id}",
        data={"_csrf_token": csrf(client), "action": "save"},
        follow_redirects=True,
    )
    assert locked.status_code == 200
    assert b"Assigned patterns are locked to preserve roster history." in locked.data

    invalid_soft_restriction = client.post(
        f"/administration/staff/{person_id}/work-rules",
        data={
            "_csrf_token": csrf(client), "action": "add_rule",
            "rule_type": "NO_NIGHT", "hardness": "SOFT",
            "effective_from": "2026-10-01", "penalty_weight": "5",
        },
        follow_redirects=True,
    )
    assert b"This restriction must be configured as a hard rule." in invalid_soft_restriction.data
    with app.app.app_context():
        assert app.StaffRule.query.filter_by(
            unit_id=1, staff_id=person_id, rule_type="NO_NIGHT"
        ).count() == 1


def test_flexible_pattern_admin_is_permission_and_tenant_scoped():
    ordinary = app.app.test_client()
    login_as(ordinary, "staff_test")
    assert ordinary.get("/administration/work-patterns").status_code == 403
    assert ordinary.get(
        "/administration/work-patterns/migration"
    ).status_code == 403

    admin_client = app.app.test_client()
    login(admin_client)
    with app.app.app_context():
        other = Staff.query.filter_by(unit_id=3, username="other_staff_test").one()
        other_id = other.id
    assert admin_client.get(
        f"/administration/staff/{other_id}/work-rules"
    ).status_code == 404


def test_admin_dry_runs_and_migrates_only_exact_legacy_pattern(client):
    login(client)
    effective_from = date(2025, 1, 1)
    with app.app.app_context():
        app.work_pattern_admin_service.seed_standard_patterns(1)
        person = Staff(
            unit_id=1,
            username="legacy_pattern_migration_test",
            password_hash="unused",
            name="Legacy Migration Test",
            staff_no="LEG-MIG-1",
            role="user",
            pattern_override=True,
            pattern_csv="M,M,A,A,N,N,OFF,OFF,OFF,OFF",
            pattern_anchor=effective_from,
        )
        db.session.add(person)
        db.session.commit()
        person_id = person.id
        duties_before = [
            (row.id, row.staff_id, row.day, row.code, row.version)
            for row in Assignment.query.filter_by(unit_id=1).order_by(Assignment.id)
        ]

    dry_run = client.get(
        "/administration/work-patterns/migration?effective_from=2025-01-01"
    )
    assert dry_run.status_code == 200
    assert b"Exact Match" in dry_run.data
    assert b"Standard 6-on/4-off" in dry_run.data

    migrated = client.post(
        "/administration/work-patterns/migration",
        data={
            "_csrf_token": csrf(client),
            "effective_from": "2025-01-01",
            "staff_ids": str(person_id),
        },
        follow_redirects=True,
    )
    assert migrated.status_code == 200
    assert b"Existing roster duties were not changed." in migrated.data
    with app.app.app_context():
        row = app.StaffPatternAssignment.query.filter_by(
            unit_id=1, staff_id=person_id, effective_from=effective_from
        ).one()
        pattern = app.WorkPattern.query.filter_by(id=row.work_pattern_id).one()
        assert pattern.name == "Standard 6-on/4-off"
        assert row.anchor_date == effective_from
        duties_after = [
            (item.id, item.staff_id, item.day, item.code, item.version)
            for item in Assignment.query.filter_by(unit_id=1).order_by(Assignment.id)
        ]
        assert duties_after == duties_before
        db.session.delete(row)
        db.session.delete(db.session.get(Staff, person_id))
        db.session.commit()


def test_roster_shows_pattern_breach_and_blocks_publication(client):
    login(client)
    with app.app.app_context():
        person = Staff.query.filter_by(unit_id=1, username="staff_test").one()
        morning = ShiftType.query.filter_by(unit_id=1, code="M").one()
        pattern = app.WorkPattern(
            unit_id=1, name="Publication blocker test", cycle_length_days=1,
            contracted_minutes_per_cycle=0,
        )
        db.session.add(pattern)
        db.session.flush()
        db.session.add(app.WorkPatternDay(
            unit_id=1, work_pattern_id=pattern.id, day_index=0,
            day_type="OFF", required_work=False,
        ))
        db.session.add(app.StaffPatternAssignment(
            unit_id=1, staff_id=person.id, work_pattern_id=pattern.id,
            effective_from=date(2025, 9, 1), effective_to=date(2025, 9, 30),
            anchor_date=date(2025, 9, 1), anchor_day_index=0,
        ))
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=person.id, day=date(2025, 9, 1)
        ).first()
        if not assignment:
            assignment = Assignment(
                unit_id=1, staff_id=person.id, day=date(2025, 9, 1), code="M"
            )
            db.session.add(assignment)
        assignment.code = morning.code
        db.session.commit()
        pattern_id = pattern.id

    roster = client.get("/roster/2025-09")
    assert roster.status_code == 200
    assert b"Pre-publication validation" not in roster.data
    assert b"Publication blocker" in roster.data
    assert b"protected non-working day" in roster.data
    assert b"Resolve blocking validation findings first" in roster.data

    published = client.post(
        "/roster/2025-09/publish",
        data={"_csrf_token": csrf(client)},
        follow_redirects=True,
    )
    assert published.status_code == 200
    assert b"Roster publication blocked" in published.data
    with app.app.app_context():
        assert app.RosterPublication.query.filter_by(
            unit_id=1, year=2025, month=9, state="published"
        ).count() == 0
        app.StaffPatternAssignment.query.filter_by(
            unit_id=1, work_pattern_id=pattern_id
        ).delete()
        app.WorkPatternDay.query.filter_by(
            unit_id=1, work_pattern_id=pattern_id
        ).delete()
        app.WorkPattern.query.filter_by(id=pattern_id, unit_id=1).delete()
        db.session.commit()


def test_roster_keeps_day_header_below_sticky_site_header(client):
    login(client)
    response = client.get("/roster/2025-09")

    assert response.status_code == 200
    assert b"--roster-sticky-top" in response.data
    assert b"ResizeObserver(updateRosterStickyTop)" in response.data
    assert b"Math.ceil(height / scale)" in response.data
    assert b"MutationObserver(updateRosterStickyTop)" in response.data
    assert response.data.count(b'class="sticky left col-name"') == 1
    assert b'class="sticky left col-date"' not in response.data
    assert b'class="sticky col-date">Medical' in response.data
    with open(
        os.path.join(app.BASE_DIR, "static", "styles.css"), encoding="utf-8"
    ) as stylesheet_file:
        stylesheet = stylesheet_file.read()
    assert ".roster th.dayhead" in stylesheet
    assert "top:var(--roster-sticky-top, 0px)" in stylesheet
    assert "table.roster th.sticky" in stylesheet
    assert "table.roster th.sticky{" in stylesheet
    assert ".roster th.dayhead.is-today" in stylesheet
    assert "background: #363d2e !important;" in stylesheet
    assert ".roster th.col-name{\n  z-index:12;" in stylesheet
    assert "#roster{overflow:visible;}" in stylesheet
    assert "#roster{overflow-x:auto" not in stylesheet
