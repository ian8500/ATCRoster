from datetime import date, time

import pytest

import app
from conftest import finish_operational_login
from app import (
    AnnotationAudit,
    AnnotationType,
    Assignment,
    Notification,
    PersonQualification,
    QualificationType,
    RequestAudit,
    ShiftRequest,
    ShiftType,
    Staff,
    Unit,
    Watch,
    _lock_date_for_target_month,
    _request_date_bounds,
    db,
    refresh_annotation_cache,
    refresh_shift_cache,
)


@pytest.fixture()
def secured_client():
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        unit_a = Unit(id=1, code="AAA", name="Airport A", request_months_ahead=3, request_lock_day=20)
        unit_b = Unit(id=2, code="BBB", name="Airport B", request_months_ahead=6, request_lock_day=15)
        watch_a = Watch(id=1, unit_id=1, name="A")
        watch_b = Watch(id=2, unit_id=2, name="B")
        admin = Staff(unit_id=1, username="admin-a", name="Admin A", staff_no="A1", role="admin", watch=watch_a)
        user = Staff(unit_id=1, username="user-a", name="User A", staff_no="A2", role="user", watch=watch_a)
        other = Staff(unit_id=2, username="user-b", name="Secret User B", staff_no="B1", role="user", watch=watch_b)
        for member in (admin, user, other):
            member.set_password("password123")
        shifts = [
            ShiftType(unit_id=1, code="REQ", name="Requestable", start_time=time(9), end_time=time(17),
                      is_working=True, is_active=True, is_requestable=True),
            ShiftType(unit_id=1, code="HID", name="Not requestable", is_working=True,
                      is_active=True, is_requestable=False),
            ShiftType(unit_id=1, code="REST", name="Invalid rest request",
                      is_working=False, is_active=True, is_requestable=True),
            ShiftType(unit_id=1, code="QUAL", name="Qualified", start_time=time(9), end_time=time(17),
                      is_working=True, is_active=True, is_requestable=True,
                      required_qualification="medical"),
            ShiftType(unit_id=2, code="REQ", name="Other requestable", is_working=True,
                      is_active=True, is_requestable=True),
        ]
        db.session.add_all([unit_a, unit_b, watch_a, watch_b, admin, user, other, *shifts])
        db.session.flush()
        for member in (admin, user, other):
            identity = app.PlatformIdentity(
                public_id=f"test-{member.username}",
                username=member.username,
                password_hash=member.password_hash,
            )
            db.session.add(identity)
            db.session.flush()
            db.session.add(app.UnitMembership(
                identity_id=identity.id,
                unit_id=member.unit_id,
                person_id=member.id,
                role="UnitAdmin" if member.role == "admin" else "StaffUser",
                status="active",
            ))
        db.session.commit()
        refresh_shift_cache()
    client = app.app.test_client()
    yield client
    with app.app.app_context():
        db.session.remove()
        db.drop_all()


def login(client, username="user-a"):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    response = client.post("/login", data={
        "_csrf_token": token,
        "username": username,
        "password": "password123",
    })
    assert response.status_code == 302
    finish_operational_login(client)
    response = client.get("/requests")
    assert response.status_code == 200
    with client.session_transaction() as sess:
        return sess["_csrf_token"]


def request_day():
    # October remains within the three-month window on the test execution date.
    return date(2026, 10, 10)


def test_configurable_window_and_lock_boundaries(secured_client):
    with app.app.app_context():
        assert _request_date_bounds(date(2026, 7, 24), 1) == (
            date(2026, 8, 1), date(2026, 10, 31)
        )
        assert _request_date_bounds(date(2026, 7, 24), 2)[1] == date(2027, 1, 31)
        assert _lock_date_for_target_month(2026, 10, 1) == date(2026, 9, 20)
        assert _lock_date_for_target_month(2026, 10, 2) == date(2026, 9, 15)


def test_comment_persistence_update_and_single_record(secured_client):
    token = login(secured_client)
    day = request_day().isoformat()
    first = secured_client.post("/requests", data={
        "_csrf_token": token, "form": "add", "day": day, "code": "REQ", "comment": "First note",
    })
    assert first.status_code == 302
    second = secured_client.post("/requests", data={
        "_csrf_token": token, "form": "add", "day": day, "code": "REQ", "comment": "Updated note",
    })
    assert second.status_code == 302
    with app.app.app_context():
        rows = ShiftRequest.query.filter_by(unit_id=1, day=request_day()).all()
        assert len(rows) == 1
        assert rows[0].requester_comment == "Updated note"
        assert RequestAudit.query.filter_by(request_id=rows[0].id).count() == 2


def test_profile_navigation_shows_and_clears_unread_notification_count(
    secured_client,
):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        db.session.add_all([
            Notification(
                unit_id=1, recipient_id=user.id,
                kind="request_update", message="First unread update",
            ),
            Notification(
                unit_id=1, recipient_id=user.id,
                kind="request_update", message="Second unread update",
            ),
            Notification(
                unit_id=1, recipient_id=user.id,
                kind="request_update", message="Already read",
                read_at=app.utcnow(),
            ),
        ])
        db.session.commit()

    token = login(secured_client, "user-a")
    page = secured_client.get("/requests")
    assert b"Profile, 2 unread notifications" in page.data
    assert b"nav-notification-count" in page.data

    marked = secured_client.post(
        "/notifications/read",
        data={"_csrf_token": token},
    )
    assert marked.status_code == 302
    page = secured_client.get("/requests")
    assert b"unread notification" not in page.data
    assert b"nav-notification-count" not in page.data


def test_csrf_and_non_requestable_shift_rejected(secured_client):
    login(secured_client)
    missing = secured_client.post("/requests", data={
        "form": "add", "day": request_day().isoformat(), "code": "REQ",
    })
    assert missing.status_code == 400
    with secured_client.session_transaction() as sess:
        token = sess["_csrf_token"]
    invalid = secured_client.post("/requests", data={
        "_csrf_token": token, "form": "add", "day": request_day().isoformat(), "code": "HID",
    }, follow_redirects=True)
    assert b"inactive or cannot be requested" in invalid.data
    non_working = secured_client.post("/requests", data={
        "_csrf_token": token, "form": "add",
        "day": request_day().isoformat(), "code": "REST",
    }, follow_redirects=True)
    assert b"inactive or cannot be requested" in non_working.data


def test_malformed_admin_month_is_safely_rejected_to_default(secured_client):
    login(secured_client, "admin-a")
    response = secured_client.get("/requests?ym=not-a-month")
    assert response.status_code == 200
    assert b"Shift Requests" in response.data


def test_approved_request_cannot_be_forged_deleted(secured_client):
    token = login(secured_client)
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        row = ShiftRequest(unit_id=1, staff_id=user.id, day=request_day(), code="REQ", status="approved")
        db.session.add(row)
        db.session.commit()
        rid = row.id
    response = secured_client.post("/requests", data={
        "_csrf_token": token, "form": "del", "rid": rid,
    })
    assert response.status_code == 409
    with app.app.app_context():
        assert db.session.get(ShiftRequest, rid).status == "approved"


def test_invalid_status_and_cross_unit_isolation(secured_client):
    with app.app.app_context():
        user_a = Staff.query.filter_by(username="user-a").one()
        user_b = Staff.query.filter_by(username="user-b").one()
        own = ShiftRequest(unit_id=1, staff_id=user_a.id, day=request_day(), code="REQ")
        secret = ShiftRequest(unit_id=2, staff_id=user_b.id, day=request_day(), code="REQ",
                              requester_comment="UNIT-B-SECRET")
        db.session.add_all([own, secret])
        db.session.commit()
        own_id = own.id
        secret_id = secret.id
    token = login(secured_client, "admin-a")
    invalid = secured_client.post(f"/admin/requests/{own_id}/respond", data={
        "_csrf_token": token, "status": "hacked", "action": "status",
    })
    assert invalid.status_code == 400
    cross_write = secured_client.post(f"/admin/requests/{secret_id}/respond", data={
        "_csrf_token": token, "status": "approved", "action": "status",
    })
    assert cross_write.status_code == 404
    page = secured_client.get(f"/requests?ym={request_day():%Y-%m}")
    assert b"UNIT-B-SECRET" not in page.data


@pytest.mark.parametrize(
    ("start", "target", "expected_status"),
    [
        ("pending", "approved", 302),
        ("pending", "rejected", 302),
        ("pending", "cancelled", 302),
        ("pending", "pending", 409),
        ("pending", "fulfilled", 400),
        ("approved", "rejected", 302),
        ("approved", "cancelled", 302),
        ("approved", "approved", 409),
        ("rejected", "approved", 409),
        ("cancelled", "approved", 409),
        ("fulfilled", "rejected", 409),
    ],
)
def test_request_transition_table(
    secured_client, start, target, expected_status
):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        row = ShiftRequest(
            unit_id=1, staff_id=user.id, day=request_day(),
            code="REQ", status=start,
        )
        if start == "fulfilled":
            assignment = Assignment(
                unit_id=1, staff_id=user.id, day=request_day(),
                code="REQ", source="request",
            )
            db.session.add(assignment)
            db.session.flush()
            row.resulting_assignment_id = assignment.id
            row.fulfilled_at = app.utcnow()
        db.session.add(row)
        db.session.commit()
        request_id = row.id
    token = login(secured_client, "admin-a")
    response = secured_client.post(
        f"/admin/requests/{request_id}/respond",
        data={
            "_csrf_token": token,
            "action": "status",
            "status": target,
            "admin_response": "Documented operational reason",
        },
    )
    assert response.status_code == expected_status


def test_admin_cancellation_records_timestamp_audit_and_safe_redirect(secured_client):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        row = ShiftRequest(
            unit_id=1, staff_id=user.id, day=request_day(), code="REQ",
        )
        db.session.add(row)
        db.session.commit()
        rid = row.id
    token = login(secured_client, "admin-a")
    response = secured_client.post(
        f"/admin/requests/{rid}/respond",
        data={
            "_csrf_token": token,
            "action": "status",
            "status": "cancelled",
            "admin_response": "Operational reason",
            "ym": "../../platform/admin",
        },
    )
    assert response.status_code == 302
    assert "/requests?ym=" in response.headers["Location"]
    with app.app.app_context():
        row = db.session.get(ShiftRequest, rid)
        assert row.cancelled_at is not None
        audit = RequestAudit.query.filter_by(
            request_id=rid, transition="status"
        ).one()
        assert "cancelled" in audit.new_value


def test_sequential_tenant_sessions_do_not_reuse_first_unit_filter(secured_client):
    with app.app.app_context():
        user_a = Staff.query.filter_by(username="user-a").one()
        user_b = Staff.query.filter_by(username="user-b").one()
        db.session.add_all([
            ShiftRequest(
                unit_id=1, staff_id=user_a.id, day=request_day(), code="REQ",
                requester_comment="AIRPORT-A-ONLY",
            ),
            ShiftRequest(
                unit_id=2, staff_id=user_b.id, day=request_day(), code="REQ",
                requester_comment="AIRPORT-B-ONLY",
            ),
        ])
        db.session.commit()

    login(secured_client, "user-a")
    airport_a = secured_client.get("/requests")
    assert b"AIRPORT-A-ONLY" in airport_a.data
    assert b"AIRPORT-B-ONLY" not in airport_a.data
    assert b"Airport A" in airport_a.data

    second_client = app.app.test_client()
    login(second_client, "user-b")
    airport_b = second_client.get("/requests")
    assert b"AIRPORT-B-ONLY" in airport_b.data
    assert b"AIRPORT-A-ONLY" not in airport_b.data
    assert b"Airport B" in airport_b.data


def test_operational_assurance_is_tenant_isolated(secured_client):
    with app.app.app_context():
        own = app.OperationalPosition(
            unit_id=1, code="TWR-A", label="Airport A Tower"
        )
        secret = app.OperationalPosition(
            unit_id=2, code="TWR-B", label="AIRPORT-B-SECRET-POSITION"
        )
        db.session.add_all([own, secret])
        db.session.commit()
        secret_id = secret.id
    token = login(secured_client, "admin-a")
    page = secured_client.get("/operations/2026-10")
    assert page.status_code == 200
    assert b"Airport A Tower" in page.data
    assert b"AIRPORT-B-SECRET-POSITION" not in page.data
    cross_write = secured_client.post("/operations/2026-10", data={
        "_csrf_token": token,
        "action": "grant_endorsement",
        "person_id": 1,
        "position_id": secret_id,
        "valid_from": "2026-01-01",
    })
    assert cross_write.status_code == 404


def test_rule_approval_supersedes_only_current_airport(secured_client):
    with app.app.app_context():
        own_rule = app.RosterRuleVersion(
            unit_id=1, version=1, name="Airport A draft",
            state="draft", rules_json="{}",
            change_reference="A-CHANGE",
            consultation_summary="Airport A consultation completed.",
        )
        other_rule = app.RosterRuleVersion(
            unit_id=2, version=1, name="Airport B approved",
            state="approved", rules_json="{}",
            change_reference="B-CHANGE",
            consultation_summary="Airport B consultation completed.",
        )
        db.session.add_all([own_rule, other_rule])
        db.session.commit()
        own_id = own_rule.id
        other_id = other_rule.id
    token = login(secured_client, "admin-a")
    response = secured_client.post("/operations/2026-10", data={
        "_csrf_token": token,
        "action": "approve_rule_version",
        "rule_id": own_id,
        "effective_from": "2026-10-01",
    })
    assert response.status_code == 302
    with app.app.app_context():
        assert db.session.get(app.RosterRuleVersion, own_id).state == "approved"
        assert db.session.get(app.RosterRuleVersion, other_id).state == "approved"


def test_approve_only_then_apply_and_notify(secured_client):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        row = ShiftRequest(unit_id=1, staff_id=user.id, day=request_day(), code="REQ")
        db.session.add(row)
        db.session.commit()
        rid = row.id
    token = login(secured_client, "admin-a")
    approved = secured_client.post(f"/admin/requests/{rid}/respond", data={
        "_csrf_token": token, "action": "approve_only", "admin_response": "Approved",
        "ym": f"{request_day():%Y-%m}",
    })
    assert approved.status_code == 302
    with app.app.app_context():
        row = db.session.get(ShiftRequest, rid)
        assert row.status == "approved"
        assert row.resulting_assignment_id is None
    applied = secured_client.post(f"/admin/requests/{rid}/respond", data={
        "_csrf_token": token, "action": "approve_apply",
        "admin_response": "Approved override after operational review",
        "ym": f"{request_day():%Y-%m}", "confirm_override": "yes",
    })
    assert applied.status_code == 302
    with app.app.app_context():
        row = db.session.get(ShiftRequest, rid)
        assert row.status == "fulfilled"
        assert db.session.get(Assignment, row.resulting_assignment_id).code == "REQ"
        assert Notification.query.filter_by(recipient_id=row.staff_id, kind="shift_request_fulfilled").count() == 1
    roster = secured_client.get(f"/roster/{request_day():%Y-%m}")
    assert roster.status_code == 200
    assert b"request-applied" in roster.data
    assert b"Applied from an approved shift request" in roster.data
    assert b"request-applied-marker" in roster.data


def test_simple_manager_approve_and_refuse_actions_notify_user(secured_client):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        approved_request = ShiftRequest(
            unit_id=1, staff_id=user.id, day=request_day(), code="REQ"
        )
        refused_request = ShiftRequest(
            unit_id=1,
            staff_id=user.id,
            day=request_day() + app.timedelta(days=1),
            code="REQ",
        )
        db.session.add_all([approved_request, refused_request])
        db.session.commit()
        approved_id = approved_request.id
        refused_id = refused_request.id

    token = login(secured_client, "admin-a")
    attention_page = secured_client.get(
        f"/requests?ym={request_day():%Y-%m}"
    )
    assert b"shift requests awaiting your decision" in attention_page.data
    assert b"nav-attention-count" in attention_page.data
    assert b"Requests, 2 awaiting decision" in attention_page.data
    approved = secured_client.post(
        f"/admin/requests/{approved_id}/respond",
        data={
            "_csrf_token": token,
            "action": "approve",
            "admin_response": "Approved for requested cover.",
            "ym": f"{request_day():%Y-%m}",
        },
    )
    assert approved.status_code == 302

    refused = secured_client.post(
        f"/admin/requests/{refused_id}/respond",
        data={
            "_csrf_token": token,
            "action": "refuse",
            "admin_response": "Unable to release the requested shift.",
            "ym": f"{request_day():%Y-%m}",
        },
    )
    assert refused.status_code == 302

    with app.app.app_context():
        approved_row = db.session.get(ShiftRequest, approved_id)
        refused_row = db.session.get(ShiftRequest, refused_id)
        assert approved_row.status == "fulfilled"
        assert db.session.get(
            Assignment, approved_row.resulting_assignment_id
        ).code == "REQ"
        assert refused_row.status == "rejected"
        approved_notice = Notification.query.filter_by(
            recipient_id=approved_row.staff_id,
            kind="shift_request_fulfilled",
        ).order_by(Notification.id.desc()).first()
        refused_notice = Notification.query.filter_by(
            recipient_id=refused_row.staff_id,
            kind="shift_request_rejected",
        ).order_by(Notification.id.desc()).first()
        assert "approved and added to the roster" in approved_notice.message
        assert "Approved for requested cover." in approved_notice.message
        assert "was refused" in refused_notice.message
        assert "Unable to release" in refused_notice.message


def test_admin_self_approval_requires_another_admin_unless_sole_admin(
    secured_client,
):
    with app.app.app_context():
        watch = db.session.get(Watch, 1)
        requester = Staff.query.filter_by(username="admin-a").one()
        second_admin = Staff(
            unit_id=1, username="admin-second", name="Second Admin",
            staff_no="A3", role="admin", membership_status="active",
            watch=watch,
        )
        second_admin.set_password("password123")
        request_row = ShiftRequest(
            unit_id=1, staff_id=requester.id,
            day=request_day(), code="REQ",
        )
        db.session.add_all([second_admin, request_row])
        db.session.commit()
        request_id = request_row.id
        second_admin_id = second_admin.id

    token = login(secured_client, "admin-a")
    page = secured_client.get(f"/requests?ym={request_day():%Y-%m}")
    assert b"Another administrator must approve your request." in page.data
    blocked = secured_client.post(
        f"/admin/requests/{request_id}/respond",
        data={
            "_csrf_token": token,
            "action": "approve",
            "admin_response": "Attempted self approval",
            "ym": f"{request_day():%Y-%m}",
        },
    )
    assert blocked.status_code == 403
    with app.app.app_context():
        assert db.session.get(ShiftRequest, request_id).status == "pending"
        db.session.get(Staff, second_admin_id).membership_status = "inactive"
        db.session.commit()

    allowed = secured_client.post(
        f"/admin/requests/{request_id}/respond",
        data={
            "_csrf_token": token,
            "action": "approve",
            "admin_response": "Sole administrator approval",
            "ym": f"{request_day():%Y-%m}",
        },
    )
    assert allowed.status_code == 302
    with app.app.app_context():
        row = db.session.get(ShiftRequest, request_id)
        assert row.status == "fulfilled"
        assert row.resulting_assignment_id is not None


def test_requester_can_dismiss_only_fulfilled_or_rejected_requests(
    secured_client,
):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        fulfilled = ShiftRequest(
            unit_id=1,
            staff_id=user.id,
            day=request_day(),
            code="REQ",
            status="fulfilled",
            requester_comment="DISMISS-FULFILLED",
        )
        pending = ShiftRequest(
            unit_id=1,
            staff_id=user.id,
            day=request_day() + app.timedelta(days=1),
            code="REQ",
            status="pending",
            requester_comment="KEEP-PENDING",
        )
        db.session.add_all([fulfilled, pending])
        db.session.commit()
        fulfilled_id = fulfilled.id
        pending_id = pending.id

    token = login(secured_client, "user-a")
    removed = secured_client.post(
        "/requests",
        data={
            "_csrf_token": token,
            "form": "dismiss",
            "rid": fulfilled_id,
        },
        follow_redirects=True,
    )
    assert removed.status_code == 200
    assert b"DISMISS-FULFILLED" not in removed.data
    assert b"KEEP-PENDING" in removed.data

    denied = secured_client.post(
        "/requests",
        data={
            "_csrf_token": token,
            "form": "dismiss",
            "rid": pending_id,
        },
    )
    assert denied.status_code == 409

    with app.app.app_context():
        row = db.session.get(ShiftRequest, fulfilled_id)
        assert row is not None
        assert row.dismissed_by_requester_at is not None
        assert RequestAudit.query.filter_by(
            request_id=fulfilled_id,
            transition="dismissed_by_requester",
        ).count() == 1


def test_qualification_conflict_requires_confirmed_override(secured_client):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        row = ShiftRequest(unit_id=1, staff_id=user.id, day=request_day(), code="QUAL")
        db.session.add(row)
        db.session.commit()
        rid = row.id
    token = login(secured_client, "admin-a")
    response = secured_client.post(f"/admin/requests/{rid}/respond", data={
        "_csrf_token": token, "action": "approve_apply", "ym": f"{request_day():%Y-%m}",
    }, follow_redirects=True)
    assert b"has conflicts" in response.data
    with app.app.app_context():
        assert db.session.get(ShiftRequest, rid).status == "pending"


def test_authoritative_qualification_is_duty_date_and_unit_scoped(
    secured_client,
):
    with app.app.app_context():
        user = Staff.query.filter_by(username="user-a").one()
        own_type = QualificationType(
            unit_id=1, code="MEDICAL", label="Medical",
            expiry_required=True, is_active=True,
        )
        other_type = QualificationType(
            unit_id=2, code="MEDICAL", label="Other medical",
            expiry_required=True, is_active=True,
        )
        db.session.add_all([own_type, other_type])
        db.session.flush()
        db.session.add(PersonQualification(
            unit_id=1, person_id=user.id,
            qualification_type_id=own_type.id,
            valid_from=request_day() - app.timedelta(days=30),
            expires_on=request_day() + app.timedelta(days=30),
            status="valid",
        ))
        row = ShiftRequest(
            unit_id=1, staff_id=user.id, day=request_day(), code="QUAL"
        )
        db.session.add(row)
        db.session.commit()
        request_id = row.id
    token = login(secured_client, "admin-a")
    with app.app.app_context():
        from tenancy import bind_authenticated_unit, reset_authenticated_unit
        context_token = bind_authenticated_unit(1)
        try:
            user = Staff.query.filter_by(username="user-a").one()
            shift = ShiftType.query.filter_by(unit_id=1, code="QUAL").one()
            assert app._staff_has_shift_qualification(
                user, shift, request_day()
            )
        finally:
            reset_authenticated_unit(context_token)
    response = secured_client.post(
        f"/admin/requests/{request_id}/respond",
        data={
            "_csrf_token": token,
            "action": "approve_apply",
            "admin_response": "Competence and operational conflicts checked",
            "confirm_override": "yes",
        },
    )
    assert response.status_code == 302
    with app.app.app_context():
        row = db.session.get(ShiftRequest, request_id)
        assert row.status == "fulfilled"
        assert row.resulting_assignment_id is not None


def test_annotation_application_is_tenant_scoped_and_audited(secured_client):
    with app.app.app_context():
        secret = AnnotationType(
            unit_id=2, code="SECRET", label="UNIT-B-SECRET",
            is_active=True,
        )
        own = AnnotationType(
            unit_id=1, code="OWN", label="Own annotation",
            is_active=True,
        )
        other_person = Staff.query.filter_by(username="user-b").one()
        db.session.add_all([secret, own])
        db.session.commit()
        other_person_id = other_person.id
        refresh_annotation_cache()
    token = login(secured_client, "admin-a")
    page = secured_client.get(f"/roster/{request_day():%Y-%m}")
    assert b"Own annotation" in page.data
    assert b"UNIT-B-SECRET" not in page.data
    cross_write = secured_client.post(
        f"/assign/{other_person_id}/{request_day():%Y-%m}/{request_day():%Y-%m-%d}",
        data={"_csrf_token": token, "annotation": "SECRET"},
    )
    assert cross_write.status_code == 404


def test_roster_shows_annotation_once_and_hover_detail_can_be_edited(
    secured_client,
):
    with app.app.app_context():
        definition = AnnotationType(
            unit_id=1, code="NEAT", label="Neat annotation",
            is_active=True,
        )
        target = Staff.query.filter_by(username="user-a").one()
        db.session.add(definition)
        db.session.commit()
        target_id = target.id
        refresh_annotation_cache()

    token = login(secured_client, "admin-a")
    with app.app.app_context():
        db.session.add(Assignment(
            unit_id=1,
            staff_id=target_id,
            day=request_day(),
            code="",
            source="manual",
        ))
        db.session.commit()
    endpoint = (
        f"/assign/{target_id}/{request_day():%Y-%m}/"
        f"{request_day():%Y-%m-%d}"
    )
    applied = secured_client.post(
        endpoint,
        data={"_csrf_token": token, "annotation": "NEAT"},
    )
    assert applied.status_code == 302

    page = secured_client.get(f"/roster/{request_day():%Y-%m}")
    assert page.status_code == 200
    assert b'<option value="" selected>' in page.data
    assert b'class="annotation-code annotation-code--editable"' in page.data
    assert b"annotation-display--shift-line" in page.data
    assert b'class="annotation-dialog"' in page.data
    assert b'class="annotation-remove-form"' in page.data
    assert b'aria-label="Remove NEAT annotation"' in page.data
    assert b"data-roster-shift-select" in page.data
    assert b"data-roster-shift-open" not in page.data
    assert b"annotation-code--editable" in page.data
    assert b"Click to add annotation text" in page.data
    assert b'<svg viewBox="0 0 16 16"' not in page.data
    assert b"Save text" in page.data
    assert b"Current: NEAT" not in page.data

    updated = secured_client.post(
        endpoint,
        data={
            "_csrf_token": token,
            "annotation": "NEAT",
            "annotation_detail_update": "1",
            "annotation_note": "Cover requested by tower",
        },
    )
    assert updated.status_code == 302
    with app.app.app_context():
        assignment = Assignment.query.filter_by(
            unit_id=1, staff_id=target_id, day=request_day()
        ).one()
        assert assignment.annotation_note == "Cover requested by tower"
        assert assignment.note != "Cover requested by tower"
        audit = AnnotationAudit.query.filter_by(
            unit_id=1,
            assignment_id=assignment.id,
            action="detail_updated",
        ).one()
        assert audit.old_value == ""
        assert audit.new_value == "Cover requested by tower"

    page = secured_client.get(f"/roster/{request_day():%Y-%m}")
    assert b'title="Cover requested by tower \xe2\x80\x94 click to edit"' in page.data
    assert b"annotation-display--has-detail" in page.data
    assert page.data.count(b"NEAT\n      </button>") == 1
    assert b">Cover requested by tower</textarea>" in page.data


def test_bulk_annotation_feature_is_removed(secured_client):
    token = login(secured_client, "admin-a")
    response = secured_client.post("/annotations/bulk", data={
        "_csrf_token": token,
        "action": "preview"
    })
    assert response.status_code == 404


def test_watch_manager_needs_explicit_annotation_permission(secured_client):
    with app.app.app_context():
        watch = db.session.get(Watch, 1)
        manager = Staff(
            unit_id=1, username="wm-limited", name="Limited WM",
            staff_no="WM-LIMIT", role="user", watch_id=watch.id, is_wm=True,
            permissions_json='{"edit_roster": true}',
        )
        manager.set_password("password123")
        definition = AnnotationType(
            unit_id=1, code="WMTEST", label="Manager test", is_active=True,
        )
        target = Staff.query.filter_by(username="user-a").one()
        db.session.add_all([manager, definition])
        db.session.flush()
        identity = app.PlatformIdentity(
            public_id="test-wm-limited",
            username=manager.username,
            password_hash=manager.password_hash,
        )
        db.session.add(identity)
        db.session.flush()
        db.session.add(app.UnitMembership(
            identity_id=identity.id,
            unit_id=manager.unit_id,
            person_id=manager.id,
            role="StaffUser",
            status="active",
        ))
        db.session.commit()
        target_id = target.id
        refresh_annotation_cache()
    token = login(secured_client, "wm-limited")
    response = secured_client.post(
        f"/assign/{target_id}/{request_day():%Y-%m}/{request_day():%Y-%m-%d}",
        data={"_csrf_token": token, "annotation": "WMTEST"},
    )
    assert response.status_code == 403
