from datetime import date, time

import pytest

import app
from app import (
    Assignment, Notification, RequestAudit, ShiftRequest, ShiftType, Staff,
    Unit, Watch, _lock_date_for_target_month, _request_date_bounds, db,
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
            ShiftType(unit_id=1, code="QUAL", name="Qualified", start_time=time(9), end_time=time(17),
                      is_working=True, is_active=True, is_requestable=True,
                      required_qualification="medical"),
            ShiftType(unit_id=2, code="REQ", name="Other requestable", is_working=True,
                      is_active=True, is_requestable=True),
        ]
        db.session.add_all([unit_a, unit_b, watch_a, watch_b, admin, user, other, *shifts])
        db.session.commit()
        refresh_shift_cache()
    client = app.app.test_client()
    yield client
    with app.app.app_context():
        db.session.remove()
        db.drop_all()


def login(client, username="user-a"):
    response = client.post("/login", data={"username": username, "password": "password123"})
    assert response.status_code == 302
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
        "_csrf_token": token, "action": "approve_apply", "admin_response": "Applied",
        "ym": f"{request_day():%Y-%m}", "confirm_override": "yes",
    })
    assert applied.status_code == 302
    with app.app.app_context():
        row = db.session.get(ShiftRequest, rid)
        assert row.status == "fulfilled"
        assert db.session.get(Assignment, row.resulting_assignment_id).code == "REQ"
        assert Notification.query.filter_by(recipient_id=row.staff_id, kind="shift_request_fulfilled").count() == 1


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
