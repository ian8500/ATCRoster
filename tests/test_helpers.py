import os
import sys
import tempfile
from datetime import date, time

import pytest


# Ensure the repository root (where app.py lives) is on the import path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Use a temporary SQLite database for these tests **before** importing the app
TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "atc_roster_test.db")
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import app  # noqa: E402  (import after setting DATABASE_URL)
from app import (
    _context_month_for_date,
    _is_empty_like,
    _is_working_code_prefix,
    _is_working_day_code,
    _is_working_m_code,
    _is_working_n_code,
    _month_add,
    _parse_date,
    _parse_hhmm,
    bootstrap_reference_data,
    parse_annotation,
    is_month_locked,
    lock_date_for_month,
    refresh_shift_cache,
    db,
    ShiftRequest,
    ShiftType,
    Staff,
)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Create a clean database for the duration of the module."""
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        bootstrap_reference_data()
    yield
    with app.app.app_context():
        db.session.remove()
        db.drop_all()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def test_parse_hhmm():
    assert _parse_hhmm("07:30") == time(7, 30)
    assert _parse_hhmm("  19:05  ") == time(19, 5)
    assert _parse_hhmm("not-a-time") is None
    assert _parse_hhmm("") is None


def test_parse_date():
    assert _parse_date("2025-04-12") == date(2025, 4, 12)
    assert _parse_date(" 2024-01-01 ") == date(2024, 1, 1)
    assert _parse_date("2024/01/01") is None
    assert _parse_date("") is None


def test_parse_annotation():
    with app.app.app_context():
        assert parse_annotation("a6m") == {"type": "A6", "suffix": "M"}
        assert parse_annotation(" TOAI ") == {"type": "TOAI", "suffix": None}
        assert parse_annotation("invalid") is None
        assert parse_annotation("") is None


def test_context_month_for_date():
    assert _context_month_for_date(date(2025, 4, 3)) == "2025-04"
    assert _context_month_for_date(None) is None


def test_month_add_and_lock_logic():
    assert _month_add(2025, 4, -1) == (2025, 3)
    assert _month_add(2025, 1, -1) == (2024, 12)
    lock_dt = lock_date_for_month(2025, 4)
    assert lock_dt == date(2025, 2, 20)
    assert is_month_locked(2025, 4, today=date(2025, 2, 20)) is True
    assert is_month_locked(2025, 4, today=date(2025, 2, 19)) is False


def test_is_empty_like():
    assert _is_empty_like("-")
    assert _is_empty_like("—")
    assert _is_empty_like("")
    assert not _is_empty_like("OFF")


def test_is_working_code_prefix_with_shift_records():
    with app.app.app_context():
        db.session.add(ShiftType(code="D10", name="Day", is_working=True))
        db.session.add(ShiftType(code="N00", name="Night", is_working=False))
        db.session.commit()
        refresh_shift_cache()

        assert _is_working_code_prefix("D10", "D") is True
        assert _is_working_day_code("D10") is True
        # Non-working codes should be filtered regardless of prefix
        assert _is_working_code_prefix("OFF", "D") is False
        assert _is_working_code_prefix("N00", "N") is False
        assert _is_working_n_code("N00") is False
        # Unknown code falls back to prefix-only matching
        assert _is_working_m_code("M123") is True


def test_assign_cell_keeps_request_record_and_marks_closed():
    req_day = date(2025, 7, 1)

    with app.app.app_context():
        # Ensure the shift code we will use exists and is cached.
        if not ShiftType.query.filter_by(code="M").first():
            db.session.add(ShiftType(code="M", name="Morning", is_working=True))
            db.session.commit()
        refresh_shift_cache()

        admin = Staff(
            username="admin_req_test",
            name="Admin Request Tester",
            staff_no="ADM-REQ-1",
            role="admin",
        )
        admin.set_password("password123")

        requester = Staff(
            username="user_req_test",
            name="Requester",
            staff_no="REQ-USER-1",
        )
        requester.set_password("userpass123")

        db.session.add_all([admin, requester])
        db.session.commit()

        db.session.add(ShiftRequest(staff_id=requester.id, day=req_day, code="M"))
        db.session.commit()

    client = app.app.test_client()
    login_resp = client.post(
        "/login",
        data={"username": "admin_req_test", "password": "password123"},
        follow_redirects=True,
    )
    assert login_resp.status_code == 200

    assign_resp = client.post(
        f"/assign/{requester.id}/2025-07/{req_day.isoformat()}",
        data={"code": "M"},
        follow_redirects=False,
    )
    assert assign_resp.status_code == 302

    with app.app.app_context():
        req = ShiftRequest.query.filter_by(staff_id=requester.id, day=req_day).one()
        # The request remains but is auto-closed because it had not been actioned.
        assert req.status == "closed"
        assert req.responded_by_id == admin.id
        assert req.responded_at is not None

