import os
import sys
import tempfile
from datetime import time

import pytest

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

        watch_a = Watch(name="Watch A", order_index=1)
        db.session.add(watch_a)

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


def test_login_page_loads(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Login" in resp.data


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
