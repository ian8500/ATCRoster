from datetime import date, time

import pytest

import app
from app import (
    Assignment,
    DatabaseRoutingMetadata,
    PlatformIdentity,
    ShiftRequest,
    ShiftType,
    Staff,
    Unit,
    UnitMembership,
    Watch,
    db,
)
from tenancy import (
    bind_authenticated_unit,
    bind_platform_control,
    dispose_operational_engines,
    operational_engine_for_authenticated_unit,
    reset_authenticated_unit,
    reset_platform_control,
)


def _operational_tables():
    return [
        table for table in db.metadata.sorted_tables
        if table.name in app.OPERATIONAL_TABLE_NAMES
    ]


def _seed_operational_unit(
    unit_id, secret_name, username, marker, create_schema=True
):
    token = bind_authenticated_unit(unit_id, secret_name)
    try:
        engine = operational_engine_for_authenticated_unit()
        if create_schema:
            db.metadata.create_all(
                bind=engine, tables=_operational_tables()
            )
        watch = Watch(unit_id=unit_id, name=f"Watch {marker}")
        db.session.add(watch)
        db.session.flush()
        person = Staff(
            unit_id=unit_id, username=username, name=f"Controller {marker}",
            staff_no=f"{marker}-1", role="user", watch_id=watch.id,
        )
        person.set_password("Physical-Test-2026!")
        db.session.add(person)
        db.session.flush()
        db.session.add(ShiftType(
            unit_id=unit_id, code="M", name="Morning",
            start_time=time(7), end_time=time(15), is_working=True,
            is_active=True, is_requestable=True,
        ))
        db.session.add(ShiftRequest(
            unit_id=unit_id, staff_id=person.id,
            day=date(2026, 10, 10), code="M",
            requester_comment=f"{marker}-ONLY",
        ))
        db.session.commit()
        return person.id, person.password_hash, str(engine.url)
    finally:
        db.session.remove()
        reset_authenticated_unit(token)


def test_authenticated_airports_use_physically_distinct_databases(
    tmp_path, monkeypatch
):
    dispose_operational_engines()
    secret_a = "TEST_UNIT_A_DATABASE_URL"
    secret_b = "TEST_UNIT_B_DATABASE_URL"
    monkeypatch.setenv(secret_a, f"sqlite:///{tmp_path / 'unit-a.db'}")
    monkeypatch.setenv(secret_b, f"sqlite:///{tmp_path / 'unit-b.db'}")
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        db.session.add_all([
            Unit(id=1, code="AAA", name="Airport A"),
            Unit(id=2, code="BBB", name="Airport B"),
        ])
        db.session.commit()
        person_a, password_a, url_a = _seed_operational_unit(
            1, secret_a, "physical-a", "AIRPORT-A"
        )
        person_b, password_b, url_b = _seed_operational_unit(
            2, secret_b, "physical-b", "AIRPORT-B"
        )
        assert url_a != url_b
        identity_a = PlatformIdentity(
            public_id="physical-a", username="physical-a",
            password_hash=password_a,
        )
        identity_b = PlatformIdentity(
            public_id="physical-b", username="physical-b",
            password_hash=password_b,
        )
        db.session.add_all([identity_a, identity_b])
        db.session.flush()
        db.session.add_all([
            UnitMembership(
                identity_id=identity_a.id, unit_id=1, person_id=person_a,
                role="StaffUser", status="active",
            ),
            UnitMembership(
                identity_id=identity_b.id, unit_id=2, person_id=person_b,
                role="StaffUser", status="active",
            ),
            DatabaseRoutingMetadata(unit_id=1, secret_name=secret_a),
            DatabaseRoutingMetadata(unit_id=2, secret_name=secret_b),
        ])
        db.session.commit()

    client_a = app.app.test_client()
    client_b = app.app.test_client()
    client_a.get("/login")
    client_b.get("/login")
    with client_a.session_transaction() as session:
        token_a = session["_csrf_token"]
    with client_b.session_transaction() as session:
        token_b = session["_csrf_token"]
    assert client_a.post("/login", data={
        "_csrf_token": token_a,
        "username": "physical-a", "password": "Physical-Test-2026!",
    }).status_code == 302
    assert client_b.post("/login", data={
        "_csrf_token": token_b,
        "username": "physical-b", "password": "Physical-Test-2026!",
    }).status_code == 302
    page_a = client_a.get("/requests")
    page_b = client_b.get("/requests")
    assert b"AIRPORT-A-ONLY" in page_a.data
    assert b"AIRPORT-B-ONLY" not in page_a.data
    assert b"AIRPORT-B-ONLY" in page_b.data
    assert b"AIRPORT-A-ONLY" not in page_b.data

    with app.app.app_context():
        platform_token = bind_platform_control()
        try:
            with pytest.raises(PermissionError):
                Assignment.query.count()
        finally:
            reset_platform_control(platform_token)
        db.session.remove()
        db.drop_all()
    dispose_operational_engines()
