"""PostgreSQL-only proof of the physical control/airport boundary."""
import os

import pytest
from sqlalchemy import create_engine, inspect, text

CONTROL_URL = os.environ.get("ATCROSTER_TEST_CONTROL_DATABASE_URL")
AIRPORT_A_URL = os.environ.get("ATCROSTER_TEST_A_DATABASE_URL")
AIRPORT_B_URL = os.environ.get("ATCROSTER_TEST_B_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not all((CONTROL_URL, AIRPORT_A_URL, AIRPORT_B_URL)),
    reason="three PostgreSQL integration database URLs are required",
)

if CONTROL_URL:
    os.environ["DATABASE_URL"] = CONTROL_URL
    os.environ["CONTROL_DATABASE_URL"] = CONTROL_URL
    os.environ["ATCROSTER_SKIP_RUNTIME_SCHEMA"] = "1"

import app  # noqa: E402
from app import (  # noqa: E402
    DatabaseRoutingMetadata,
    PlatformIdentity,
    Unit,
    UnitMembership,
    db,
)
from scripts.migrate_all_databases import upgrade_database  # noqa: E402
from tenancy import dispose_operational_engines  # noqa: E402
from tests.test_physical_database_isolation import (  # noqa: E402
    _seed_operational_unit,
)


def _reset_postgres(url):
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


def test_postgresql_control_and_two_airport_databases_are_isolated(
    monkeypatch,
):
    dispose_operational_engines()
    for url in (CONTROL_URL, AIRPORT_A_URL, AIRPORT_B_URL):
        _reset_postgres(url)
    assert upgrade_database(CONTROL_URL, "control") == "20260726_09"
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260726_09"
    assert upgrade_database(AIRPORT_B_URL, "operational") == "20260726_09"
    secret_a = "ATCROSTER_UNIT_1_DATABASE_URL"
    secret_b = "ATCROSTER_UNIT_2_DATABASE_URL"
    monkeypatch.setenv(secret_a, AIRPORT_A_URL)
    monkeypatch.setenv(secret_b, AIRPORT_B_URL)
    with app.app.app_context():
        db.session.add_all([
            Unit(id=1, code="PGA", name="Postgres Airport A"),
            Unit(id=2, code="PGB", name="Postgres Airport B"),
        ])
        db.session.commit()
        person_a, password_a, _ = _seed_operational_unit(
            1, secret_a, "postgres-a", "POSTGRES-A",
            create_schema=False,
        )
        person_b, password_b, _ = _seed_operational_unit(
            2, secret_b, "postgres-b", "POSTGRES-B",
            create_schema=False,
        )
        identity_a = PlatformIdentity(
            public_id="postgres-a", username="postgres-a",
            password_hash=password_a,
        )
        identity_b = PlatformIdentity(
            public_id="postgres-b", username="postgres-b",
            password_hash=password_b,
        )
        db.session.add_all([identity_a, identity_b])
        db.session.flush()
        db.session.add_all([
            UnitMembership(
                identity_id=identity_a.id, unit_id=1,
                person_id=person_a, role="StaffUser", status="active",
            ),
            UnitMembership(
                identity_id=identity_b.id, unit_id=2,
                person_id=person_b, role="StaffUser", status="active",
            ),
            DatabaseRoutingMetadata(
                unit_id=1, secret_name=secret_a,
                provisioning_state="active",
            ),
            DatabaseRoutingMetadata(
                unit_id=2, secret_name=secret_b,
                provisioning_state="active",
            ),
        ])
        db.session.commit()
    client_a = app.app.test_client()
    client_b = app.app.test_client()
    assert client_a.post("/login", data={
        "username": "postgres-a", "password": "Physical-Test-2026!",
    }).status_code == 302
    assert client_b.post("/login", data={
        "username": "postgres-b", "password": "Physical-Test-2026!",
    }).status_code == 302
    page_a = client_a.get("/requests")
    page_b = client_b.get("/requests")
    assert b"POSTGRES-A-ONLY" in page_a.data
    assert b"POSTGRES-B-ONLY" not in page_a.data
    assert b"POSTGRES-B-ONLY" in page_b.data
    assert b"POSTGRES-A-ONLY" not in page_b.data
    control_tables = set(inspect(create_engine(CONTROL_URL)).get_table_names())
    airport_a_tables = set(
        inspect(create_engine(AIRPORT_A_URL)).get_table_names()
    )
    airport_b_tables = set(
        inspect(create_engine(AIRPORT_B_URL)).get_table_names()
    )
    assert "platform_identity" in control_tables
    assert "staff" not in control_tables
    assert "staff" in airport_a_tables and "staff" in airport_b_tables
    assert "platform_identity" not in airport_a_tables
    assert "unit" not in airport_a_tables
    dispose_operational_engines()
