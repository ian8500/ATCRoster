"""PostgreSQL-only proof of the physical control/airport boundary."""
import os
import hashlib
import threading

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
    ProvisioningJob,
    Unit,
    UnitMembership,
    db,
)
from scripts.migrate_all_databases import upgrade_database  # noqa: E402
import platform_provisioning  # noqa: E402
from platform_provisioning import ProvisioningWorker  # noqa: E402
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
    assert upgrade_database(CONTROL_URL, "control") == "20260731_32"
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260731_32"
    assert upgrade_database(AIRPORT_B_URL, "operational") == "20260731_32"
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
    client_a.get("/login")
    client_b.get("/login")
    with client_a.session_transaction() as session:
        token_a = session["_csrf_token"]
    with client_b.session_transaction() as session:
        token_b = session["_csrf_token"]
    assert client_a.post("/login", data={
        "_csrf_token": token_a,
        "username": "postgres-a", "password": "Physical-Test-2026!",
    }).status_code == 302
    assert client_b.post("/login", data={
        "_csrf_token": token_b,
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
    assert "special_requirement" not in control_tables
    assert "sms_audit" not in control_tables
    assert "staff" in airport_a_tables and "staff" in airport_b_tables
    assert (
        "special_requirement" in airport_a_tables
        and "special_requirement" in airport_b_tables
    )
    assert "sms_audit" in airport_a_tables and "sms_audit" in airport_b_tables
    for table in (
        "training_level", "training_objective",
        "training_session", "training_score",
    ):
        assert table in airport_a_tables and table in airport_b_tables
    assert "platform_identity" not in airport_a_tables
    assert "unit" not in airport_a_tables
    assert "special_requirement" in app.OPERATIONAL_TABLE_NAMES
    assert "sms_audit" in app.OPERATIONAL_TABLE_NAMES
    for table in (
        "training_level", "training_objective",
        "training_session", "training_score",
    ):
        assert table in app.OPERATIONAL_TABLE_NAMES

    # Two independent workers use separate PostgreSQL sessions. The second
    # cannot claim or migrate the airport while the first owns its lease.
    with app.app.app_context():
        job = ProvisioningJob(
            unit_id=1,
            idempotency_key=hashlib.sha256(b"postgres-concurrency").hexdigest(),
            state="queued", active_key="active", next_attempt_at=app.utcnow(),
        )
        db.session.add(job)
        db.session.commit()
    migration_started = threading.Event()
    release_migration = threading.Event()
    calls = []
    real_upgrade = platform_provisioning.upgrade_database

    def slow_upgrade(url, role):
        calls.append((url, role))
        migration_started.set()
        assert release_migration.wait(10)
        return real_upgrade(url, role)

    monkeypatch.setattr(platform_provisioning, "upgrade_database", slow_upgrade)
    first = ProvisioningWorker(app.app)
    second = ProvisioningWorker(app.app)
    thread = threading.Thread(target=first.run_once)
    thread.start()
    assert migration_started.wait(10)
    assert second.run_once() is False
    release_migration.set()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert len(calls) == 1
    with app.app.app_context():
        completed = ProvisioningJob.query.filter_by(
            idempotency_key=hashlib.sha256(
                b"postgres-concurrency"
            ).hexdigest()
        ).one()
        assert completed.state == "completed"
    dispose_operational_engines()
