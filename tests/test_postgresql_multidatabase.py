"""PostgreSQL-only proof of the physical control/airport boundary."""

import os
import hashlib
import threading
from pathlib import Path

import pytest
import psycopg
from psycopg import sql
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

CONTROL_URL = os.environ.get("ATCROSTER_TEST_CONTROL_DATABASE_URL")
AIRPORT_A_URL = os.environ.get("ATCROSTER_TEST_A_DATABASE_URL")
AIRPORT_B_URL = os.environ.get("ATCROSTER_TEST_B_DATABASE_URL")
RESTORE_URL = os.environ.get("ATCROSTER_TEST_RESTORE_DATABASE_URL")

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
from scripts.database_backup import create_backup, restore_backup  # noqa: E402
from scripts.database_grants import (  # noqa: E402
    apply_runtime_grants,
    verify_runtime_grants,
)
import platform_provisioning  # noqa: E402
from platform_provisioning import ProvisioningWorker  # noqa: E402
from tenancy import dispose_operational_engines, operational_unit_context  # noqa: E402
from toil_service import apply_toil_transaction  # noqa: E402
from tests.test_physical_database_isolation import (  # noqa: E402
    _seed_operational_unit,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _reset_postgres(url):
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


def _upgrade_to(url, revision, role="operational"):
    config = Config(str(REPOSITORY / "alembic.ini"))
    config.set_main_option("script_location", str(REPOSITORY / "migrations"))
    previous_url = os.environ.get("DATABASE_URL")
    previous_role = os.environ.get("ATCROSTER_SCHEMA_ROLE")
    os.environ["DATABASE_URL"] = url
    os.environ["ATCROSTER_SCHEMA_ROLE"] = role
    try:
        command.upgrade(config, revision)
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
        if previous_role is None:
            os.environ.pop("ATCROSTER_SCHEMA_ROLE", None)
        else:
            os.environ["ATCROSTER_SCHEMA_ROLE"] = previous_role


def test_postgresql_control_and_two_airport_databases_are_isolated(
    monkeypatch,
):
    dispose_operational_engines()
    for url in (CONTROL_URL, AIRPORT_A_URL, AIRPORT_B_URL):
        _reset_postgres(url)
    assert upgrade_database(CONTROL_URL, "control") == "20260801_37"
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260801_37"
    assert upgrade_database(AIRPORT_B_URL, "operational") == "20260801_37"
    secret_a = "ATCROSTER_UNIT_1_DATABASE_URL"
    secret_b = "ATCROSTER_UNIT_2_DATABASE_URL"
    monkeypatch.setenv(secret_a, AIRPORT_A_URL)
    monkeypatch.setenv(secret_b, AIRPORT_B_URL)
    with app.app.app_context():
        db.session.add_all(
            [
                Unit(id=1, code="PGA", name="Postgres Airport A"),
                Unit(id=2, code="PGB", name="Postgres Airport B"),
            ]
        )
        db.session.commit()
        person_a, password_a, _ = _seed_operational_unit(
            1,
            secret_a,
            "postgres-a",
            "POSTGRES-A",
            create_schema=False,
        )
        person_b, password_b, _ = _seed_operational_unit(
            2,
            secret_b,
            "postgres-b",
            "POSTGRES-B",
            create_schema=False,
        )
        identity_a = PlatformIdentity(
            public_id="postgres-a",
            username="postgres-a",
            password_hash=password_a,
        )
        identity_b = PlatformIdentity(
            public_id="postgres-b",
            username="postgres-b",
            password_hash=password_b,
        )
        db.session.add_all([identity_a, identity_b])
        db.session.flush()
        db.session.add_all(
            [
                UnitMembership(
                    identity_id=identity_a.id,
                    unit_id=1,
                    person_id=person_a,
                    role="StaffUser",
                    status="active",
                ),
                UnitMembership(
                    identity_id=identity_b.id,
                    unit_id=2,
                    person_id=person_b,
                    role="StaffUser",
                    status="active",
                ),
                DatabaseRoutingMetadata(
                    unit_id=1,
                    secret_name=secret_a,
                    provisioning_state="active",
                ),
                DatabaseRoutingMetadata(
                    unit_id=2,
                    secret_name=secret_b,
                    provisioning_state="active",
                ),
            ]
        )
        db.session.commit()
    client_a = app.app.test_client()
    client_b = app.app.test_client()
    client_a.get("/login")
    client_b.get("/login")
    with client_a.session_transaction() as session:
        token_a = session["_csrf_token"]
    with client_b.session_transaction() as session:
        token_b = session["_csrf_token"]
    assert (
        client_a.post(
            "/login",
            data={
                "_csrf_token": token_a,
                "username": "postgres-a",
                "password": "Physical-Test-2026!",
            },
        ).status_code
        == 302
    )
    assert (
        client_b.post(
            "/login",
            data={
                "_csrf_token": token_b,
                "username": "postgres-b",
                "password": "Physical-Test-2026!",
            },
        ).status_code
        == 302
    )
    page_a = client_a.get("/requests")
    page_b = client_b.get("/requests")
    assert b"POSTGRES-A-ONLY" in page_a.data
    assert b"POSTGRES-B-ONLY" not in page_a.data
    assert b"POSTGRES-B-ONLY" in page_b.data
    assert b"POSTGRES-A-ONLY" not in page_b.data
    control_tables = set(inspect(create_engine(CONTROL_URL)).get_table_names())
    airport_a_tables = set(inspect(create_engine(AIRPORT_A_URL)).get_table_names())
    airport_b_tables = set(inspect(create_engine(AIRPORT_B_URL)).get_table_names())
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
        "training_level",
        "training_objective",
        "training_session",
        "training_score",
    ):
        assert table in airport_a_tables and table in airport_b_tables
    assert "platform_identity" not in airport_a_tables
    assert "unit" not in airport_a_tables
    assert "special_requirement" in app.OPERATIONAL_TABLE_NAMES
    assert "sms_audit" in app.OPERATIONAL_TABLE_NAMES
    for table in (
        "training_level",
        "training_objective",
        "training_session",
        "training_score",
    ):
        assert table in app.OPERATIONAL_TABLE_NAMES

    # Two independent workers use separate PostgreSQL sessions. The second
    # cannot claim or migrate the airport while the first owns its lease.
    with app.app.app_context():
        job = ProvisioningJob(
            unit_id=1,
            idempotency_key=hashlib.sha256(b"postgres-concurrency").hexdigest(),
            state="queued",
            active_key="active",
            next_attempt_at=app.utcnow(),
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
            idempotency_key=hashlib.sha256(b"postgres-concurrency").hexdigest()
        ).one()
        assert completed.state == "completed"
    dispose_operational_engines()


@pytest.mark.skipif(not RESTORE_URL, reason="restore test database URL is required")
def test_generated_postgresql_backup_restores_and_preserves_key_records(tmp_path):
    _reset_postgres(AIRPORT_A_URL)
    _reset_postgres(RESTORE_URL)
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260801_37"
    source = create_engine(AIRPORT_A_URL)
    with source.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE recovery_probe "
                "(id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
            )
        )
        connection.execute(
            text("INSERT INTO recovery_probe (id, value) VALUES (1, 'verified')")
        )
    source.dispose()
    archive, metadata = create_backup(
        AIRPORT_A_URL, tmp_path, "airport-test", "operational"
    )
    result = restore_backup(archive, metadata, RESTORE_URL)
    assert result.alembic_revision == "20260801_37"
    restored = create_engine(RESTORE_URL)
    try:
        with restored.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT value FROM recovery_probe WHERE id=1")
                ).scalar_one()
                == "verified"
            )
    finally:
        restored.dispose()


def _insert_staff(connection, unit_id, username):
    return connection.execute(
        text(
            "INSERT INTO staff "
            "(unit_id, username, password_hash, role, membership_status, "
            "permissions_json, name, staff_no) VALUES "
            "(:unit_id, :username, 'not-a-login-hash', 'user', 'active', "
            "'{}', :username, :staff_no) RETURNING id"
        ),
        {
            "unit_id": unit_id,
            "username": username,
            "staff_no": f"STAFF-{unit_id}",
        },
    ).scalar_one()


def test_postgresql_rejects_cross_unit_operational_relationships():
    _reset_postgres(AIRPORT_A_URL)
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260801_37"
    engine = create_engine(AIRPORT_A_URL)
    try:
        with engine.begin() as connection:
            staff_a = _insert_staff(connection, 1, "integrity-a")
            staff_b = _insert_staff(connection, 2, "integrity-b")
            connection.execute(
                text(
                    "INSERT INTO assignment (unit_id, staff_id, day, code) "
                    "VALUES (1, :staff_id, DATE '2026-08-01', 'M'), "
                    "(2, :staff_b, DATE '2026-08-01', 'A')"
                ),
                {"staff_id": staff_a, "staff_b": staff_b},
            )
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO assignment "
                        "(unit_id, staff_id, day, code) "
                        "VALUES (2, :staff_id, DATE '2026-08-02', 'M')"
                    ),
                    {"staff_id": staff_a},
                )
    finally:
        engine.dispose()


def test_tenant_integrity_migration_refuses_inconsistent_legacy_data():
    _reset_postgres(AIRPORT_A_URL)
    _upgrade_to(AIRPORT_A_URL, "20260801_34")
    engine = create_engine(AIRPORT_A_URL)
    try:
        with engine.begin() as connection:
            staff_a = _insert_staff(connection, 1, "legacy-a")
            connection.execute(
                text(
                    "INSERT INTO assignment (unit_id, staff_id, day, code) "
                    "VALUES (2, :staff_id, DATE '2026-08-01', 'M')"
                ),
                {"staff_id": staff_a},
            )
        with pytest.raises(RuntimeError, match="cross-unit or orphaned rows"):
            _upgrade_to(AIRPORT_A_URL, "head")
    finally:
        engine.dispose()


def test_postgresql_concurrent_toil_retry_changes_balance_once(monkeypatch):
    _reset_postgres(AIRPORT_A_URL)
    assert upgrade_database(AIRPORT_A_URL, "operational") == "20260801_37"
    engine = create_engine(AIRPORT_A_URL)
    with engine.begin() as connection:
        person_id = _insert_staff(connection, 1, "toil-concurrency")
    engine.dispose()
    secret_name = "ATCROSTER_TEST_TOIL_DATABASE_URL"
    monkeypatch.setenv(secret_name, AIRPORT_A_URL)
    dispose_operational_engines()
    barrier = threading.Barrier(2)
    row_ids = []
    errors = []

    def apply_once():
        try:
            with app.app.app_context(), operational_unit_context(1, secret_name):
                barrier.wait(timeout=10)
                row = apply_toil_transaction(
                    app.db,
                    app.Staff,
                    app.ToilTransaction,
                    unit_id=1,
                    person_id=person_id,
                    delta_half_days=2,
                    reason="Concurrent integration test",
                    actor_id=person_id,
                    utcnow=app.utcnow,
                    transaction_key="same-toil-retry",
                    source_type="integration_test",
                )
                app.db.session.commit()
                row_ids.append(row.id)
                app.db.session.remove()
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=apply_once)
    second = threading.Thread(target=apply_once)
    first.start()
    second.start()
    first.join(timeout=20)
    second.join(timeout=20)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(set(row_ids)) == 1
    check = create_engine(AIRPORT_A_URL)
    try:
        with check.connect() as connection:
            assert connection.execute(
                text("SELECT toil_half_days FROM staff WHERE id=:id"),
                {"id": person_id},
            ).scalar_one() == 2
            assert connection.execute(
                text(
                    "SELECT count(*) FROM toil_transaction "
                    "WHERE unit_id=1 AND transaction_key='same-toil-retry'"
                )
            ).scalar_one() == 1
    finally:
        check.dispose()
        dispose_operational_engines()


def test_postgresql_runtime_role_cannot_mutate_audit_evidence():
    _reset_postgres(AIRPORT_B_URL)
    assert upgrade_database(AIRPORT_B_URL, "operational") == "20260801_37"
    role = f"atcroster_runtime_{os.getpid()}"
    password = "runtime-integration-only"
    owner_dsn = str(
        make_url(AIRPORT_B_URL)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )
    with psycopg.connect(owner_dsn, autocommit=True) as owner:
        owner.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
    try:
        apply_runtime_grants(AIRPORT_B_URL, role)
        result = verify_runtime_grants(AIRPORT_B_URL, role)
        assert result.audit_tables_checked > 0
        runtime_url = str(
            make_url(AIRPORT_B_URL)
            .set(drivername="postgresql", username=role, password=password)
            .render_as_string(hide_password=False)
        )
        with psycopg.connect(runtime_url) as runtime:
            staff_id = runtime.execute(
                "INSERT INTO staff "
                "(unit_id, username, password_hash, role, membership_status, "
                "permissions_json, name, staff_no) VALUES "
                "(1, 'grant-user', 'not-a-login-hash', 'user', 'active', "
                "'{}', 'Grant User', 'GRANT-1') RETURNING id"
            ).fetchone()[0]
            runtime.execute(
                "INSERT INTO change_log "
                '(unit_id, "when", entity_type, entity_id, field) '
                "VALUES (1, CURRENT_TIMESTAMP, 'staff', %s, 'name')",
                (staff_id,),
            )
            assert runtime.execute("SELECT count(*) FROM change_log").fetchone()[0] == 1
            runtime.execute(
                "UPDATE staff SET name = 'Updated Grant User' WHERE id = %s",
                (staff_id,),
            )
            runtime.commit()
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                runtime.execute("UPDATE change_log SET field = 'tampered'")
            runtime.rollback()
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                runtime.execute("DELETE FROM change_log")
            runtime.rollback()
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                runtime.execute("CREATE TABLE forbidden_migration (id integer)")
            runtime.rollback()
    finally:
        with psycopg.connect(owner_dsn, autocommit=True) as owner:
            owner.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            owner.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
