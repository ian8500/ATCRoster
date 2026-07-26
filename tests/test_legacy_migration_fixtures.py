import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

REPOSITORY = Path(__file__).resolve().parents[1]


FIXTURES = {
    "clean": "",
    "minimal": """
        CREATE TABLE staff (
          id INTEGER PRIMARY KEY, username VARCHAR(80), password_hash VARCHAR(200),
          name VARCHAR(80), staff_no VARCHAR(20)
        );
        INSERT INTO staff VALUES (1, 'legacy', 'hash', 'Legacy Person', 'L1');
    """,
    "full_original": """
        CREATE TABLE watch (id INTEGER PRIMARY KEY, name VARCHAR(32), order_index INTEGER);
        CREATE TABLE staff (
          id INTEGER PRIMARY KEY, username VARCHAR(80), password_hash VARCHAR(200),
          name VARCHAR(80), staff_no VARCHAR(20), watch_id INTEGER,
          medical_expiry DATE, tower_ut BOOLEAN, radar_ut BOOLEAN,
          tower_ue_expiry DATE, radar_ue_expiry DATE
        );
        CREATE TABLE shift_type (
          id INTEGER PRIMARY KEY, code VARCHAR(10), name VARCHAR(40),
          start_time TIME, end_time TIME, is_working BOOLEAN
        );
        CREATE TABLE assignment (
          id INTEGER PRIMARY KEY, staff_id INTEGER, day DATE, code VARCHAR(10),
          source VARCHAR(10), note VARCHAR(140), annotation VARCHAR(20)
        );
        CREATE TABLE shift_request (
          id INTEGER PRIMARY KEY, staff_id INTEGER, day DATE, code VARCHAR(10),
          status VARCHAR(20), admin_response TEXT, responded_by_id INTEGER,
          responded_at DATETIME
        );
        INSERT INTO watch VALUES (1, 'A', 1);
        INSERT INTO staff VALUES (
          1, 'legacy', 'hash', 'Legacy Person', 'L1', 1,
          '2030-01-01', 1, 0, '2030-01-01', NULL
        );
        INSERT INTO shift_type VALUES (1, 'M', 'Morning', '07:00', '15:00', 1);
        INSERT INTO assignment VALUES (1, 1, '2025-01-01', 'M', 'manual', '', '');
        INSERT INTO shift_request VALUES (
          1, 1, '2025-01-02', 'M', 'approved', '', NULL, NULL
        );
    """,
    "partial": """
        CREATE TABLE unit (
          id INTEGER PRIMARY KEY, code VARCHAR(12), name VARCHAR(120),
          created_at DATETIME
        );
        INSERT INTO unit VALUES (1, 'FIRST', 'First unit', CURRENT_TIMESTAMP);
        CREATE TABLE staff (
          id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL DEFAULT 1,
          username VARCHAR(80), password_hash VARCHAR(200),
          name VARCHAR(80), staff_no VARCHAR(20)
        );
        INSERT INTO staff VALUES (1, 1, 'partial', 'hash', 'Partial', 'P1');
    """,
    "historical": """
        CREATE TABLE staff (
          id INTEGER PRIMARY KEY, username VARCHAR(80), password_hash VARCHAR(200),
          name VARCHAR(80), staff_no VARCHAR(20), medical_expiry DATE
        );
        CREATE TABLE assignment (
          id INTEGER PRIMARY KEY, staff_id INTEGER, day DATE, code VARCHAR(10)
        );
        CREATE TABLE shift_request (
          id INTEGER PRIMARY KEY, staff_id INTEGER, day DATE, code VARCHAR(10),
          status VARCHAR(20)
        );
        INSERT INTO staff VALUES (7, 'history', 'hash', 'Historical', 'H7', '2029-06-30');
        INSERT INTO assignment VALUES (11, 7, '2024-12-31', 'N');
        INSERT INTO shift_request VALUES (13, 7, '2025-01-03', 'N', 'approved');
    """,
}


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_legacy_fixture_upgrades_to_head_without_data_loss(
    fixture_name, tmp_path
):
    database = tmp_path / f"{fixture_name}.db"
    if FIXTURES[fixture_name]:
        connection = sqlite3.connect(database)
        connection.executescript(FIXTURES[fixture_name])
        connection.commit()
        connection.close()
    before = {}
    engine = create_engine(f"sqlite:///{database}")
    with engine.connect() as connection:
        tables = set(inspect(connection).get_table_names())
        for table in ("staff", "assignment", "shift_request"):
            if table in tables:
                before[table] = connection.execute(
                    text(f'SELECT COUNT(*) FROM "{table}"')
                ).scalar_one()
    environment = os.environ.copy()
    environment.update({
        "DATABASE_URL": f"sqlite:///{database}",
        "FLASK_SECRET_KEY": "migration-fixture-secret-2026-long",
        "ATCROSTER_SKIP_BOOTSTRAP": "true",
        "ATCROSTER_SKIP_RUNTIME_SCHEMA": "1",
    })
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPOSITORY, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with engine.connect() as connection:
        inspector = inspect(connection)
        assert connection.execute(
            text("SELECT version_num FROM alembic_version")
            ).scalar_one() == "20260726_12"
        for table, count in before.items():
            assert connection.execute(
                text(f'SELECT COUNT(*) FROM "{table}"')
            ).scalar_one() == count
            assert "unit_id" in {
                column["name"] for column in inspector.get_columns(table)
            }
            assert any(
                key.get("constrained_columns") == ["unit_id"]
                and key.get("referred_table") == "unit"
                for key in inspector.get_foreign_keys(table)
            )
        if fixture_name in {"full_original", "historical"}:
            assert connection.execute(
                text("SELECT COUNT(*) FROM person_qualification")
            ).scalar_one() >= 1
        scoped = {
            "staff": ("unit_id", "staff_no"),
            "assignment": ("unit_id", "staff_id", "day"),
            "shift_request": ("unit_id", "staff_id", "day"),
            "shift_type": ("unit_id", "code"),
            "watch": ("unit_id", "name"),
        }
        available = set(inspector.get_table_names())
        for table, expected_columns in scoped.items():
            if table not in available:
                continue
            constraints = inspector.get_unique_constraints(table)
            assert expected_columns in {
                tuple(item.get("column_names") or ())
                for item in constraints
            }
