import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from scripts.report_assignment_migration import classification_summary

REPOSITORY = Path(__file__).resolve().parents[1]


def _alembic_head_revision() -> str:
    config = Config(str(REPOSITORY / "alembic.ini"))
    config.set_main_option("script_location", str(REPOSITORY / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


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
def test_legacy_fixture_upgrades_to_head_without_data_loss(fixture_name, tmp_path):
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
    environment.update(
        {
            "DATABASE_URL": f"sqlite:///{database}",
            "FLASK_SECRET_KEY": "migration-fixture-secret-2026-long",
            "ATCROSTER_SKIP_BOOTSTRAP": "true",
            "ATCROSTER_SKIP_RUNTIME_SCHEMA": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with engine.connect() as connection:
        inspector = inspect(connection)
        if "staff" in inspector.get_table_names() and "shift_type" in inspector.get_table_names():
            assert {
                "work_pattern",
                "work_pattern_day",
                "work_pattern_day_allowed_shift",
                "staff_pattern_assignment",
                "staff_rule",
            } <= set(inspector.get_table_names())
        assert (
            connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
                == _alembic_head_revision()
        )
        if "unit" in inspector.get_table_names():
            unit_columns = {
                column["name"] for column in inspector.get_columns("unit")
            }
            assert "protected_roster_months_ahead" in unit_columns
            assert "preserve_redundant_overrides" in unit_columns
        for table, count in before.items():
            assert (
                connection.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
                == count
            )
            assert "unit_id" in {
                column["name"] for column in inspector.get_columns(table)
            }
            assert any(
                key.get("constrained_columns") == ["unit_id"]
                and key.get("referred_table") == "unit"
                for key in inspector.get_foreign_keys(table)
            )
        if fixture_name in {"full_original", "historical"}:
            assert (
                connection.execute(
                    text("SELECT COUNT(*) FROM person_qualification")
                ).scalar_one()
                >= 1
            )
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
                tuple(item.get("column_names") or ()) for item in constraints
            }


def _run_alembic(database: Path, revision: str):
    environment = os.environ.copy()
    environment.update(
        {
            "DATABASE_URL": f"sqlite:///{database}",
            "FLASK_SECRET_KEY": "migration-fixture-secret-2026-long",
            "ATCROSTER_SKIP_BOOTSTRAP": "true",
            "ATCROSTER_SKIP_RUNTIME_SCHEMA": "1",
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_position_matrix_is_converted_to_constrained_relational_rows(tmp_path):
    database = tmp_path / "position-matrix.db"
    assert _run_alembic(database, "20260801_33").returncode == 0
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO operational_position "
                "(id, unit_id, code, label, description, is_safety_critical, "
                "is_active, maximum_session_duration_minutes, "
                "maximum_session_duration_matrix_json) "
                "VALUES (7, 1, 'TWR', 'Tower', '', 1, 1, 120, :matrix)"
            ),
            {"matrix": '{"0:8": 75, "6:22": 90}'},
        )
    result = _run_alembic(database, "head")
    assert result.returncode == 0, result.stdout + result.stderr
    with engine.connect() as connection:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns("operational_position")
        }
        assert "maximum_session_duration_matrix_json" not in columns
        assert connection.execute(
            text(
                "SELECT weekday, start_hour, maximum_duration_minutes "
                "FROM operational_position_time_allowance "
                "WHERE position_id=7 ORDER BY weekday, start_hour"
            )
        ).all() == [(0, 8, 75), (6, 22, 90)]


def test_invalid_position_matrix_blocks_the_data_migration(tmp_path):
    database = tmp_path / "invalid-position-matrix.db"
    assert _run_alembic(database, "20260801_33").returncode == 0
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO operational_position "
                "(id, unit_id, code, label, description, is_safety_critical, "
                "is_active, maximum_session_duration_minutes, "
                "maximum_session_duration_matrix_json) "
                "VALUES (8, 1, 'BAD', 'Invalid', '', 1, 1, 120, :matrix)"
            ),
            {"matrix": '{"7:25": 0}'},
        )
    result = _run_alembic(database, "head")
    assert result.returncode != 0
    assert "outside the weekly matrix" in result.stderr


def test_assignment_values_are_conservatively_classified(tmp_path):
    database = tmp_path / "assignment-baseline-override.db"
    assert _run_alembic(database, "20260803_41").returncode == 0
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO assignment "
            "(id, unit_id, day, code, source, note) VALUES "
            "(1, 1, '2026-11-01', 'M', 'auto', 'pattern'), "
            "(2, 1, '2026-11-02', 'A', 'auto', 'generated watch coverage'), "
            "(3, 1, '2026-11-03', 'N', 'manual', 'pattern'), "
            "(4, 1, '2026-11-04', 'AL', 'auto', 'leave'), "
            "(5, 1, '2026-11-05', 'D', 'legacy-import', '')"
        ))
    result = _run_alembic(database, "head")
    assert result.returncode == 0, result.stdout + result.stderr
    with engine.connect() as connection:
        rows = connection.execute(text(
            "SELECT id, code, generated_code, override_code, override_type "
            "FROM assignment ORDER BY id"
        )).all()
    assert rows == [
        (1, "M", "M", None, None),
        (2, "A", "A", None, None),
        (3, "N", None, "N", "MIGRATED_MANUAL"),
        (4, "AL", None, "AL", "MIGRATED_ABSENCE"),
        (5, "D", None, "D", "MIGRATED_UNCERTAIN"),
    ]
    summary, uncertain = classification_summary(f"sqlite:///{database}")
    assert summary == {
        "GENERATED_BASELINE": 2,
        "MIGRATED_ABSENCE": 1,
        "MIGRATED_MANUAL": 1,
        "MIGRATED_UNCERTAIN": 1,
    }
    assert uncertain == [5]


def test_roster_impact_status_migration_handles_populated_legacy_rows(tmp_path):
    database = tmp_path / "populated-roster-impact.db"
    connection = sqlite3.connect(database)
    connection.executescript(FIXTURES["full_original"])
    connection.commit()
    connection.close()
    result = _run_alembic(database, "20260803_48")
    assert result.returncode == 0, result.stdout + result.stderr
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO roster_impact_event "
            "(id, unit_id, event_type, effective_from, staff_ids_json, "
            "watch_ids_json, rebuild_baseline, recalculate_coverage, "
            "preserve_overrides, reason, status, result_json, created_at) "
            "VALUES (1, 1, 'MANUAL_RECALCULATION', '2026-08-01', '[]', "
            "'[]', 0, 1, 1, '', 'PROCESSING', '{}', CURRENT_TIMESTAMP)"
        ))
        connection.execute(text(
            "INSERT INTO roster_impact_exception "
            "(unit_id, event_id, effective_from, effective_to, exception_type, "
            "severity, description, status, resolution_note, created_at) "
            "VALUES (1, 1, '2026-08-01', '2026-08-01', "
            "'PATTERN_CHANGE_REQUIRES_REVIEW', 'WARNING', 'legacy', "
            "'DISMISSED', '', CURRENT_TIMESTAMP)"
        ))
    result = _run_alembic(database, "head")
    assert result.returncode == 0, result.stdout + result.stderr
    with engine.connect() as connection:
        assert connection.execute(text(
            "SELECT status FROM roster_impact_event WHERE id=1"
        )).scalar_one() == "RUNNING"
        assert connection.execute(text(
            "SELECT status FROM roster_impact_exception WHERE event_id=1"
        )).scalar_one() == "NOT_APPLICABLE"
