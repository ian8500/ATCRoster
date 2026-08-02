import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.migrate_all_databases import (
    _apply_runtime_grants_after_upgrade,
    _migration_database_url,
)


def test_migration_database_url_prefers_separate_control_owner(monkeypatch):
    monkeypatch.setenv("CONTROL_MIGRATION_DATABASE_URL", "postgresql://owner/control")
    assert (
        _migration_database_url("CONTROL_DATABASE_URL", "postgresql://runtime/control")
        == "postgresql://owner/control"
    )


def test_migration_database_url_prefers_matching_unit_owner(monkeypatch):
    monkeypatch.setenv(
        "ATCROSTER_UNIT_2_MIGRATION_DATABASE_URL",
        "postgresql://owner/operational",
    )
    assert (
        _migration_database_url(
            "ATCROSTER_UNIT_2_DATABASE_URL", "postgresql://runtime/operational"
        )
        == "postgresql://owner/operational"
    )


def test_migration_database_url_retains_local_runtime_fallback(monkeypatch):
    monkeypatch.delenv("CONTROL_MIGRATION_DATABASE_URL", raising=False)
    assert (
        _migration_database_url("CONTROL_DATABASE_URL", "sqlite:///local.db")
        == "sqlite:///local.db"
    )


def test_migration_database_url_rejects_uncontrolled_secret_names():
    with pytest.raises(ValueError, match="Unsupported database secret"):
        _migration_database_url("SOME_DATABASE_URL", "sqlite:///unsafe.db")


def test_postgresql_upgrade_refreshes_runtime_grants(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "scripts.migrate_all_databases.apply_runtime_grants",
        lambda *args: calls.append(args),
    )
    monkeypatch.setenv("ATCROSTER_AUDIT_READ_ROLE", "audit_reader")

    _apply_runtime_grants_after_upgrade(
        "postgresql://owner:secret@database.example/airport",
        "postgresql://runtime:secret@database.example/airport",
    )

    assert calls == [
        (
            "postgresql://owner:secret@database.example/airport",
            "runtime",
            "audit_reader",
        )
    ]


def test_local_upgrade_does_not_apply_postgresql_grants(monkeypatch):
    monkeypatch.setattr(
        "scripts.migrate_all_databases.apply_runtime_grants",
        lambda *args: pytest.fail("runtime grants should not run for SQLite"),
    )

    _apply_runtime_grants_after_upgrade("sqlite:///local.db", "sqlite:///local.db")


def test_migration_script_supports_railway_direct_invocation(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "migrate_all_databases.py"
    environment = os.environ.copy()
    environment.pop("CONTROL_DATABASE_URL", None)
    environment.pop("DATABASE_URL", None)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0
    assert "CONTROL_DATABASE_URL is required" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
