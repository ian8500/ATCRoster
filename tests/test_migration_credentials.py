import pytest

from scripts.migrate_all_databases import _migration_database_url


def test_migration_database_url_prefers_separate_control_owner(monkeypatch):
    monkeypatch.setenv("CONTROL_MIGRATION_DATABASE_URL", "postgresql://owner/control")
    assert _migration_database_url(
        "CONTROL_DATABASE_URL", "postgresql://runtime/control"
    ) == "postgresql://owner/control"


def test_migration_database_url_prefers_matching_unit_owner(monkeypatch):
    monkeypatch.setenv(
        "ATCROSTER_UNIT_2_MIGRATION_DATABASE_URL",
        "postgresql://owner/operational",
    )
    assert _migration_database_url(
        "ATCROSTER_UNIT_2_DATABASE_URL", "postgresql://runtime/operational"
    ) == "postgresql://owner/operational"


def test_migration_database_url_retains_local_runtime_fallback(monkeypatch):
    monkeypatch.delenv("CONTROL_MIGRATION_DATABASE_URL", raising=False)
    assert _migration_database_url(
        "CONTROL_DATABASE_URL", "sqlite:///local.db"
    ) == "sqlite:///local.db"


def test_migration_database_url_rejects_uncontrolled_secret_names():
    with pytest.raises(ValueError, match="Unsupported database secret"):
        _migration_database_url("SOME_DATABASE_URL", "sqlite:///unsafe.db")
