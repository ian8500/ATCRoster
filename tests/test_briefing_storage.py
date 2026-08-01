
import pytest

from briefing_storage import (
    BriefingStorageError,
    LocalBriefingStorage,
    configured_briefing_storage,
)


def test_development_storage_can_use_instance_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "development")
    monkeypatch.delenv("BRIEFING_STORAGE_PROVIDER", raising=False)
    monkeypatch.delenv("ATCROSTER_BRIEFING_UPLOAD_DIR", raising=False)

    storage = configured_briefing_storage(str(tmp_path))

    assert isinstance(storage, LocalBriefingStorage)
    assert storage.root == tmp_path / "briefing_uploads"


def test_production_storage_rejects_implicit_local_directory(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "production")
    monkeypatch.setenv("BRIEFING_STORAGE_PROVIDER", "local")
    monkeypatch.delenv("ATCROSTER_BRIEFING_DURABLE_DIR", raising=False)
    monkeypatch.setenv(
        "ATCROSTER_BRIEFING_UPLOAD_DIR", str(tmp_path / "not-a-mount")
    )

    with pytest.raises(BriefingStorageError, match="durable"):
        configured_briefing_storage(str(tmp_path))


def test_production_storage_accepts_explicit_absolute_durable_mount(
    monkeypatch, tmp_path
):
    durable_root = tmp_path / "briefing"
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "production")
    monkeypatch.setenv("BRIEFING_STORAGE_PROVIDER", "mounted")
    monkeypatch.setenv(
        "ATCROSTER_BRIEFING_DURABLE_DIR", str(durable_root)
    )

    storage = configured_briefing_storage(str(tmp_path))

    assert isinstance(storage, LocalBriefingStorage)
    assert storage.root == durable_root


def test_production_storage_rejects_relative_durable_mount(monkeypatch):
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "production")
    monkeypatch.setenv("BRIEFING_STORAGE_PROVIDER", "mounted")
    monkeypatch.setenv("ATCROSTER_BRIEFING_DURABLE_DIR", "briefing")

    with pytest.raises(BriefingStorageError, match="absolute"):
        configured_briefing_storage("/srv/atcroster")


def test_s3_storage_rejects_incomplete_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "production")
    monkeypatch.setenv("BRIEFING_STORAGE_PROVIDER", "s3")
    for name in (
        "BRIEFING_STORAGE_BUCKET",
        "BRIEFING_STORAGE_ENDPOINT",
        "BRIEFING_STORAGE_ACCESS_KEY",
        "BRIEFING_STORAGE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(BriefingStorageError, match="incomplete"):
        configured_briefing_storage(str(tmp_path))
