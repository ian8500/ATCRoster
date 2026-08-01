import json

import pytest

from scripts.database_backup import connection_environment, load_metadata, sha256_file


def test_connection_credentials_are_kept_out_of_process_arguments():
    environment, arguments = connection_environment(
        "postgresql+psycopg://backup-user:very-secret@db.internal:5433/airport?sslmode=require"
    )
    assert environment["PGPASSWORD"] == "very-secret"
    assert environment["PGSSLMODE"] == "require"
    assert "very-secret" not in " ".join(arguments)
    assert arguments == [
        "--dbname",
        "airport",
        "--host",
        "db.internal",
        "--port",
        "5433",
        "--username",
        "backup-user",
    ]


def test_backup_metadata_rejects_unknown_format(tmp_path):
    metadata = tmp_path / "backup.json"
    metadata.write_text(
        json.dumps(
            {
                "format_version": 2,
                "database_label": "control",
                "schema_role": "control",
                "created_at": "2026-08-01T00:00:00+00:00",
                "alembic_revision": "head",
                "archive_file": "control.dump",
                "archive_bytes": 1,
                "archive_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported"):
        load_metadata(metadata)


def test_checksum_changes_when_archive_is_tampered(tmp_path):
    archive = tmp_path / "control.dump"
    archive.write_bytes(b"original")
    original = sha256_file(archive)
    archive.write_bytes(b"tampered")
    assert sha256_file(archive) != original


def test_backup_metadata_requires_all_fields(tmp_path):
    metadata = tmp_path / "backup.json"
    metadata.write_text('{"format_version": 1}')
    with pytest.raises(ValueError, match="metadata is invalid"):
        load_metadata(metadata)
