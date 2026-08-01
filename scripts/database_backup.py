"""Safe PostgreSQL backup/restore primitives shared by operator scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import psycopg
from sqlalchemy.engine import make_url


SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass(frozen=True)
class BackupMetadata:
    format_version: int
    database_label: str
    schema_role: str
    created_at: str
    alembic_revision: str
    archive_file: str
    archive_bytes: int
    archive_sha256: str


def require_postgresql_url(database_url: str) -> None:
    if make_url(database_url).get_backend_name() not in {"postgresql", "postgres"}:
        raise ValueError("Backup and restore require a PostgreSQL database URL.")


def psycopg_dsn(database_url: str) -> str:
    require_postgresql_url(database_url)
    return (
        make_url(database_url)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )


def connection_environment(database_url: str) -> tuple[dict[str, str], list[str]]:
    """Return libpq environment and non-secret connection arguments."""
    require_postgresql_url(database_url)
    url = make_url(database_url)
    environment = os.environ.copy()
    if url.password:
        environment["PGPASSWORD"] = url.password
    if url.query.get("sslmode"):
        environment["PGSSLMODE"] = str(url.query["sslmode"])
    arguments = ["--dbname", url.database or ""]
    if url.host:
        arguments.extend(["--host", url.host])
    if url.port:
        arguments.extend(["--port", str(url.port)])
    if url.username:
        arguments.extend(["--username", url.username])
    return environment, arguments


def executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"Required PostgreSQL utility {name!r} was not found.")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_revision(database_url: str) -> str:
    with psycopg.connect(psycopg_dsn(database_url)) as connection:
        row = connection.execute(
            "SELECT version_num FROM alembic_version LIMIT 1"
        ).fetchone()
    if not row or not row[0]:
        raise RuntimeError("Database does not contain an Alembic schema version.")
    return str(row[0])


def create_backup(
    database_url: str,
    output_directory: Path,
    database_label: str,
    schema_role: str,
) -> tuple[Path, Path]:
    if not SAFE_LABEL.fullmatch(database_label):
        raise ValueError(
            "Database label must contain lowercase letters, digits or hyphens."
        )
    if schema_role not in {"control", "operational"}:
        raise ValueError("Schema role must be control or operational.")
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_directory / f"{database_label}-{timestamp}.dump"
    metadata_path = archive.with_suffix(".json")
    if archive.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing recovery artifact.")
    environment, connection_args = connection_environment(database_url)
    subprocess.run(
        [
            executable("pg_dump"),
            *connection_args,
            "--format=custom",
            "--no-owner",
            "--no-acl",
            "--file",
            str(archive),
        ],
        env=environment,
        check=True,
    )
    archive.chmod(0o600)
    metadata = BackupMetadata(
        format_version=1,
        database_label=database_label,
        schema_role=schema_role,
        created_at=datetime.now(timezone.utc).isoformat(),
        alembic_revision=schema_revision(database_url),
        archive_file=archive.name,
        archive_bytes=archive.stat().st_size,
        archive_sha256=sha256_file(archive),
    )
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n")
    metadata_path.chmod(0o600)
    verify_backup(archive, metadata_path)
    return archive, metadata_path


def load_metadata(metadata_path: Path) -> BackupMetadata:
    try:
        raw = json.loads(metadata_path.read_text())
        metadata = BackupMetadata(**raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Backup metadata is invalid.") from exc
    if metadata.format_version != 1:
        raise ValueError("Unsupported backup metadata format.")
    return metadata


def verify_backup(archive: Path, metadata_path: Path) -> BackupMetadata:
    metadata = load_metadata(metadata_path)
    if archive.name != metadata.archive_file:
        raise ValueError("Backup archive name does not match its metadata.")
    if archive.stat().st_size != metadata.archive_bytes:
        raise ValueError("Backup archive size does not match its metadata.")
    if sha256_file(archive) != metadata.archive_sha256:
        raise ValueError("Backup archive checksum verification failed.")
    subprocess.run(
        [executable("pg_restore"), "--list", str(archive)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return metadata


def assert_empty_restore_target(database_url: str) -> None:
    require_postgresql_url(database_url)
    with psycopg.connect(psycopg_dsn(database_url)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
        ).fetchone()[0]
    if count:
        raise RuntimeError("Restore target is not empty; use a new isolated database.")


def restore_backup(
    archive: Path, metadata_path: Path, database_url: str
) -> BackupMetadata:
    metadata = verify_backup(archive, metadata_path)
    assert_empty_restore_target(database_url)
    environment, connection_args = connection_environment(database_url)
    subprocess.run(
        [
            executable("pg_restore"),
            *connection_args,
            "--exit-on-error",
            "--no-owner",
            "--no-acl",
            str(archive),
        ],
        env=environment,
        check=True,
    )
    restored_revision = schema_revision(database_url)
    if restored_revision != metadata.alembic_revision:
        raise RuntimeError("Restored schema version does not match backup metadata.")
    return metadata
