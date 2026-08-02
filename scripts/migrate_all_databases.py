#!/usr/bin/env python3
"""Upgrade the control database, then every secret-routed airport database."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

try:
    from scripts.database_grants import apply_runtime_grants
except ModuleNotFoundError:  # Railway invokes this file directly.
    from database_grants import apply_runtime_grants

REPOSITORY = Path(__file__).resolve().parents[1]
SECRET_NAME_PATTERN = re.compile(r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL")
MIGRATION_ADVISORY_LOCK_ID = 4_287_603_356


def _sqlalchemy_database_url(database_url: str) -> str:
    """Select the pinned psycopg 3 driver for generic PostgreSQL URLs."""
    parsed = make_url(database_url)
    if parsed.get_backend_name() in {"postgres", "postgresql"}:
        parsed = parsed.set(drivername="postgresql+psycopg")
    return parsed.render_as_string(hide_password=False)


def _migration_database_url(secret_name: str, runtime_url: str) -> str:
    """Prefer owner-only DDL credentials without changing runtime settings."""
    if secret_name == "CONTROL_DATABASE_URL":
        migration_secret_name = "CONTROL_MIGRATION_DATABASE_URL"
    elif SECRET_NAME_PATTERN.fullmatch(secret_name):
        migration_secret_name = (
            secret_name.removesuffix("_DATABASE_URL") + "_MIGRATION_DATABASE_URL"
        )
    else:
        raise ValueError("Unsupported database secret name.")
    return os.environ.get(migration_secret_name) or runtime_url


def _apply_runtime_grants_after_upgrade(migration_url: str, runtime_url: str) -> None:
    """Grant the runtime login access to relations created by this release.

    PostgreSQL does not automatically extend existing table grants to tables
    created later by Alembic. Railway's pre-deploy migration therefore needs to
    refresh the least-privilege grants before the new application starts.
    """
    migration = make_url(migration_url)
    runtime = make_url(runtime_url)
    if migration.get_backend_name() not in {"postgres", "postgresql"}:
        return
    if _canonical_database_url(migration_url) == _canonical_database_url(runtime_url):
        return
    if not runtime.username:
        raise RuntimeError("Runtime database URL must identify a PostgreSQL role.")
    apply_runtime_grants(
        migration_url,
        runtime.username,
        os.environ.get("ATCROSTER_AUDIT_READ_ROLE") or None,
    )


@contextmanager
def deployment_migration_lock(database_url: str):
    """Serialize concurrent Railway pre-deploy migrations on PostgreSQL.

    Railway runs the shared pre-deploy command for both the web and worker
    services.  A session-level advisory lock keeps those processes from racing
    while Alembic updates the same version tables.  Other database engines are
    retained for local development and do not need this PostgreSQL primitive.
    """
    engine = create_engine(_sqlalchemy_database_url(database_url), pool_pre_ping=True)
    connection = engine.connect()
    locked = False
    try:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_lock(:lock_id)"),
                {"lock_id": MIGRATION_ADVISORY_LOCK_ID},
            )
            locked = True
        yield
    finally:
        if locked:
            connection.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": MIGRATION_ADVISORY_LOCK_ID},
            )
        connection.close()
        engine.dispose()


def _ensure_info_annotation(database_url: str, unit_id: int) -> None:
    """Idempotently seed the non-reportable INFO annotation."""
    engine = create_engine(_sqlalchemy_database_url(database_url), pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            if "annotation_type" not in inspect(connection).get_table_names():
                return
            exists = connection.execute(
                text(
                    "SELECT 1 FROM annotation_type "
                    "WHERE unit_id = :unit_id AND code = 'INFO'"
                ),
                {"unit_id": unit_id},
            ).first()
            if exists:
                return
            connection.execute(
                text(
                    "INSERT INTO annotation_type "
                    "(unit_id, code, label, category, colour, description, "
                    "allow_suffix, suffixes, toil_half_days, tags, "
                    "note_required, admin_only, has_been_used, is_active, "
                    "sort_order) VALUES "
                    "(:unit_id, 'INFO', 'Information', 'Information', "
                    "'#6c757d', :description, false, '', 0, "
                    "'info,report_exclude', false, false, false, true, 0)"
                ),
                {
                    "unit_id": unit_id,
                    "description": (
                        "Additional roster information. Excluded from reports."
                    ),
                },
            )
    finally:
        engine.dispose()


def upgrade_database(database_url: str, schema_role: str) -> str:
    if schema_role not in {"control", "operational", "combined"}:
        raise ValueError("schema_role must be control, operational, or combined")
    if (
        os.environ.get("ATCROSTER_ENVIRONMENT") == "production"
        and schema_role == "combined"
    ):
        raise RuntimeError("Combined schema migrations are forbidden in production.")
    previous = os.environ.get("DATABASE_URL")
    previous_role = os.environ.get("ATCROSTER_SCHEMA_ROLE")
    os.environ["DATABASE_URL"] = _sqlalchemy_database_url(database_url)
    os.environ["ATCROSTER_SCHEMA_ROLE"] = schema_role
    try:
        config = Config(str(REPOSITORY / "alembic.ini"))
        config.set_main_option("script_location", str(REPOSITORY / "migrations"))
        command.upgrade(config, "head")
        engine = create_engine(
            _sqlalchemy_database_url(database_url), pool_pre_ping=True
        )
        try:
            with engine.connect() as connection:
                return (
                    MigrationContext.configure(connection).get_current_revision() or ""
                )
        finally:
            engine.dispose()
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        if previous_role is None:
            os.environ.pop("ATCROSTER_SCHEMA_ROLE", None)
        else:
            os.environ["ATCROSTER_SCHEMA_ROLE"] = previous_role


def main() -> None:
    control_runtime_url = os.environ.get("CONTROL_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if not control_runtime_url:
        raise SystemExit("CONTROL_DATABASE_URL is required.")
    control_url = _migration_database_url("CONTROL_DATABASE_URL", control_runtime_url)
    with deployment_migration_lock(control_url):
        control_version = upgrade_database(control_url, "control")
        _apply_runtime_grants_after_upgrade(control_url, control_runtime_url)
        print(f"Control database upgraded to {control_version}.")
        control_engine = create_engine(
            _sqlalchemy_database_url(control_url), pool_pre_ping=True
        )
        try:
            with control_engine.connect() as connection:
                routes = connection.execute(
                    text(
                        "SELECT r.unit_id, r.secret_name "
                        "FROM database_routing_metadata r "
                        "ORDER BY r.unit_id"
                    )
                ).all()
        finally:
            control_engine.dispose()
        configured_routes = {secret_name: unit_id for unit_id, secret_name in routes}
        # Deployment manifests may provide an airport database before its control
        # metadata is created. Migrate every strictly named secret as well as every
        # registered route; values are never printed.
        for secret_name in os.environ:
            if SECRET_NAME_PATTERN.fullmatch(secret_name):
                configured_routes.setdefault(
                    secret_name,
                    int(
                        secret_name.removeprefix("ATCROSTER_UNIT_").removesuffix(
                            "_DATABASE_URL"
                        )
                    ),
                )
        for secret_name, unit_id in sorted(
            configured_routes.items(), key=lambda item: item[1]
        ):
            if not SECRET_NAME_PATTERN.fullmatch(secret_name or ""):
                raise SystemExit(
                    f"Unit {unit_id} has an invalid deployment-secret name."
                )
            runtime_operational_url = os.environ.get(secret_name)
            if not runtime_operational_url:
                raise SystemExit(
                    f"Required deployment secret {secret_name} is unavailable."
                )
            operational_url = _migration_database_url(
                secret_name, runtime_operational_url
            )
            if _canonical_database_url(operational_url) == _canonical_database_url(
                control_url
            ):
                raise SystemExit(
                    f"Unit {unit_id} operational database must differ from control."
                )
            version = upgrade_database(operational_url, "operational")
            _ensure_info_annotation(operational_url, unit_id)
            _apply_runtime_grants_after_upgrade(
                operational_url, runtime_operational_url
            )
            print(f"Operational database for unit {unit_id} upgraded to {version}.")


def _canonical_database_url(value: str) -> str:
    """Compare endpoints without ever logging or returning their credentials."""
    from sqlalchemy.engine import make_url

    parsed = make_url(value)
    return str(parsed.set(password=None))


if __name__ == "__main__":
    main()
