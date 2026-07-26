#!/usr/bin/env python3
"""Upgrade the control database, then every secret-routed airport database."""
from __future__ import annotations

import os
import re
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text

REPOSITORY = Path(__file__).resolve().parents[1]
SECRET_NAME_PATTERN = re.compile(r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL")


def upgrade_database(
    database_url: str, schema_role: str = "combined"
) -> str:
    previous = os.environ.get("DATABASE_URL")
    previous_role = os.environ.get("ATCROSTER_SCHEMA_ROLE")
    os.environ["DATABASE_URL"] = database_url
    os.environ["ATCROSTER_SCHEMA_ROLE"] = schema_role
    try:
        config = Config(str(REPOSITORY / "alembic.ini"))
        config.set_main_option("script_location", str(REPOSITORY / "migrations"))
        command.upgrade(config, "head")
        engine = create_engine(database_url, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                return MigrationContext.configure(
                    connection
                ).get_current_revision() or ""
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
    control_url = os.environ.get("CONTROL_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if not control_url:
        raise SystemExit("CONTROL_DATABASE_URL is required.")
    control_version = upgrade_database(control_url, "control")
    print(f"Control database upgraded to {control_version}.")
    control_engine = create_engine(control_url, pool_pre_ping=True)
    try:
        with control_engine.connect() as connection:
            routes = connection.execute(text(
                "SELECT r.unit_id, r.secret_name "
                "FROM database_routing_metadata r "
                "ORDER BY r.unit_id"
            )).all()
    finally:
        control_engine.dispose()
    for unit_id, secret_name in routes:
        if not SECRET_NAME_PATTERN.fullmatch(secret_name or ""):
            raise SystemExit(
                f"Unit {unit_id} has an invalid deployment-secret name."
            )
        operational_url = os.environ.get(secret_name)
        if not operational_url:
            raise SystemExit(
                f"Required deployment secret {secret_name} is unavailable."
            )
        if operational_url == control_url:
            raise SystemExit(
                f"Unit {unit_id} operational database must differ from control."
            )
        version = upgrade_database(operational_url, "operational")
        print(
            f"Operational database for unit {unit_id} upgraded to "
            f"{version}."
        )


if __name__ == "__main__":
    main()
