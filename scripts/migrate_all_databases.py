#!/usr/bin/env python3
"""Upgrade the control database, then every secret-routed airport database."""
from __future__ import annotations

import os
import re
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

REPOSITORY = Path(__file__).resolve().parents[1]
SECRET_NAME_PATTERN = re.compile(r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL")


def _upgrade(database_url: str) -> None:
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        config = Config(str(REPOSITORY / "alembic.ini"))
        config.set_main_option("script_location", str(REPOSITORY / "migrations"))
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def main() -> None:
    control_url = os.environ.get("CONTROL_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )
    if not control_url:
        raise SystemExit("CONTROL_DATABASE_URL is required.")
    _upgrade(control_url)
    control_engine = create_engine(control_url, pool_pre_ping=True)
    try:
        with control_engine.connect() as connection:
            routes = connection.execute(text(
                "SELECT r.unit_id, r.secret_name, u.code, u.name, "
                "u.timezone, u.locale, u.date_format, u.branding_json "
                "FROM database_routing_metadata r "
                "JOIN unit u ON u.id=r.unit_id ORDER BY r.unit_id"
            )).all()
    finally:
        control_engine.dispose()
    for (
        unit_id, secret_name, code, name, timezone_name, locale,
        date_format, branding_json,
    ) in routes:
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
        _upgrade(operational_url)
        operational_engine = create_engine(
            operational_url, pool_pre_ping=True
        )
        try:
            with operational_engine.begin() as connection:
                connection.execute(text(
                    "INSERT INTO unit "
                    "(id,code,name,timezone,locale,date_format,branding_json,"
                    "status,plan,request_months_ahead,request_lock_day,"
                    "active_user_limit,onboarding_step,created_at) VALUES "
                    "(:id,:code,:name,:timezone,:locale,:date_format,"
                    ":branding,'active','operational',3,20,1,1,"
                    "CURRENT_TIMESTAMP) ON CONFLICT (id) DO NOTHING"
                ), {
                    "id": unit_id, "code": code, "name": name,
                    "timezone": timezone_name, "locale": locale,
                    "date_format": date_format,
                    "branding": branding_json or "{}",
                })
        finally:
            operational_engine.dispose()
        print(f"Upgraded operational database for unit {unit_id}.")


if __name__ == "__main__":
    main()
