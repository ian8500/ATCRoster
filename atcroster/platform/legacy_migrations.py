"""Compatibility upgrades for legacy single-database desktop installations."""

from __future__ import annotations

import secrets
from typing import Any

from sqlalchemy import text


def execute_session_ddl(*, db: Any, statement: str) -> None:
    """Apply optional legacy DDL without leaving a failed session behind."""
    try:
        db.session.execute(text(statement))
        db.session.commit()
    except Exception:
        db.session.rollback()


def add_assignment_annotation(*, db: Any) -> None:
    execute_session_ddl(
        db=db,
        statement="ALTER TABLE assignment ADD COLUMN annotation VARCHAR(20)",
    )


def add_unique_assignment_key(*, db: Any) -> None:
    execute_session_ddl(
        db=db,
        statement=(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_assignment_staff_day "
            "ON assignment(staff_id, day)"
        ),
    )


def add_performance_indexes(*, db: Any, app: Any) -> None:
    """Create the indexes used by legacy desktop roster queries."""
    with app.app_context():
        for statement in (
            "CREATE INDEX IF NOT EXISTS ix_assignment_day ON assignment(day)",
            "CREATE INDEX IF NOT EXISTS ix_assignment_staff_day "
            "ON assignment(staff_id, day)",
            "CREATE INDEX IF NOT EXISTS ix_requirement_ym ON requirement(year, month)",
        ):
            db.session.execute(text(statement))
        db.session.commit()


def add_columns_if_missing(*, db: Any, table: str, columns: dict[str, str]) -> None:
    """Add a bounded set of columns to a legacy SQLite table."""
    with db.engine.connect() as connection:
        existing = {
            row[1] for row in connection.execute(text(f"PRAGMA table_info({table})"))
        }
        for name, definition in columns.items():
            if name not in existing:
                try:
                    connection.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {definition}")
                    )
                except Exception:
                    pass
    db.session.commit()


def add_role_and_calendar_token(*, db: Any, Staff: Any) -> None:
    """Add legacy authentication fields and normalize existing staff records."""
    with db.engine.connect() as connection:
        columns = {
            row[1] for row in connection.execute(text("PRAGMA table_info(staff)"))
        }
        statements = {
            "role": "ALTER TABLE staff ADD COLUMN role VARCHAR(10) DEFAULT 'user'",
            "calendar_token": (
                "ALTER TABLE staff ADD COLUMN calendar_token VARCHAR(64)"
            ),
            "email": (
                "ALTER TABLE staff ADD COLUMN email VARCHAR(254) NOT NULL DEFAULT ''"
            ),
        }
        for name, statement in statements.items():
            if name not in columns:
                try:
                    connection.execute(text(statement))
                except Exception:
                    # Legacy SQLite variants can report already-applied DDL
                    # differently; record normalization below remains safe.
                    pass
        try:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "ux_staff_calendar_token ON staff (calendar_token)"
                )
            )
        except Exception:
            pass

    changed = False
    for staff in Staff.query.all():
        if not staff.role or staff.role not in {
            "superadmin",
            "admin",
            "editor",
            "user",
        }:
            staff.role = "admin" if getattr(staff, "is_admin", False) else "user"
            changed = True
        if not staff.calendar_token:
            staff.calendar_token = secrets.token_hex(16)
            changed = True
    if changed:
        db.session.commit()
