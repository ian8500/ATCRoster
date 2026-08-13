"""Compatibility upgrades for legacy single-database desktop installations."""

from __future__ import annotations

import secrets
from typing import Any

from sqlalchemy import text


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
