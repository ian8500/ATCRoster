"""Compatibility upgrades for legacy single-database desktop installations."""

from __future__ import annotations

import secrets
from typing import Any

from sqlalchemy import inspect, text


def upgrade_tenant_foundation(*, db: Any, Unit: Any) -> None:
    """Idempotently add tenant ownership to legacy desktop databases."""
    inspector = inspect(db.engine)
    if "unit" not in inspector.get_table_names():
        db.create_all()
        inspector = inspect(db.engine)
    if Unit.query.count() == 0:
        db.session.add(Unit(id=1, code="FIRST", name="First airport unit"))
        db.session.commit()

    additions = {
        "staff": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "membership_status": "VARCHAR(20) NOT NULL DEFAULT 'active'",
            "permissions_json": "TEXT NOT NULL DEFAULT '{}'",
        },
        "watch": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "requirement": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "leave": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "sickness": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "ai_rule_set": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "change_log": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "staff_watch_history": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "shift_type": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
            "is_requestable": "BOOLEAN NOT NULL DEFAULT 0",
            "required_qualification": "VARCHAR(40) NOT NULL DEFAULT ''",
        },
        "assignment": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "shift_request": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "requester_comment": "VARCHAR(500) NOT NULL DEFAULT ''",
            "created_at": "DATETIME",
            "updated_at": "DATETIME",
            "fulfilled_at": "DATETIME",
            "cancelled_at": "DATETIME",
            "resulting_assignment_id": "INTEGER",
        },
        "annotation_type": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "colour": "VARCHAR(20) NOT NULL DEFAULT '#6c757d'",
            "description": "TEXT NOT NULL DEFAULT ''",
            "note_required": "BOOLEAN NOT NULL DEFAULT 0",
            "admin_only": "BOOLEAN NOT NULL DEFAULT 0",
            "has_been_used": "BOOLEAN NOT NULL DEFAULT 0",
        },
    }
    tables = set(inspector.get_table_names())
    for table_name, columns in additions.items():
        if table_name not in tables:
            continue
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        for name, definition in columns.items():
            if name not in existing:
                db.session.execute(
                    text(f'ALTER TABLE "{table_name}" ADD COLUMN "{name}" {definition}')
                )
        db.session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_unit_id" '
                f'ON "{table_name}" ("unit_id")'
            )
        )
    if "shift_request" in tables:
        db.session.execute(
            text(
                "UPDATE shift_request SET "
                "created_at = COALESCE(created_at, submitted_at), "
                "updated_at = COALESCE(updated_at, submitted_at)"
            )
        )
    db.session.commit()


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
