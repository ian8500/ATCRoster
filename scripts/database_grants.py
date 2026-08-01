"""PostgreSQL runtime-role grants and verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os

import psycopg
from psycopg import sql

from scripts.database_backup import psycopg_dsn


AUDIT_TABLES = frozenset(
    {
        "annotation_audit",
        "briefing_audit",
        "central_security_audit",
        "change_log",
        "position_session_audit",
        "request_audit",
        "sms_audit",
        "super_admin_audit",
    }
)
RUNTIME_TABLE_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"})
AUDIT_RUNTIME_PRIVILEGES = frozenset({"SELECT", "INSERT"})


@dataclass(frozen=True)
class GrantVerification:
    database: str
    runtime_role: str
    tables_checked: int
    audit_tables_checked: int
    sequences_checked: int


def required_environment(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def _relations(connection: psycopg.Connection, kind: str) -> list[str]:
    if kind == "table":
        rows = connection.execute(
            "SELECT tablename FROM pg_catalog.pg_tables "
            "WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
    elif kind == "sequence":
        rows = connection.execute(
            "SELECT sequencename FROM pg_catalog.pg_sequences "
            "WHERE schemaname = 'public' ORDER BY sequencename"
        ).fetchall()
    else:
        raise ValueError("Unsupported relation kind")
    return [str(row[0]) for row in rows]


def _database_name(connection: psycopg.Connection) -> str:
    return str(connection.execute("SELECT current_database()").fetchone()[0])


def apply_runtime_grants(
    database_url: str,
    runtime_role: str,
    audit_read_role: str | None = None,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Apply least-privilege grants using a migration-owner connection."""
    statements: list[str] = []
    with psycopg.connect(psycopg_dsn(database_url)) as connection:
        tables = _relations(connection, "table")
        sequences = _relations(connection, "sequence")
        if not tables:
            raise RuntimeError("Database has no public tables; run migrations first.")
        roles = {
            str(row[0])
            for row in connection.execute(
                "SELECT rolname FROM pg_catalog.pg_roles WHERE rolname = ANY(%s)",
                ([role for role in (runtime_role, audit_read_role) if role],),
            ).fetchall()
        }
        for required in (runtime_role, audit_read_role):
            if required and required not in roles:
                raise RuntimeError(f"PostgreSQL role {required!r} does not exist.")

        def execute(statement: sql.Composed) -> None:
            rendered = statement.as_string(connection)
            statements.append(rendered)
            if not dry_run:
                connection.execute(statement)

        runtime = sql.Identifier(runtime_role)
        execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM PUBLIC"))
        execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(_database_name(connection)), runtime
            )
        )
        execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(runtime))
        for table in tables:
            relation = sql.Identifier("public", table)
            execute(
                sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {} FROM {}").format(
                    relation, runtime
                )
            )
            privileges = (
                "SELECT, INSERT"
                if table in AUDIT_TABLES
                else "SELECT, INSERT, UPDATE, DELETE"
            )
            execute(
                sql.SQL("GRANT " + privileges + " ON TABLE {} TO {}").format(
                    relation, runtime
                )
            )
        for sequence in sequences:
            execute(
                sql.SQL("GRANT USAGE, SELECT ON SEQUENCE {} TO {}").format(
                    sql.Identifier("public", sequence), runtime
                )
            )
        if audit_read_role:
            reader = sql.Identifier(audit_read_role)
            execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(_database_name(connection)), reader
                )
            )
            execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(reader))
            for table in sorted(AUDIT_TABLES.intersection(tables)):
                execute(
                    sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(
                        sql.Identifier("public", table), reader
                    )
                )
        if dry_run:
            connection.rollback()
        else:
            connection.commit()
    return statements


def verify_runtime_grants(
    database_url: str,
    runtime_role: str,
) -> GrantVerification:
    """Fail unless the runtime role has exactly the required table boundary."""
    failures: list[str] = []
    with psycopg.connect(psycopg_dsn(database_url)) as connection:
        database = _database_name(connection)
        tables = _relations(connection, "table")
        sequences = _relations(connection, "sequence")
        if not tables:
            raise RuntimeError("Database has no public tables; run migrations first.")
        if not connection.execute(
            "SELECT has_schema_privilege(%s, 'public', 'USAGE')", (runtime_role,)
        ).fetchone()[0]:
            failures.append("public schema: missing USAGE")
        if connection.execute(
            "SELECT has_schema_privilege(%s, 'public', 'CREATE')", (runtime_role,)
        ).fetchone()[0]:
            failures.append("public schema: unexpected CREATE")
        for table in tables:
            expected = (
                AUDIT_RUNTIME_PRIVILEGES
                if table in AUDIT_TABLES
                else RUNTIME_TABLE_PRIVILEGES
            )
            for privilege in RUNTIME_TABLE_PRIVILEGES | {"TRUNCATE"}:
                actual = bool(
                    connection.execute(
                        "SELECT has_table_privilege(%s, %s, %s)",
                        (runtime_role, f"public.{table}", privilege),
                    ).fetchone()[0]
                )
                if actual != (privilege in expected):
                    failures.append(
                        f"{table}: {privilege} expected={privilege in expected} "
                        f"actual={actual}"
                    )
        for sequence in sequences:
            for privilege in ("USAGE", "SELECT"):
                if not connection.execute(
                    "SELECT has_sequence_privilege(%s, %s, %s)",
                    (runtime_role, f"public.{sequence}", privilege),
                ).fetchone()[0]:
                    failures.append(f"{sequence}: missing sequence {privilege}")
        if failures:
            raise RuntimeError(
                "Runtime database grant verification failed: " + "; ".join(failures)
            )
        return GrantVerification(
            database=database,
            runtime_role=runtime_role,
            tables_checked=len(tables),
            audit_tables_checked=len(AUDIT_TABLES.intersection(tables)),
            sequences_checked=len(sequences),
        )
