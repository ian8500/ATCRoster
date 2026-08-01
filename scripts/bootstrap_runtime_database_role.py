#!/usr/bin/env python3
"""Create or rotate a PostgreSQL runtime login without printing credentials."""

from __future__ import annotations

import argparse
import re

import psycopg
from psycopg import sql

from scripts.database_backup import psycopg_dsn
from scripts.database_grants import required_environment


ROLE_PATTERN = re.compile(r"[a-z_][a-z0-9_]{0,62}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url-env", required=True)
    arguments = parser.parse_args()
    database_url = required_environment(arguments.database_url_env)
    role = required_environment("ATCROSTER_RUNTIME_DATABASE_ROLE")
    password = required_environment("ATCROSTER_RUNTIME_DATABASE_PASSWORD")
    if not ROLE_PATTERN.fullmatch(role):
        raise RuntimeError("Runtime role has an invalid PostgreSQL identifier.")
    with psycopg.connect(psycopg_dsn(database_url)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (role,)
        ).fetchone()
        identifier = sql.Identifier(role)
        password_literal = sql.Literal(password)
        if exists:
            connection.execute(
                sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    identifier, password_literal
                )
            )
        else:
            connection.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    identifier, password_literal
                )
            )
        connection.execute(
            sql.SQL("ALTER ROLE {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION")
            .format(identifier)
        )
        connection.commit()
    print("Runtime database role is present with restricted role attributes.")


if __name__ == "__main__":
    main()
