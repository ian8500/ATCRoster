#!/usr/bin/env python3
"""Apply least-privilege runtime grants to one migrated PostgreSQL database."""

from __future__ import annotations

import argparse

from scripts.database_grants import (
    apply_runtime_grants,
    required_environment,
    verify_runtime_grants,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url-env",
        required=True,
        help="Name of the environment variable containing the owner/migration URL",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    database_url = required_environment(arguments.database_url_env)
    runtime_role = required_environment("ATCROSTER_RUNTIME_DATABASE_ROLE")
    audit_read_role = None
    try:
        audit_read_role = required_environment("ATCROSTER_AUDIT_READ_ROLE")
    except RuntimeError:
        pass
    statements = apply_runtime_grants(
        database_url,
        runtime_role,
        audit_read_role,
        dry_run=arguments.dry_run,
    )
    if arguments.dry_run:
        print(f"Dry run validated {len(statements)} grant statements.")
        return
    result = verify_runtime_grants(database_url, runtime_role)
    print(
        f"Verified runtime grants for {result.database}: "
        f"{result.tables_checked} tables, {result.audit_tables_checked} audit "
        f"tables and {result.sequences_checked} sequences."
    )


if __name__ == "__main__":
    main()
