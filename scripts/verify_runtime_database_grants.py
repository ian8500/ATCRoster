#!/usr/bin/env python3
"""Verify least-privilege runtime grants on one PostgreSQL database."""

from __future__ import annotations

import argparse

from scripts.database_grants import required_environment, verify_runtime_grants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url-env", required=True)
    arguments = parser.parse_args()
    result = verify_runtime_grants(
        required_environment(arguments.database_url_env),
        required_environment("ATCROSTER_RUNTIME_DATABASE_ROLE"),
    )
    print(
        f"Runtime grants are valid for {result.database}: "
        f"{result.tables_checked} tables, {result.audit_tables_checked} audit "
        f"tables and {result.sequences_checked} sequences."
    )


if __name__ == "__main__":
    main()
