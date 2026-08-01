#!/usr/bin/env python3
"""Report verifiable release-candidate facts without claiming CI success."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PRODUCTION_ENV = (
    "ATCROSTER_ENVIRONMENT",
    "ATCROSTER_FIELD_ENCRYPTION_KEYS",
    "ATCROSTER_INTERNAL_METRICS_TOKEN",
    "ATCROSTER_TOKEN_ENCRYPTION_KEYS",
    "ATCROSTER_TRUSTED_HOSTS",
    "ATCROSTER_TRUSTED_PROXY_HOPS",
    "CONTROL_DATABASE_URL",
    "DATABASE_URL",
    "FLASK_SECRET_KEY",
    "REDIS_URL",
)
REQUIRED_VERIFICATION_COMMANDS = (
    "python -m pytest --cov --cov-report=term-missing -q",
    "python -m pytest -q tests/test_postgresql_multidatabase.py tests/test_redis_integration.py",
    "ruff check <all tracked Python files>",
    "mypy --ignore-missing-imports --follow-imports=skip atcroster production_operations.py rate_limiting.py signup_locking.py",
    "bandit -q -ll -r <maintained application sources>",
    "pip-audit -r requirements-prod.txt",
)


def command(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-production",
        action="store_true",
        help="fail when production environment variables or backup tools are absent",
    )
    args = parser.parse_args()

    commit_code, commit = command("git", "rev-parse", "HEAD")
    status_code, dirty_output = command(
        "git", "status", "--porcelain", "--", ".", ":(exclude)instance"
    )
    heads_code, heads_output = command(sys.executable, "-m", "alembic", "heads")
    heads = [line.split()[0] for line in heads_output.splitlines() if line.strip()]
    environment = {
        name: "set" if os.environ.get(name) else "missing"
        for name in REQUIRED_PRODUCTION_ENV
    }
    tools = {
        name: shutil.which(name) or "missing" for name in ("pg_dump", "pg_restore")
    }
    report = {
        "commit": commit if commit_code == 0 else "unavailable",
        "dirty_working_tree": bool(dirty_output) if status_code == 0 else None,
        "alembic_heads": heads,
        "schema_compatibility": (
            "not_checked_without_database_connection"
            if not os.environ.get("DATABASE_URL")
            else "must_be_verified_by_migration_and_readiness_checks"
        ),
        "required_environment": environment,
        "backup_tooling": tools,
        "runtime_grants": "not_checked_run_scripts_verify_runtime_database_grants.py",
        "build_metadata": {
            "commit_sha_env": os.environ.get("ATCROSTER_COMMIT_SHA", "missing"),
            "deployment_environment": os.environ.get(
                "ATCROSTER_ENVIRONMENT", "missing"
            ),
        },
        "test_status": "not_run_by_this_script",
        "required_verification_commands": REQUIRED_VERIFICATION_COMMANDS,
        "ci_status": "not_inferred_check_the_exact_commit_in_GitHub_Actions",
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    failures = []
    if commit_code or status_code or dirty_output:
        failures.append("working tree must be a clean Git commit")
    if heads_code or len(heads) != 1:
        failures.append("exactly one Alembic head is required")
    if args.strict_production:
        if any(value == "missing" for value in environment.values()):
            failures.append("required production environment is incomplete")
        if any(value == "missing" for value in tools.values()):
            failures.append("pg_dump and pg_restore are required")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
