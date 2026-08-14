"""Restore a verified backup into a new, empty PostgreSQL database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


# Permit the documented ``python scripts/restore_database.py`` invocation from
# the repository root without requiring callers to set PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.database_backup import restore_backup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--target-url-env", required=True)
    parser.add_argument("--confirm", required=True)
    arguments = parser.parse_args()
    if arguments.confirm != "RESTORE-INTO-EMPTY-DATABASE":
        parser.error("--confirm must be RESTORE-INTO-EMPTY-DATABASE")
    database_url = os.environ.get(arguments.target_url_env, "")
    if not database_url:
        parser.error(f"{arguments.target_url_env} is not set")
    result = restore_backup(arguments.archive, arguments.metadata, database_url)
    print(f"Restored {result.database_label} at revision {result.alembic_revision}.")


if __name__ == "__main__":
    main()
