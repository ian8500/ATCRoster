"""Verify a PostgreSQL recovery artifact without restoring it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Permit the documented ``python scripts/verify_backup.py`` invocation from
# the repository root without requiring callers to set PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.database_backup import verify_backup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("metadata", type=Path)
    arguments = parser.parse_args()
    result = verify_backup(arguments.archive, arguments.metadata)
    print(
        f"Verified {result.database_label} backup at revision "
        f"{result.alembic_revision}."
    )


if __name__ == "__main__":
    main()
