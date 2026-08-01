"""Verify a PostgreSQL recovery artifact without restoring it."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.database_backup import verify_backup


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
