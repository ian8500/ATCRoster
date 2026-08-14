"""Create verified PostgreSQL backups without embedding credentials in metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


# Permit the documented ``python scripts/backup_databases.py`` invocation from
# the repository root without requiring callers to set PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.database_backup import create_backup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--database",
        action="append",
        nargs=3,
        metavar=("LABEL", "ROLE", "URL_ENV"),
        required=True,
        help="Repeat for control and every operational database.",
    )
    arguments = parser.parse_args()
    for label, role, url_environment_name in arguments.database:
        database_url = os.environ.get(url_environment_name, "")
        if not database_url:
            parser.error(f"{url_environment_name} is not set")
        archive, metadata = create_backup(database_url, arguments.output, label, role)
        print(f"Created verified backup {archive.name} with {metadata.name}.")


if __name__ == "__main__":
    main()
