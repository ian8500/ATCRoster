#!/usr/bin/env python3
"""Report baseline/override migration classifications without changing data."""

from __future__ import annotations

import argparse
import os
import re

from sqlalchemy import create_engine, inspect, text


SECRET_NAME = re.compile(r"ATCROSTER_UNIT_([1-9][0-9]*)_DATABASE_URL")


def classification_summary(database_url: str) -> tuple[dict[str, int], list[int]]:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("assignment")
            }
            required = {"generated_code", "override_code", "override_type"}
            if not required.issubset(columns):
                raise RuntimeError("Assignment baseline/override migration is not installed.")
            summary = dict(connection.execute(text(
                "SELECT coalesce(override_type, 'GENERATED_BASELINE'), count(*) "
                "FROM assignment GROUP BY 1 ORDER BY 1"
            )).all())
            uncertain = list(connection.execute(text(
                "SELECT id FROM assignment "
                "WHERE override_type = 'MIGRATED_UNCERTAIN' ORDER BY id"
            )).scalars())
            return summary, uncertain
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--fail-on-uncertain", action="store_true")
    args = parser.parse_args()
    found = False
    uncertain_total = 0
    for secret_name, database_url in sorted(os.environ.items()):
        match = SECRET_NAME.fullmatch(secret_name)
        if not match or not database_url:
            continue
        found = True
        unit_id = int(match.group(1))
        summary, uncertain = classification_summary(database_url)
        uncertain_total += len(uncertain)
        print(f"Unit {unit_id}: {summary}")
        if args.details and uncertain:
            print(f"Unit {unit_id} uncertain assignment ids: {uncertain}")
    if not found:
        raise SystemExit("No unit operational database secrets were found.")
    if args.fail_on_uncertain and uncertain_total:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
