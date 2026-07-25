#!/usr/bin/env python3
"""Repeatable 30-unit database scale smoke test.

This is deliberately independent of production data. It creates a temporary
SQLite database containing 30 fictitious units, 40 people per unit and 90 days
of assignments, then exercises the tenant-scoped month query.
"""
from __future__ import annotations

import argparse
import sqlite3
import tempfile
import time
from pathlib import Path


def run(units: int = 30, people: int = 40, days: int = 90) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="atcroster-scale-") as directory:
        database = Path(directory) / "scale.db"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE assignment (
              id INTEGER PRIMARY KEY,
              unit_id INTEGER NOT NULL,
              staff_id INTEGER NOT NULL,
              duty_day INTEGER NOT NULL,
              shift_code TEXT NOT NULL
            );
            CREATE INDEX ix_assignment_unit_day
              ON assignment(unit_id, duty_day);
            """
        )
        started = time.perf_counter()
        rows = (
            (unit, unit * 1000 + person, day, ("M", "A", "N", "OFF")[day % 4])
            for unit in range(1, units + 1)
            for person in range(1, people + 1)
            for day in range(days)
        )
        connection.executemany(
            "INSERT INTO assignment(unit_id, staff_id, duty_day, shift_code) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        seed_seconds = time.perf_counter() - started

        query_started = time.perf_counter()
        result = connection.execute(
            "SELECT staff_id, duty_day, shift_code FROM assignment "
            "WHERE unit_id = ? AND duty_day BETWEEN ? AND ?",
            (17, 31, 61),
        ).fetchall()
        query_seconds = time.perf_counter() - query_started
        assert len(result) == people * 31
        assert all(row[0] // 1000 == 17 for row in result)
        connection.close()
        return {
            "assignments": float(units * people * days),
            "seed_seconds": seed_seconds,
            "tenant_month_query_ms": query_seconds * 1000,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=30)
    parser.add_argument("--people", type=int, default=40)
    parser.add_argument("--days", type=int, default=90)
    arguments = parser.parse_args()
    outcome = run(arguments.units, arguments.people, arguments.days)
    print(
        f"{int(outcome['assignments'])} assignments; "
        f"seed {outcome['seed_seconds']:.3f}s; "
        f"tenant month query {outcome['tenant_month_query_ms']:.3f}ms"
    )
