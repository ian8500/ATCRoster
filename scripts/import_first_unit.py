#!/usr/bin/env python3
"""Resumable legacy-database importer with before/after reconciliation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, inspect, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--checkpoint", default=".import-first-unit.json")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    state = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"completed": []}
    source = create_engine(args.source_url)
    target = create_engine(args.target_url)
    source_meta, target_meta = MetaData(), MetaData()
    source_meta.reflect(bind=source)
    target_meta.reflect(bind=target)
    report = {}
    for table_name in sorted(set(source_meta.tables) & set(target_meta.tables)):
        if table_name in {"unit", "alembic_version"}:
            continue
        src, dst = source_meta.tables[table_name], target_meta.tables[table_name]
        with source.connect() as src_conn, target.begin() as dst_conn:
            before = src_conn.execute(select(src)).mappings().all()
            if table_name not in state["completed"]:
                target_columns = set(dst.c.keys())
                for raw in before:
                    row = {key: value for key, value in raw.items() if key in target_columns}
                    if "unit_id" in target_columns:
                        row["unit_id"] = args.unit_id
                    pk_names = [col.name for col in dst.primary_key.columns]
                    exists = False
                    if pk_names and all(row.get(key) is not None for key in pk_names):
                        exists = dst_conn.execute(
                            select(dst).where(*[dst.c[key] == row[key] for key in pk_names])
                        ).first() is not None
                    if not exists:
                        dst_conn.execute(dst.insert().values(**row))
                state["completed"].append(table_name)
                checkpoint_path.write_text(json.dumps(state, indent=2))
            after_count = dst_conn.execute(
                select(dst).where(dst.c.unit_id == args.unit_id)
                if "unit_id" in dst.c else select(dst)
            ).fetchall()
            report[table_name] = {"before": len(before), "after": len(after_count)}
    print(json.dumps({"unit_id": args.unit_id, "tables": report}, indent=2))


if __name__ == "__main__":
    main()
