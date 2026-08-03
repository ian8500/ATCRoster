"""Add generated baseline and editor override assignment values.

Revision ID: 20260803_42
Revises: 20260803_41

Migration classification is deliberately conservative. Only legacy rows whose
source is ``auto`` and whose note is ``pattern`` or ``generated watch coverage``
are treated as generated baselines. Every other displayed value is retained as
an override; unfamiliar provenance is marked ``MIGRATED_UNCERTAIN``.
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_42"
down_revision = "20260803_41"
branch_labels = None
depends_on = None


NEW_COLUMNS = (
    sa.Column("generated_code", sa.String(10)),
    sa.Column("override_code", sa.String(10)),
    sa.Column("generated_from_pattern_id", sa.Integer()),
    sa.Column("generated_from_pattern_day_index", sa.Integer()),
    sa.Column("generated_at", sa.DateTime()),
    sa.Column("generation_event_id", sa.Integer()),
    sa.Column("generation_version", sa.String(40)),
    sa.Column("override_type", sa.String(40)),
    sa.Column("override_reason", sa.String(500), nullable=False, server_default=""),
    sa.Column("override_by_user_id", sa.Integer()),
    sa.Column("override_at", sa.DateTime()),
)


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "assignment" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("assignment")}
    missing = [column for column in NEW_COLUMNS if column.name not in columns]
    if missing:
        with op.batch_alter_table("assignment") as batch:
            for column in missing:
                batch.add_column(column)

    has_source = "source" in columns
    has_note = "note" in columns
    source = "coalesce(source, '<null>')" if has_source else "'<null>'"
    note = "lower(trim(coalesce(note, '')))" if has_note else "''"
    generated = (
        f"source = 'auto' AND {note} IN ('pattern', 'generated watch coverage')"
        if has_source and has_note
        else "0 = 1"
    )
    manual = "source = 'manual'" if has_source else "0 = 1"
    request = "source = 'request'" if has_source else "0 = 1"
    absence = (
        f"source IN ('leave', 'sickness') OR {note} IN "
        "('leave', 'annual leave', 'sickness')"
        if has_source
        else f"{note} IN ('leave', 'annual leave', 'sickness')"
    )
    bind.execute(sa.text(
        "UPDATE assignment SET "
        f"generated_code = CASE WHEN {generated} THEN code ELSE NULL END, "
        f"override_code = CASE WHEN {generated} THEN NULL ELSE code END, "
        f"generated_at = CASE WHEN {generated} THEN CURRENT_TIMESTAMP ELSE NULL END, "
        f"generation_version = CASE WHEN {generated} THEN 'legacy-migration-v1' ELSE NULL END, "
        f"override_type = CASE WHEN {generated} THEN NULL "
        f"WHEN {manual} THEN 'MIGRATED_MANUAL' "
        f"WHEN {request} THEN 'MIGRATED_REQUEST' "
        f"WHEN {absence} THEN 'MIGRATED_ABSENCE' "
        "ELSE 'MIGRATED_UNCERTAIN' END, "
        "override_reason = CASE WHEN "
        f"{generated} THEN '' ELSE 'Preserved from legacy assignment source=' || {source} END"
    ))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "assignment" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("assignment")}
    with op.batch_alter_table("assignment") as batch:
        for column in reversed(NEW_COLUMNS):
            if column.name in columns:
                batch.drop_column(column.name)
