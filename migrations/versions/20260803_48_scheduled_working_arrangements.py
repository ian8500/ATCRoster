"""Add scheduled working-arrangement metadata.

Revision ID: 20260803_48
Revises: 20260803_47
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_48"
down_revision = "20260803_47"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "staff_pattern_assignment" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("staff_pattern_assignment")}
    with op.batch_alter_table("staff_pattern_assignment") as batch:
        if "change_type" not in existing:
            batch.add_column(sa.Column(
                "change_type", sa.String(40), nullable=False,
                server_default="WORK_PATTERN_CHANGE",
            ))
        if "contracted_minutes_per_week" not in existing:
            batch.add_column(sa.Column("contracted_minutes_per_week", sa.Integer()))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "staff_pattern_assignment" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("staff_pattern_assignment")}
    with op.batch_alter_table("staff_pattern_assignment") as batch:
        for name in ("contracted_minutes_per_week", "change_type"):
            if name in existing:
                batch.drop_column(name)
