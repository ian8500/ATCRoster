"""Add effective watch-transfer alignment metadata.

Revision ID: 20260803_47
Revises: 20260803_46
"""

import os
from alembic import op
import sqlalchemy as sa

revision = "20260803_47"
down_revision = "20260803_46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "staff_watch_history" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("staff_watch_history")}
    columns = (
        sa.Column("effective_to", sa.Date()),
        sa.Column("reason", sa.String(500), nullable=False, server_default=""),
        sa.Column("alignment_mode", sa.String(40), nullable=False, server_default="ALIGN_WITH_DESTINATION_WATCH"),
        sa.Column("starting_cycle_day", sa.Integer()),
        sa.Column("pattern_anchor", sa.Date()),
    )
    with op.batch_alter_table("staff_watch_history") as batch:
        for column in columns:
            if column.name not in existing:
                batch.add_column(column)


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "staff_watch_history" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("staff_watch_history")}
    with op.batch_alter_table("staff_watch_history") as batch:
        for name in ("pattern_anchor", "starting_cycle_day", "alignment_mode", "reason", "effective_to"):
            if name in existing:
                batch.drop_column(name)
