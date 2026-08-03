"""Place the override-retention policy on central unit records.

Revision ID: 20260803_52
Revises: 20260803_51
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_52"
down_revision = "20260803_51"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "operational":
        return
    bind = op.get_bind()
    if "unit" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("unit")}
    if "preserve_redundant_overrides" not in existing:
        with op.batch_alter_table("unit") as batch:
            batch.add_column(sa.Column(
                "preserve_redundant_overrides", sa.Boolean(),
                nullable=False, server_default=sa.true(),
            ))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "operational":
        return
    bind = op.get_bind()
    if "unit" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("unit")}
    if "preserve_redundant_overrides" in existing:
        with op.batch_alter_table("unit") as batch:
            batch.drop_column("preserve_redundant_overrides")
