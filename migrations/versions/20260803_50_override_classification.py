"""Add configurable roster override classifications.

Revision ID: 20260803_50
Revises: 20260803_49
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_50"
down_revision = "20260803_49"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "unit" in tables:
        existing = {c["name"] for c in sa.inspect(bind).get_columns("unit")}
        if "preserve_redundant_overrides" not in existing:
            with op.batch_alter_table("unit") as batch:
                batch.add_column(sa.Column(
                    "preserve_redundant_overrides", sa.Boolean(),
                    nullable=False, server_default=sa.true(),
                ))
    if "assignment" in tables:
        existing = {c["name"] for c in sa.inspect(bind).get_columns("assignment")}
        with op.batch_alter_table("assignment") as batch:
            if "override_classification" not in existing:
                batch.add_column(sa.Column("override_classification", sa.String(50)))
            if "override_classified_at" not in existing:
                batch.add_column(sa.Column("override_classified_at", sa.Date()))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "assignment" in tables:
        existing = {c["name"] for c in sa.inspect(bind).get_columns("assignment")}
        with op.batch_alter_table("assignment") as batch:
            for name in ("override_classified_at", "override_classification"):
                if name in existing:
                    batch.drop_column(name)
    if "unit" in tables:
        existing = {c["name"] for c in sa.inspect(bind).get_columns("unit")}
        if "preserve_redundant_overrides" in existing:
            with op.batch_alter_table("unit") as batch:
                batch.drop_column("preserve_redundant_overrides")
