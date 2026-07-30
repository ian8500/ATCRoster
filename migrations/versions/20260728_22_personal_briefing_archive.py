"""Add personal briefing archive and soft deletion.

Revision ID: 20260728_22
Revises: 20260728_21
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260728_22"
down_revision = "20260728_21"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = inspect(op.get_bind())
    if "briefing_delivery" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("briefing_delivery")
    }
    if "archived_at" not in columns:
        op.add_column(
            "briefing_delivery",
            sa.Column("archived_at", sa.DateTime(), nullable=True),
        )
    if "deleted_at" not in columns:
        op.add_column(
            "briefing_delivery",
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
        )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = inspect(op.get_bind())
    if "briefing_delivery" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("briefing_delivery")
    }
    if "deleted_at" in columns:
        op.drop_column("briefing_delivery", "deleted_at")
    if "archived_at" in columns:
        op.drop_column("briefing_delivery", "archived_at")
