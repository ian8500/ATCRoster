"""Add configurable briefing instruction message types.

Revision ID: 20260728_23
Revises: 20260728_22
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260728_23"
down_revision = "20260728_22"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "briefing_message_type" not in tables:
        op.create_table(
            "briefing_message_type",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(80), nullable=False),
            sa.Column(
                "active", sa.Boolean(), nullable=False,
                server_default=sa.true(),
            ),
            sa.Column(
                "display_order", sa.Integer(), nullable=False,
                server_default="0",
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "unit_id", "name",
                name="uq_briefing_message_type_name",
            ),
        )
        op.create_index(
            "ix_briefing_message_type_unit_id",
            "briefing_message_type",
            ["unit_id"],
        )
        op.execute(sa.text(
            "INSERT INTO briefing_message_type "
            "(unit_id, name, active, display_order, created_at, updated_at) "
            "SELECT DISTINCT unit_id, 'General instruction', "
            "true, 10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP FROM staff"
        ))
    columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("briefing_item")
    }
    if "message_type_id" not in columns:
        op.add_column(
            "briefing_item",
            sa.Column("message_type_id", sa.Integer(), nullable=True),
        )
        op.create_index(
            "ix_briefing_item_message_type_id",
            "briefing_item",
            ["message_type_id"],
        )
    if "message_type_name" not in columns:
        op.add_column(
            "briefing_item",
            sa.Column(
                "message_type_name", sa.String(80),
                nullable=False, server_default="",
            ),
        )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    tables = set(inspect(op.get_bind()).get_table_names())
    if "briefing_item" in tables:
        columns = {
            column["name"]
            for column in inspect(op.get_bind()).get_columns("briefing_item")
        }
        if "message_type_name" in columns:
            op.drop_column("briefing_item", "message_type_name")
        if "message_type_id" in columns:
            op.drop_index(
                "ix_briefing_item_message_type_id",
                table_name="briefing_item",
            )
            op.drop_column("briefing_item", "message_type_id")
    if "briefing_message_type" in tables:
        op.drop_table("briefing_message_type")
