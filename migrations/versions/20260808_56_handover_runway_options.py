"""Add configurable runway options to the handover operational state.

Revision ID: 20260808_56
Revises: 20260808_55
"""

import os

from alembic import op
import sqlalchemy as sa

revision = "20260808_56"
down_revision = "20260808_55"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "handover_operational_state" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("handover_operational_state")}
    if "runway_options_json" not in columns:
        op.add_column(
            "handover_operational_state",
            sa.Column("runway_options_json", sa.Text(), nullable=False, server_default="[]"),
        )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "handover_operational_state" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("handover_operational_state")}
    if "runway_options_json" in columns:
        op.drop_column("handover_operational_state", "runway_options_json")
