"""Add persistent handover operational and equipment state.

Revision ID: 20260808_55
Revises: 20260808_54
"""

import os
from alembic import op
import sqlalchemy as sa

revision = "20260808_55"
down_revision = "20260808_54"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "handover_operational_state" not in existing:
        op.create_table(
            "handover_operational_state",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("runway_in_use", sa.String(40), nullable=False, server_default=""),
            sa.Column("metar_icao", sa.String(4), nullable=False, server_default=""),
            sa.Column("updated_by_id", sa.Integer()),
            sa.Column("updated_by_name", sa.String(80), nullable=False, server_default=""),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("unit_id", name="uq_handover_operational_state_unit"),
        )
        op.create_index("ix_handover_operational_state_unit_id", "handover_operational_state", ["unit_id"], unique=True)
    if "handover_equipment" not in existing:
        op.create_table(
            "handover_equipment",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("status", sa.String(10), nullable=False, server_default="green"),
            sa.Column("note", sa.String(240), nullable=False, server_default=""),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("updated_by_id", sa.Integer()),
            sa.Column("updated_by_name", sa.String(80), nullable=False, server_default=""),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_handover_equipment_unit_id", "handover_equipment", ["unit_id"])


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for table in ("handover_equipment", "handover_operational_state"):
        if table in existing:
            op.drop_table(table)
