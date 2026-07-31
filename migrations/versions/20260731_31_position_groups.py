"""add reusable live-position display groups

Revision ID: 20260731_31
Revises: 20260731_30
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260731_31"
down_revision = "20260731_30"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    op.create_table(
        "operational_position_group",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("unit_id", "name", name="uq_position_group_unit_name"),
    )
    op.create_index(
        "ix_operational_position_group_unit_id",
        "operational_position_group",
        ["unit_id"],
        unique=False,
    )
    connection = op.get_bind()
    existing = connection.execute(
        sa.text(
            "SELECT unit_id, group_name, MIN(display_order) AS display_order "
            "FROM operational_position WHERE group_name <> '' "
            "GROUP BY unit_id, group_name"
        )
    )
    groups = sa.table(
        "operational_position_group",
        sa.column("unit_id", sa.Integer()),
        sa.column("name", sa.String(80)),
        sa.column("display_order", sa.Integer()),
        sa.column("is_active", sa.Boolean()),
    )
    rows = [
        {
            "unit_id": row.unit_id,
            "name": row.group_name,
            "display_order": row.display_order,
            "is_active": True,
        }
        for row in existing
    ]
    if rows:
        connection.execute(groups.insert(), rows)


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    op.drop_index(
        "ix_operational_position_group_unit_id",
        table_name="operational_position_group",
    )
    op.drop_table("operational_position_group")
