"""Add explicit roster-period lifecycle records.

Revision ID: 20260803_51
Revises: 20260803_50
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_51"
down_revision = "20260803_50"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "roster_period" in tables:
        return
    unit_constraints = (
        (sa.ForeignKeyConstraint(["unit_id"], ["unit.id"], ondelete="CASCADE"),)
        if "unit" in tables else ()
    )
    op.create_table(
        "roster_period",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("month", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("generated_at", sa.DateTime()),
        sa.Column("generated_by_user_id", sa.Integer()),
        sa.Column("generation_method", sa.String(40), nullable=False, server_default="AUTOMATIC"),
        sa.Column("generation_version", sa.String(40), nullable=False, server_default=""),
        sa.Column("notes", sa.String(1000), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        *unit_constraints,
        sa.UniqueConstraint("unit_id", "year", "month", name="uq_roster_period_unit_month"),
        sa.CheckConstraint("month BETWEEN 1 AND 12", name="ck_roster_period_month"),
        sa.CheckConstraint(
            "status IN ('CURRENT','PROTECTED','FUTURE_AUTOMATIC','HISTORICAL','CLOSED')",
            name="ck_roster_period_status",
        ),
    )
    op.create_index("ix_roster_period_unit_id", "roster_period", ["unit_id"])
    op.create_index("ix_roster_period_status", "roster_period", ["status"])


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    if "roster_period" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("roster_period")
