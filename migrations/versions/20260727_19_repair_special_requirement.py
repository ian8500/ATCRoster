"""Repair missing operational special-requirement tables."""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260727_19"
down_revision = "20260727_18"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "special_requirement" in inspect(bind).get_table_names():
        return
    tables = set(inspect(bind).get_table_names())
    unit_column = (
        sa.Column(
            "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
            nullable=False,
        )
        if "unit" in tables
        else sa.Column("unit_id", sa.Integer(), nullable=False)
    )
    op.create_table(
        "special_requirement",
        sa.Column("id", sa.Integer(), primary_key=True),
        unit_column,
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "label", sa.String(length=80), nullable=False,
            server_default="",
        ),
        sa.Column(
            "req_m", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "req_d", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "req_a", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "req_n", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.UniqueConstraint(
            "unit_id", "day",
            name="uniq_unit_special_requirement_day",
        ),
    )
    op.create_index(
        "ix_special_requirement_unit_id",
        "special_requirement", ["unit_id"],
    )
    op.create_index(
        "ix_special_requirement_day",
        "special_requirement", ["day"],
    )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    if "special_requirement" in inspect(op.get_bind()).get_table_names():
        op.drop_table("special_requirement")
