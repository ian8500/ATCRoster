"""Add weekend defaults and date-specific staffing requirements."""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260727_17"
down_revision = "20260727_16"
branch_labels = None
depends_on = None

WEEKEND_COLUMNS = [
    f"req_{day}_{code}"
    for day in ("sat", "sun")
    for code in ("m", "d", "a", "n")
]


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "requirement" in tables:
        columns = {
            column["name"]
            for column in inspector.get_columns("requirement")
        }
        with op.batch_alter_table("requirement") as batch:
            for name in WEEKEND_COLUMNS:
                if name not in columns:
                    batch.add_column(
                        sa.Column(
                            name, sa.Integer(), nullable=False,
                            server_default="0",
                        )
                    )
        for day in ("sat", "sun"):
            for code in ("m", "d", "a", "n"):
                bind.execute(sa.text(
                    f"UPDATE requirement SET req_{day}_{code} = req_{code}"
                ))

    inspector = inspect(bind)
    if "special_requirement" not in inspector.get_table_names():
        unit_column = (
            sa.Column(
                "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
                nullable=False,
            )
            if "unit" in inspector.get_table_names()
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
    inspector = inspect(op.get_bind())
    if "special_requirement" in inspector.get_table_names():
        op.drop_table("special_requirement")
    inspector = inspect(op.get_bind())
    if "requirement" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("requirement")
        }
        with op.batch_alter_table("requirement") as batch:
            for name in WEEKEND_COLUMNS:
                if name in columns:
                    batch.drop_column(name)
