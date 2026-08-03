"""Add dated holidays and assignment locks.

Revision ID: 20260803_39
Revises: 20260802_38
"""
import os

from alembic import op
import sqlalchemy as sa

revision = "20260803_39"
down_revision = "20260802_38"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "staff" not in tables:
        return
    if "bank_holiday" not in tables:
        op.create_table(
            "bank_holiday",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("day", sa.Date(), nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("unit_id", "day", name="uq_bank_holiday_unit_day"),
        )
        op.create_index("ix_bank_holiday_unit_id", "bank_holiday", ["unit_id"])
        op.create_index("ix_bank_holiday_day", "bank_holiday", ["day"])
    if "assignment" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("assignment")}
        with op.batch_alter_table("assignment") as batch:
            if "lock_status" not in columns:
                batch.add_column(sa.Column("lock_status", sa.String(20), nullable=False, server_default="UNLOCKED"))
            if "locked_by_user_id" not in columns:
                batch.add_column(sa.Column("locked_by_user_id", sa.Integer()))
            if "locked_at" not in columns:
                batch.add_column(sa.Column("locked_at", sa.DateTime()))
            if "lock_reason" not in columns:
                batch.add_column(sa.Column("lock_reason", sa.String(250), nullable=False, server_default=""))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "assignment" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("assignment")}
        with op.batch_alter_table("assignment") as batch:
            for name in ("lock_reason", "locked_at", "locked_by_user_id", "lock_status"):
                if name in columns:
                    batch.drop_column(name)
    if "bank_holiday" in tables:
        op.drop_table("bank_holiday")
