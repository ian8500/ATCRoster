"""Add an append-only idempotent TOIL ledger.

Revision ID: 20260801_37
Revises: 20260801_36
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260801_37"
down_revision = "20260801_36"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    staff_uniques = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints("staff")
    }
    if ("unit_id", "id") not in staff_uniques:
        with op.batch_alter_table("staff") as batch:
            batch.create_unique_constraint("uq_staff_unit_id", ["unit_id", "id"])
    op.create_table(
        "toil_transaction",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("person_id", sa.Integer(), nullable=False, index=True),
        sa.Column("delta_half_days", sa.Integer(), nullable=False),
        sa.Column("balance_after_half_days", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(500), nullable=False),
        sa.Column("source_type", sa.String(40), nullable=False),
        sa.Column("source_id", sa.Integer()),
        sa.Column("actor_id", sa.Integer(), nullable=False),
        sa.Column("transaction_key", sa.String(64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["unit_id", "person_id"],
            ["staff.unit_id", "staff.id"],
            name="fk_toil_transaction_person_unit",
        ),
        sa.UniqueConstraint(
            "unit_id", "transaction_key", name="uq_toil_transaction_unit_key"
        ),
        sa.CheckConstraint("delta_half_days <> 0", name="ck_toil_transaction_nonzero"),
    )


def downgrade() -> None:
    op.drop_table("toil_transaction")
