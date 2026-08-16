"""Allow platform-wide SuperAdmin audit events.

Revision ID: 20260816_61
Revises: 20260815_60
"""

import os

from alembic import op
import sqlalchemy as sa

revision = "20260816_61"
down_revision = "20260815_60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "operational":
        return
    inspector = sa.inspect(op.get_bind())
    if "super_admin_audit" not in inspector.get_table_names():
        return
    unit_column = next(
        column
        for column in inspector.get_columns("super_admin_audit")
        if column["name"] == "unit_id"
    )
    if unit_column["nullable"]:
        return
    with op.batch_alter_table("super_admin_audit") as batch_op:
        batch_op.alter_column(
            "unit_id",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    # Global audit rows cannot be represented by the old constraint without
    # deleting valid audit history, so the safe downgrade retains nullability.
    pass
