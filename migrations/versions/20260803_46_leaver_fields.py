"""Add effective-dated unit leaver fields.

Revision ID: 20260803_46
Revises: 20260803_45
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_46"
down_revision = "20260803_45"
branch_labels = None
depends_on = None


COLUMNS = (
    sa.Column("final_unit_date", sa.Date()),
    sa.Column("final_operational_duty_date", sa.Date()),
    sa.Column("employment_end_date", sa.Date()),
    sa.Column("leaving_reason_category", sa.String(40), nullable=False, server_default=""),
    sa.Column("leaving_notes", sa.Text(), nullable=False, server_default=""),
)


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "staff" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("staff")}
    with op.batch_alter_table("staff") as batch:
        for column in COLUMNS:
            if column.name not in existing:
                batch.add_column(column)


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "staff" not in sa.inspect(bind).get_table_names():
        return
    existing = {column["name"] for column in sa.inspect(bind).get_columns("staff")}
    with op.batch_alter_table("staff") as batch:
        for column in reversed(COLUMNS):
            if column.name in existing:
                batch.drop_column(column.name)
