"""Add optimistic roster-cell versions.

Revision ID: 20260801_36
Revises: 20260801_35
"""

from alembic import op
import sqlalchemy as sa


revision = "20260801_36"
down_revision = "20260801_35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if "assignment" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("assignment")}
    if "version" not in columns:
        with op.batch_alter_table("assignment") as batch:
            batch.add_column(
                sa.Column("version", sa.Integer(), nullable=False, server_default="1")
            )


def downgrade() -> None:
    with op.batch_alter_table("assignment") as batch:
        batch.drop_column("version")
