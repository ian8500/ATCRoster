"""Add configurable protected roster horizon.

Revision ID: 20260803_41
Revises: 20260803_40
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_41"
down_revision = "20260803_40"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "operational":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "unit" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("unit")}
    if "protected_roster_months_ahead" not in columns:
        with op.batch_alter_table("unit") as batch:
            batch.add_column(sa.Column(
                "protected_roster_months_ahead",
                sa.Integer(),
                nullable=False,
                server_default="2",
            ))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "operational":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "unit" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("unit")}
    if "protected_roster_months_ahead" in columns:
        with op.batch_alter_table("unit") as batch:
            batch.drop_column("protected_roster_months_ahead")
