"""add position controller time allowance

Revision ID: 20260731_32
Revises: 20260731_31
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260731_32"
down_revision = "20260731_31"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    with op.batch_alter_table("operational_position") as batch:
        batch.add_column(
            sa.Column(
                "maximum_session_duration_minutes",
                sa.Integer(),
                nullable=False,
                server_default="120",
            )
        )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    with op.batch_alter_table("operational_position") as batch:
        batch.drop_column("maximum_session_duration_minutes")
