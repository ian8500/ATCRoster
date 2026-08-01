"""add weekly position controller time matrix

Revision ID: 20260801_33
Revises: 20260731_32
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260801_33"
down_revision = "20260731_32"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    with op.batch_alter_table("operational_position") as batch:
        batch.add_column(
            sa.Column(
                "maximum_session_duration_matrix_json",
                sa.Text(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    with op.batch_alter_table("operational_position") as batch:
        batch.drop_column("maximum_session_duration_matrix_json")
