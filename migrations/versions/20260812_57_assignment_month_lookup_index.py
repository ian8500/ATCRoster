"""Index assignment month-range lookups by tenant.

Revision ID: 20260812_57
Revises: 20260808_56
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260812_57"
down_revision = "20260808_56"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "assignment" not in inspector.get_table_names():
        return
    indexes = {index["name"] for index in inspector.get_indexes("assignment")}
    if "ix_assignment_unit_day" not in indexes:
        op.create_index("ix_assignment_unit_day", "assignment", ["unit_id", "day"])


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "assignment" not in inspector.get_table_names():
        return
    indexes = {index["name"] for index in inspector.get_indexes("assignment")}
    if "ix_assignment_unit_day" in indexes:
        op.drop_index("ix_assignment_unit_day", table_name="assignment")
