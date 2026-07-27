"""Allow requesters to dismiss completed shift requests from their list.

Revision ID: 20260727_15
Revises: 20260727_14
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260727_15"
down_revision = "20260727_14"
branch_labels = None
depends_on = None


def upgrade():
    role = os.environ.get("ATCROSTER_SCHEMA_ROLE", "combined")
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if role not in {"combined", "operational"} or "shift_request" not in tables:
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("shift_request")
    }
    if "dismissed_by_requester_at" not in columns:
        with op.batch_alter_table("shift_request") as batch:
            batch.add_column(sa.Column("dismissed_by_requester_at", sa.DateTime()))


def downgrade():
    raise RuntimeError("Shift-request audit visibility history cannot be safely removed.")
