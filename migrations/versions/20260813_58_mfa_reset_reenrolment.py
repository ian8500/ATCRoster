"""Add forced MFA re-enrolment state.

Revision ID: 20260813_58
Revises: 20260812_57
"""
import os
from alembic import op
import sqlalchemy as sa

revision = "20260813_58"
down_revision = "20260812_57"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "mfa_credential" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("mfa_credential")}
    if "reset_required" not in columns:
        with op.batch_alter_table("mfa_credential") as batch:
            batch.add_column(sa.Column("reset_required", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "mfa_credential" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("mfa_credential")}
    if "reset_required" in columns:
        with op.batch_alter_table("mfa_credential") as batch:
            batch.drop_column("reset_required")
