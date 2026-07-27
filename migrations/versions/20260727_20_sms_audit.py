"""Add airport-scoped SMS delivery audit."""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260727_20"
down_revision = "20260727_19"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    if "sms_audit" in inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "sms_audit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("sent_at", sa.DateTime(), nullable=False),
        sa.Column("sent_by_staff_id", sa.Integer(), nullable=False),
        sa.Column("sent_by_name", sa.String(length=80), nullable=False),
        sa.Column("sender_number", sa.String(length=20), nullable=False),
        sa.Column("recipient_number", sa.String(length=20), nullable=False),
        sa.Column("recipient_label", sa.String(length=120), nullable=False),
        sa.Column("message_type", sa.String(length=30), nullable=False),
        sa.Column("message_content", sa.Text(), nullable=False),
        sa.Column(
            "provider_message_id", sa.String(length=64),
            nullable=False, server_default="",
        ),
    )
    op.create_index("ix_sms_audit_unit_id", "sms_audit", ["unit_id"])
    op.create_index("ix_sms_audit_sent_at", "sms_audit", ["sent_at"])
    op.create_index(
        "ix_sms_audit_sent_by_staff_id",
        "sms_audit", ["sent_by_staff_id"],
    )


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    if "sms_audit" in inspect(op.get_bind()).get_table_names():
        op.drop_table("sms_audit")
