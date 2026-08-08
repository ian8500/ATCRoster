"""Add provider-neutral SMS audit fields and personal sender verification.

Revision ID: 20260808_53
Revises: 20260803_52
"""

import os
from alembic import op
import sqlalchemy as sa

revision = "20260808_53"
down_revision = "20260803_52"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if "sms_audit" in tables:
        columns = {column["name"] for column in inspector.get_columns("sms_audit")}
        with op.batch_alter_table("sms_audit") as batch:
            if "provider" not in columns:
                batch.add_column(sa.Column("provider", sa.String(30), nullable=False, server_default="twilio"))
            if "delivery_status" not in columns:
                batch.add_column(sa.Column("delivery_status", sa.String(30), nullable=False, server_default="submitted"))
    if "sms_sender_registration" not in tables:
        op.create_table(
            "sms_sender_registration",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("staff_id", sa.Integer(), nullable=False),
            sa.Column("number", sa.String(20), nullable=False),
            sa.Column("provider", sa.String(30), nullable=False, server_default="messagemedia"),
            sa.Column("status", sa.String(30), nullable=False, server_default="pending_dashboard_verification"),
            sa.Column("provider_identifier", sa.String(120), nullable=False, server_default=""),
            sa.Column("verification_requested_at", sa.DateTime(), nullable=False),
            sa.Column("verified_at", sa.DateTime(), nullable=True),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("unit_id", "staff_id", "number", "provider", name="uq_sms_sender_registration"),
        )
        op.create_index("ix_sms_sender_registration_unit_id", "sms_sender_registration", ["unit_id"])
        op.create_index("ix_sms_sender_registration_staff_id", "sms_sender_registration", ["staff_id"])


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "sms_sender_registration" in inspector.get_table_names():
        op.drop_table("sms_sender_registration")
