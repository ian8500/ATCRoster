"""Add optional, rollback-safe briefing module.

Revision ID: 20260728_21
Revises: 20260727_20
"""

import os

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260728_21"
down_revision = "20260727_20"
branch_labels = None
depends_on = None


TABLES = (
    "briefing_item",
    "briefing_delivery",
    "briefing_audit",
    "briefing_assurance_run",
)


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    existing = set(inspect(op.get_bind()).get_table_names())
    if "briefing_item" not in existing:
        op.create_table(
            "briefing_item",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("title", sa.String(160), nullable=False),
            sa.Column("body", sa.Text(), nullable=False, server_default=""),
            sa.Column("effective_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("mandatory", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("priority", sa.String(20), nullable=False, server_default="routine"),
            sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
            sa.Column("target_json", sa.Text(), nullable=False, server_default='{"scope":"all"}'),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("original_filename", sa.String(255), nullable=False, server_default=""),
            sa.Column("stored_filename", sa.String(255), nullable=False, server_default=""),
            sa.Column("content_type", sa.String(120), nullable=False, server_default=""),
            sa.Column("content_sha256", sa.String(64), nullable=False, server_default=""),
            sa.Column("created_by_id", sa.Integer(), nullable=False),
            sa.Column("created_by_name", sa.String(80), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("published_at", sa.DateTime()),
            sa.Column("withdrawn_at", sa.DateTime()),
        )
        op.create_index("ix_briefing_item_unit_id", "briefing_item", ["unit_id"])
        op.create_index("ix_briefing_item_effective_at", "briefing_item", ["effective_at"])
        op.create_index("ix_briefing_item_expires_at", "briefing_item", ["expires_at"])
        op.create_index("ix_briefing_item_status", "briefing_item", ["status"])
    if "briefing_delivery" not in existing:
        op.create_table(
            "briefing_delivery",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("briefing_id", sa.Integer(), sa.ForeignKey("briefing_item.id"), nullable=False),
            sa.Column("recipient_id", sa.Integer(), nullable=False),
            sa.Column("recipient_name", sa.String(80), nullable=False),
            sa.Column("delivered_at", sa.DateTime(), nullable=False),
            sa.Column("first_opened_at", sa.DateTime()),
            sa.Column("last_opened_at", sa.DateTime()),
            sa.Column("active_view_seconds", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("acknowledged_at", sa.DateTime()),
            sa.Column("acknowledged_version", sa.Integer()),
            sa.UniqueConstraint(
                "unit_id", "briefing_id", "recipient_id",
                name="uq_briefing_delivery_recipient",
            ),
        )
        op.create_index("ix_briefing_delivery_unit_id", "briefing_delivery", ["unit_id"])
        op.create_index("ix_briefing_delivery_briefing_id", "briefing_delivery", ["briefing_id"])
        op.create_index("ix_briefing_delivery_recipient_id", "briefing_delivery", ["recipient_id"])
    if "briefing_audit" not in existing:
        op.create_table(
            "briefing_audit",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("briefing_id", sa.Integer()),
            sa.Column("delivery_id", sa.Integer()),
            sa.Column("actor_id", sa.Integer(), nullable=False),
            sa.Column("actor_name", sa.String(80), nullable=False),
            sa.Column("event_type", sa.String(40), nullable=False),
            sa.Column("occurred_at", sa.DateTime(), nullable=False),
            sa.Column("detail_json", sa.Text(), nullable=False, server_default="{}"),
        )
        op.create_index("ix_briefing_audit_unit_id", "briefing_audit", ["unit_id"])
        op.create_index("ix_briefing_audit_briefing_id", "briefing_audit", ["briefing_id"])
        op.create_index("ix_briefing_audit_delivery_id", "briefing_audit", ["delivery_id"])
        op.create_index("ix_briefing_audit_event_type", "briefing_audit", ["event_type"])
        op.create_index("ix_briefing_audit_occurred_at", "briefing_audit", ["occurred_at"])
    if "briefing_assurance_run" not in existing:
        op.create_table(
            "briefing_assurance_run",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("operational_date", sa.Date(), nullable=False),
            sa.Column("run_by_id", sa.Integer(), nullable=False),
            sa.Column("run_by_name", sa.String(80), nullable=False),
            sa.Column("run_at", sa.DateTime(), nullable=False),
            sa.Column("result_json", sa.Text(), nullable=False),
        )
        op.create_index("ix_briefing_assurance_run_unit_id", "briefing_assurance_run", ["unit_id"])
        op.create_index("ix_briefing_assurance_run_operational_date", "briefing_assurance_run", ["operational_date"])


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    existing = set(inspect(op.get_bind()).get_table_names())
    for table in reversed(TABLES):
        if table in existing:
            op.drop_table(table)
