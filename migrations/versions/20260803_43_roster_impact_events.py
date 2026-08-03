"""Add roster-impact event audit and protected-period exceptions.

Revision ID: 20260803_43
Revises: 20260803_42
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_43"
down_revision = "20260803_42"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "roster_impact_event" not in tables:
        op.create_table(
            "roster_impact_event",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), sa.ForeignKey("unit.id"), nullable=False),
            sa.Column("event_type", sa.String(50), nullable=False),
            sa.Column("effective_from", sa.Date(), nullable=False),
            sa.Column("effective_to", sa.Date()),
            sa.Column("staff_ids_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("watch_ids_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("rebuild_baseline", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("recalculate_coverage", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("preserve_overrides", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("reason", sa.String(500), nullable=False, server_default=""),
            sa.Column("triggered_by_user_id", sa.Integer()),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("protected_from", sa.Date()),
            sa.Column("protected_to", sa.Date()),
            sa.Column("automatic_from", sa.Date()),
            sa.Column("automatic_to", sa.Date()),
            sa.Column("result_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime()),
            sa.UniqueConstraint("unit_id", "id", name="uq_roster_impact_event_unit_id"),
            sa.CheckConstraint("status IN ('PROCESSING','COMPLETED')", name="ck_roster_impact_event_status"),
            sa.CheckConstraint("effective_to IS NULL OR effective_to >= effective_from", name="ck_roster_impact_event_range"),
        )
        op.create_index("ix_roster_impact_event_unit_id", "roster_impact_event", ["unit_id"])
        op.create_index("ix_roster_impact_event_event_type", "roster_impact_event", ["event_type"])
        op.create_index("ix_roster_impact_event_effective_from", "roster_impact_event", ["effective_from"])
        op.create_index("ix_roster_impact_event_status", "roster_impact_event", ["status"])

    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "roster_impact_exception" not in tables:
        op.create_table(
            "roster_impact_exception",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.Integer(), nullable=False),
            sa.Column("staff_id", sa.Integer()),
            sa.Column("watch_id", sa.Integer(), sa.ForeignKey("watch.id")),
            sa.Column("effective_from", sa.Date(), nullable=False),
            sa.Column("effective_to", sa.Date(), nullable=False),
            sa.Column("exception_type", sa.String(40), nullable=False),
            sa.Column("severity", sa.String(20), nullable=False),
            sa.Column("description", sa.String(1000), nullable=False),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("resolved_by_user_id", sa.Integer()),
            sa.Column("resolved_at", sa.DateTime()),
            sa.Column("resolution_note", sa.String(1000), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["unit_id", "event_id"], ["roster_impact_event.unit_id", "roster_impact_event.id"], name="fk_roster_impact_exception_event_unit", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["unit_id", "staff_id"], ["staff.unit_id", "staff.id"], name="fk_roster_impact_exception_staff_unit"),
            sa.CheckConstraint("effective_to >= effective_from", name="ck_roster_impact_exception_range"),
            sa.CheckConstraint("severity IN ('INFO','WARNING','CRITICAL')", name="ck_roster_impact_exception_severity"),
            sa.CheckConstraint("status IN ('OPEN','ACKNOWLEDGED','RESOLVED','DISMISSED')", name="ck_roster_impact_exception_status"),
        )
        for column in ("unit_id", "event_id", "staff_id", "watch_id", "effective_from", "exception_type", "status"):
            op.create_index(f"ix_roster_impact_exception_{column}", "roster_impact_exception", [column])


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "roster_impact_exception" in tables:
        op.drop_table("roster_impact_exception")
    if "roster_impact_event" in tables:
        op.drop_table("roster_impact_event")
