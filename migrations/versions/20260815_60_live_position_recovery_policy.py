"""Add tenant-scoped cumulative position-time recovery policy.

Revision ID: 20260815_60
Revises: 20260814_59
"""
import os

from alembic import op
import sqlalchemy as sa

revision = "20260815_60"
down_revision = "20260814_59"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "live_position_recovery_policy" in inspector.get_table_names():
        return
    op.create_table(
        "live_position_recovery_policy",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("base_break_minutes", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("escalation_after_minutes", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("extra_break_minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("escalation_interval_minutes", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("escalation_cap_minutes", sa.Integer(), nullable=False, server_default="240"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by_id", sa.Integer()),
        sa.ForeignKeyConstraint(["unit_id"], ["unit.id"]),
        sa.ForeignKeyConstraint(["updated_by_id"], ["staff.id"]),
        sa.UniqueConstraint("unit_id", name="uq_live_position_recovery_policy_unit_id"),
        sa.CheckConstraint("base_break_minutes >= 1 AND base_break_minutes <= 240", name="ck_position_recovery_base_break"),
        sa.CheckConstraint("escalation_after_minutes >= 1 AND escalation_after_minutes <= 480", name="ck_position_recovery_threshold"),
        sa.CheckConstraint("extra_break_minutes >= 0 AND extra_break_minutes <= 120", name="ck_position_recovery_extra_break"),
        sa.CheckConstraint("escalation_interval_minutes >= 1 AND escalation_interval_minutes <= 240", name="ck_position_recovery_interval"),
        sa.CheckConstraint("escalation_cap_minutes >= escalation_after_minutes AND escalation_cap_minutes <= 720", name="ck_position_recovery_cap"),
    )
    op.create_index(
        "ix_live_position_recovery_policy_unit_id",
        "live_position_recovery_policy",
        ["unit_id"],
        unique=True,
    )


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    if "live_position_recovery_policy" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("live_position_recovery_policy")
