"""Add reviewable automatic roster proposals.

Revision ID: 20260803_40
Revises: 20260803_39
"""
import os

from alembic import op
import sqlalchemy as sa

revision = "20260803_40"
down_revision = "20260803_39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "staff" not in tables:
        return
    if "roster_proposal" not in tables:
        op.create_table(
            "roster_proposal",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("start_date", sa.Date(), nullable=False),
            sa.Column("end_date", sa.Date(), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("workflow_state", sa.String(20), nullable=False, server_default="draft"),
            sa.Column("objective_score", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("configuration_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("warnings_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("uncovered_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("created_by_user_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("applied_by_user_id", sa.Integer()),
            sa.Column("applied_at", sa.DateTime()),
            sa.Column("discarded_by_user_id", sa.Integer()),
            sa.Column("discarded_at", sa.DateTime()),
            sa.CheckConstraint("end_date >= start_date", name="ck_roster_proposal_date_range"),
            sa.CheckConstraint("workflow_state IN ('draft','applied','discarded')", name="ck_roster_proposal_workflow_state"),
            sa.UniqueConstraint("unit_id", "id", name="uq_roster_proposal_unit_id"),
        )
        op.create_index("ix_roster_proposal_unit_id", "roster_proposal", ["unit_id"])
        op.create_index("ix_roster_proposal_start_date", "roster_proposal", ["start_date"])
        op.create_index("ix_roster_proposal_end_date", "roster_proposal", ["end_date"])
    if "roster_proposal_assignment" not in tables:
        op.create_table(
            "roster_proposal_assignment",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("proposal_id", sa.Integer(), nullable=False),
            sa.Column("staff_id", sa.Integer(), nullable=False),
            sa.Column("day", sa.Date(), nullable=False),
            sa.Column("shift_type_id", sa.Integer(), nullable=False),
            sa.Column("shift_code", sa.String(10), nullable=False),
            sa.Column("review_state", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("score", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("explanations_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("applied_assignment_id", sa.Integer()),
            sa.Column("reviewed_by_user_id", sa.Integer()),
            sa.Column("reviewed_at", sa.DateTime()),
            sa.ForeignKeyConstraint(["unit_id", "proposal_id"], ["roster_proposal.unit_id", "roster_proposal.id"], name="fk_proposal_assignment_proposal_unit", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["unit_id", "staff_id"], ["staff.unit_id", "staff.id"], name="fk_proposal_assignment_staff_unit"),
            sa.ForeignKeyConstraint(["unit_id", "shift_type_id"], ["shift_type.unit_id", "shift_type.id"], name="fk_proposal_assignment_shift_unit"),
            sa.CheckConstraint("review_state IN ('pending','accepted','rejected','applied')", name="ck_proposal_assignment_review_state"),
            sa.UniqueConstraint("unit_id", "proposal_id", "staff_id", "day", name="uq_proposal_assignment_staff_day"),
        )
        for column in ("unit_id", "proposal_id", "staff_id", "day"):
            op.create_index(f"ix_roster_proposal_assignment_{column}", "roster_proposal_assignment", [column])


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "roster_proposal_assignment" in tables:
        op.drop_table("roster_proposal_assignment")
    if "roster_proposal" in tables:
        op.drop_table("roster_proposal")
