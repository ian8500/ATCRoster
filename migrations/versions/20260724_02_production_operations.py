"""production operational assurance domain

Revision ID: 20260724_02
Revises: 20260724_01
"""
from alembic import op
import sqlalchemy as sa

revision = "20260724_02"
down_revision = "20260724_01"
branch_labels = None
depends_on = None


def _tenant_columns():
    return [
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), sa.ForeignKey("unit.id"), nullable=False, index=True),
    ]


def upgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "operational_position" in existing:
        if "mfa_credential" not in existing:
            op.create_table(
                "mfa_credential", *_tenant_columns(),
                sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, unique=True),
                sa.Column("encrypted_secret", sa.Text(), nullable=False),
                sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
                sa.Column("enrolled_at", sa.DateTime()),
                sa.Column("last_used_step", sa.BigInteger()),
                sa.Column("recovery_codes_digest", sa.Text(), nullable=False, server_default="[]"),
            )
        return
    op.create_table(
        "operational_position", *_tenant_columns(),
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("is_safety_critical", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("unit_id", "code", name="uq_position_unit_code"),
    )
    op.create_table(
        "position_endorsement", *_tenant_columns(),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id"), nullable=False, index=True),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_until", sa.Date()),
        sa.Column("status", sa.String(20), nullable=False, server_default="valid"),
        sa.Column("restrictions", sa.Text(), nullable=False, server_default=""),
        sa.UniqueConstraint("unit_id", "person_id", "position_id", name="uq_position_endorsement_person"),
    )
    op.create_table(
        "position_requirement", *_tenant_columns(),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("shift_code", sa.String(10), nullable=False),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id"), nullable=False),
        sa.Column("required_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("contingency_count", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("unit_id", "day", "shift_code", "position_id", name="uq_position_requirement_day_shift"),
    )
    op.create_table(
        "break_plan", *_tenant_columns(),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id")),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False, server_default="break"),
        sa.Column("state", sa.String(20), nullable=False, server_default="planned"),
        sa.Column("recorded_by_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "achieved_duty", *_tenant_columns(),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("planned_assignment_id", sa.Integer(), sa.ForeignKey("assignment.id")),
        sa.Column("actual_start", sa.DateTime(), nullable=False),
        sa.Column("actual_end", sa.DateTime(), nullable=False),
        sa.Column("duty_type", sa.String(30), nullable=False, server_default="operational"),
        sa.Column("variance_reason", sa.String(500), nullable=False, server_default=""),
        sa.Column("recorded_by_id", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("unit_id", "person_id", "day", name="uq_achieved_duty_person_day"),
    )
    op.create_table(
        "fatigue_report", *_tenant_columns(),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("duty_day", sa.Date(), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("summary", sa.String(500), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("reported_at", sa.DateTime(), nullable=False),
        sa.Column("manager_response", sa.String(1000), nullable=False, server_default=""),
        sa.Column("reviewed_by_id", sa.Integer()),
        sa.Column("reviewed_at", sa.DateTime()),
        sa.Column("closed_at", sa.DateTime()),
    )
    op.create_table(
        "roster_rule_version", *_tenant_columns(),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("rules_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("state", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("effective_from", sa.Date()),
        sa.Column("change_reference", sa.String(120), nullable=False, server_default=""),
        sa.Column("consultation_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("approved_by_id", sa.Integer()),
        sa.Column("approved_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("unit_id", "version", name="uq_roster_rule_unit_version"),
    )
    op.create_table(
        "mfa_credential", *_tenant_columns(),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, unique=True),
        sa.Column("encrypted_secret", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("enrolled_at", sa.DateTime()),
        sa.Column("last_used_step", sa.BigInteger()),
        sa.Column("recovery_codes_digest", sa.Text(), nullable=False, server_default="[]"),
    )


def downgrade():
    for table in (
        "mfa_credential", "roster_rule_version", "fatigue_report", "achieved_duty",
        "break_plan", "position_requirement", "position_endorsement",
        "operational_position",
    ):
        op.drop_table(table)
