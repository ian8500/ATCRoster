"""roster optimisation foundations

Revision ID: 20260803_29
Revises: 20260730_28
"""
import sqlalchemy as sa
from alembic import op

revision = "20260803_29"
down_revision = "20260730_28"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if "staff" not in set(sa.inspect(bind).get_table_names()):
        return
    op.create_table(
        "work_pattern",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("cycle_length_days", sa.Integer(), nullable=False),
        sa.Column("contracted_minutes_per_cycle", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("unit_id", "name", name="uq_work_pattern_unit_name"),
        sa.CheckConstraint("cycle_length_days > 0", name="ck_work_pattern_cycle_positive"),
        sa.CheckConstraint("contracted_minutes_per_cycle >= 0", name="ck_work_pattern_minutes_nonnegative"),
    )
    op.create_table(
        "work_pattern_day",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("work_pattern_id", sa.Integer(), sa.ForeignKey("work_pattern.id"), nullable=False, index=True),
        sa.Column("day_index", sa.Integer(), nullable=False),
        sa.Column("day_type", sa.String(40), nullable=False),
        sa.Column("fixed_shift_type_id", sa.Integer(), sa.ForeignKey("shift_type.id")),
        sa.Column("required_work", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.UniqueConstraint("work_pattern_id", "day_index", name="uq_work_pattern_day_index"),
        sa.CheckConstraint("day_index >= 0", name="ck_work_pattern_day_index_nonnegative"),
    )
    op.create_table(
        "work_pattern_day_allowed_shift",
        sa.Column("work_pattern_day_id", sa.Integer(), sa.ForeignKey("work_pattern_day.id"), primary_key=True),
        sa.Column("shift_type_id", sa.Integer(), sa.ForeignKey("shift_type.id"), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
    )
    op.create_table(
        "staff_pattern_assignment",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("staff_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("work_pattern_id", sa.Integer(), sa.ForeignKey("work_pattern.id"), nullable=False, index=True),
        sa.Column("effective_from", sa.Date(), nullable=False, index=True),
        sa.Column("effective_to", sa.Date()),
        sa.Column("anchor_date", sa.Date(), nullable=False),
        sa.Column("anchor_day_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("contracted_minutes_override", sa.Integer()),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("anchor_day_index >= 0", name="ck_staff_pattern_anchor_nonnegative"),
        sa.CheckConstraint("contracted_minutes_override IS NULL OR contracted_minutes_override >= 0", name="ck_staff_pattern_override_nonnegative"),
        sa.CheckConstraint("effective_to IS NULL OR effective_to >= effective_from", name="ck_staff_pattern_dates"),
    )
    op.create_table(
        "staff_rule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("staff_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("rule_type", sa.String(40), nullable=False),
        sa.Column("hardness", sa.String(10), nullable=False, server_default="HARD"),
        sa.Column("effective_from", sa.Date(), nullable=False, index=True),
        sa.Column("effective_to", sa.Date()),
        sa.Column("shift_type_id", sa.Integer(), sa.ForeignKey("shift_type.id")),
        sa.Column("shift_group", sa.String(40)),
        sa.Column("maximum_count", sa.Integer()),
        sa.Column("rolling_period_days", sa.Integer()),
        sa.Column("penalty_weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("authorised_by_user_id", sa.Integer()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("hardness IN ('HARD','SOFT')", name="ck_staff_rule_hardness"),
        sa.CheckConstraint("effective_to IS NULL OR effective_to >= effective_from", name="ck_staff_rule_dates"),
        sa.CheckConstraint("penalty_weight >= 0", name="ck_staff_rule_penalty_nonnegative"),
    )
    op.create_table(
        "bank_holiday",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("day", sa.Date(), nullable=False, index=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.UniqueConstraint("unit_id", "day", name="uq_bank_holiday_unit_day"),
    )
    if "assignment" in set(sa.inspect(bind).get_table_names()):
        with op.batch_alter_table("assignment") as batch:
            batch.add_column(sa.Column("lock_status", sa.String(20), nullable=False, server_default="UNLOCKED"))
            batch.add_column(sa.Column("locked_by_user_id", sa.Integer()))
            batch.add_column(sa.Column("locked_at", sa.DateTime()))
            batch.add_column(sa.Column("lock_reason", sa.String(250), nullable=False, server_default=""))
            batch.create_check_constraint("ck_assignment_lock_status", "lock_status IN ('UNLOCKED','SOFT_LOCKED','HARD_LOCKED')")


def downgrade():
    bind = op.get_bind()
    if "staff" not in set(sa.inspect(bind).get_table_names()):
        return
    if "assignment" in set(sa.inspect(bind).get_table_names()):
        with op.batch_alter_table("assignment") as batch:
            batch.drop_constraint("ck_assignment_lock_status", type_="check")
            batch.drop_column("lock_reason")
            batch.drop_column("locked_at")
            batch.drop_column("locked_by_user_id")
            batch.drop_column("lock_status")
    for table in ("bank_holiday", "staff_rule", "staff_pattern_assignment", "work_pattern_day_allowed_shift", "work_pattern_day", "work_pattern"):
        op.drop_table(table)
