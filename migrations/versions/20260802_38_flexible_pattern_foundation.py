"""Add flexible work patterns and effective-dated staff rules.

Revision ID: 20260802_38
Revises: 20260801_37
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260802_38"
down_revision = "20260801_37"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    # Some supported legacy fixtures contain only a subset of the roster
    # schema. They remain migratable and are not suitable for pattern records
    # until the prerequisite operational tables exist.
    if not {"staff", "shift_type"}.issubset(tables):
        return
    shift_uniques = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints("shift_type")
    }
    if ("unit_id", "id") not in shift_uniques:
        with op.batch_alter_table("shift_type") as batch:
            batch.create_unique_constraint("uq_shift_unit_id", ["unit_id", "id"])

    # Revision 01 creates the current fresh schema in one pass. Consequently,
    # a brand-new database already contains these tables by the time Alembic
    # reaches this revision, whereas an upgraded database still needs them.
    # Keep this revision safe for both paths.
    pattern_tables = {
        "work_pattern",
        "work_pattern_day",
        "work_pattern_day_allowed_shift",
        "staff_pattern_assignment",
        "staff_rule",
    }
    if pattern_tables.issubset(tables):
        return

    op.create_table(
        "work_pattern",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("cycle_length_days", sa.Integer(), nullable=False),
        sa.Column(
            "contracted_minutes_per_cycle", sa.Integer(),
            nullable=False, server_default="0",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("unit_id", "name", name="uq_work_pattern_unit_name"),
        sa.UniqueConstraint("unit_id", "id", name="uq_work_pattern_unit_id"),
        sa.CheckConstraint("cycle_length_days > 0", name="ck_work_pattern_cycle_positive"),
        sa.CheckConstraint(
            "contracted_minutes_per_cycle >= 0",
            name="ck_work_pattern_minutes_nonnegative",
        ),
    )
    op.create_table(
        "work_pattern_day",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("work_pattern_id", sa.Integer(), nullable=False, index=True),
        sa.Column("day_index", sa.Integer(), nullable=False),
        sa.Column("day_type", sa.String(32), nullable=False),
        sa.Column("fixed_shift_type_id", sa.Integer()),
        sa.Column("required_work", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.String(500), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(
            ["unit_id", "work_pattern_id"],
            ["work_pattern.unit_id", "work_pattern.id"],
            name="fk_work_pattern_day_pattern_unit", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["unit_id", "fixed_shift_type_id"],
            ["shift_type.unit_id", "shift_type.id"],
            name="fk_work_pattern_day_shift_unit",
        ),
        sa.UniqueConstraint(
            "unit_id", "work_pattern_id", "day_index",
            name="uq_work_pattern_day_index",
        ),
        sa.UniqueConstraint("unit_id", "id", name="uq_work_pattern_day_unit_id"),
        sa.CheckConstraint("day_index >= 0", name="ck_work_pattern_day_index_nonnegative"),
        sa.CheckConstraint(
            "day_type IN ('FIXED_SHIFT','WORK_ANY','WORK_ALLOWED_SET','OFF',"
            "'OPTIONAL_WORK','PROTECTED_NON_OPERATIONAL')",
            name="ck_work_pattern_day_type",
        ),
        sa.CheckConstraint(
            "(day_type = 'FIXED_SHIFT' AND fixed_shift_type_id IS NOT NULL) OR "
            "(day_type <> 'FIXED_SHIFT' AND fixed_shift_type_id IS NULL)",
            name="ck_work_pattern_day_fixed_shift",
        ),
    )
    op.create_table(
        "work_pattern_day_allowed_shift",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("work_pattern_day_id", sa.Integer(), nullable=False, index=True),
        sa.Column("shift_type_id", sa.Integer(), nullable=False, index=True),
        sa.ForeignKeyConstraint(
            ["unit_id", "work_pattern_day_id"],
            ["work_pattern_day.unit_id", "work_pattern_day.id"],
            name="fk_pattern_allowed_day_unit", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["unit_id", "shift_type_id"],
            ["shift_type.unit_id", "shift_type.id"],
            name="fk_pattern_allowed_shift_unit",
        ),
        sa.UniqueConstraint(
            "unit_id", "work_pattern_day_id", "shift_type_id",
            name="uq_pattern_day_allowed_shift",
        ),
    )
    op.create_table(
        "staff_pattern_assignment",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("staff_id", sa.Integer(), nullable=False, index=True),
        sa.Column("work_pattern_id", sa.Integer(), nullable=False, index=True),
        sa.Column("effective_from", sa.Date(), nullable=False, index=True),
        sa.Column("effective_to", sa.Date(), index=True),
        sa.Column("anchor_date", sa.Date(), nullable=False),
        sa.Column("anchor_day_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("contracted_minutes_override", sa.Integer()),
        sa.Column("notes", sa.String(500), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["unit_id", "staff_id"], ["staff.unit_id", "staff.id"],
            name="fk_staff_pattern_person_unit", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["unit_id", "work_pattern_id"],
            ["work_pattern.unit_id", "work_pattern.id"],
            name="fk_staff_pattern_pattern_unit",
        ),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_staff_pattern_effective_range",
        ),
        sa.CheckConstraint("anchor_day_index >= 0", name="ck_staff_pattern_anchor_index"),
        sa.CheckConstraint(
            "contracted_minutes_override IS NULL OR contracted_minutes_override >= 0",
            name="ck_staff_pattern_minutes_nonnegative",
        ),
    )
    op.create_table(
        "staff_rule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("staff_id", sa.Integer(), nullable=False, index=True),
        sa.Column("rule_type", sa.String(40), nullable=False, index=True),
        sa.Column("hardness", sa.String(8), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False, index=True),
        sa.Column("effective_to", sa.Date(), index=True),
        sa.Column("shift_type_id", sa.Integer()),
        sa.Column("shift_group", sa.String(20)),
        sa.Column("maximum_count", sa.Integer()),
        sa.Column("rolling_period_days", sa.Integer()),
        sa.Column("weekdays_mask", sa.Integer()),
        sa.Column("penalty_weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reason", sa.String(500), nullable=False, server_default=""),
        sa.Column("authorised_by_user_id", sa.Integer()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["unit_id", "staff_id"], ["staff.unit_id", "staff.id"],
            name="fk_staff_rule_person_unit", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["unit_id", "shift_type_id"], ["shift_type.unit_id", "shift_type.id"],
            name="fk_staff_rule_shift_unit",
        ),
        sa.ForeignKeyConstraint(
            ["unit_id", "authorised_by_user_id"], ["staff.unit_id", "staff.id"],
            name="fk_staff_rule_authoriser_unit",
        ),
        sa.CheckConstraint("hardness IN ('HARD','SOFT')", name="ck_staff_rule_hardness"),
        sa.CheckConstraint(
            "rule_type IN ('NO_NIGHT','AVOID_NIGHT','NO_EARLY','AVOID_EARLY',"
            "'ALLOWED_SHIFT','DISALLOWED_SHIFT','MAX_NIGHTS_PER_CYCLE',"
            "'MAX_SHIFTS_PER_CYCLE','AVAILABLE_WEEKDAYS','UNAVAILABLE_WEEKDAYS',"
            "'MAX_CONTRACTED_MINUTES','PREFERRED_SHIFT','PREFERRED_DAY_OFF')",
            name="ck_staff_rule_type",
        ),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_staff_rule_effective_range",
        ),
        sa.CheckConstraint("maximum_count IS NULL OR maximum_count >= 0", name="ck_staff_rule_maximum_nonnegative"),
        sa.CheckConstraint("rolling_period_days IS NULL OR rolling_period_days > 0", name="ck_staff_rule_period_positive"),
        sa.CheckConstraint("weekdays_mask IS NULL OR (weekdays_mask >= 0 AND weekdays_mask <= 127)", name="ck_staff_rule_weekdays_mask"),
        sa.CheckConstraint("penalty_weight >= 0", name="ck_staff_rule_penalty"),
    )


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    for table_name in (
        "staff_rule",
        "staff_pattern_assignment",
        "work_pattern_day_allowed_shift",
        "work_pattern_day",
        "work_pattern",
    ):
        if table_name in tables:
            op.drop_table(table_name)
