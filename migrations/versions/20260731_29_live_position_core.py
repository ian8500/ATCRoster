"""live position monitoring core records

Revision ID: 20260731_29
Revises: 20260730_28
"""
import os

from alembic import op
import sqlalchemy as sa


revision = "20260731_29"
down_revision = "20260730_28"
branch_labels = None
depends_on = None


def _tenant_columns():
    return [
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
    ]


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return

    staff_columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("staff")
    }
    with op.batch_alter_table("staff") as batch:
        if "role" in staff_columns:
            batch.alter_column(
                "role", existing_type=sa.String(10), type_=sa.String(32),
                existing_nullable=False,
            )
        else:
            batch.add_column(sa.Column(
                "role", sa.String(32), nullable=False, server_default="user"
            ))

    op.create_table(
        "position_currency_category", *_tenant_columns(),
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint(
            "unit_id", "code", name="uq_position_currency_category_code"
        ),
    )
    existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "operational_position" not in existing_tables:
        op.create_table(
            "operational_position", *_tenant_columns(),
            sa.Column("code", sa.String(30), nullable=False),
            sa.Column("label", sa.String(120), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("group_name", sa.String(80), nullable=False, server_default=""),
            sa.Column("currency_category_id", sa.Integer(), sa.ForeignKey(
                "position_currency_category.id",
                name="fk_operational_position_currency_category",
            )),
            sa.Column("supporting_participants_allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("multiple_supporting_participants_allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("training_supported", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("assessment_supported", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("is_safety_critical", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.UniqueConstraint("unit_id", "code", name="uq_position_unit_code"),
        )
    else:
        with op.batch_alter_table("operational_position") as batch:
            batch.add_column(sa.Column("display_order", sa.Integer(), nullable=False, server_default="100"))
            batch.add_column(sa.Column("group_name", sa.String(80), nullable=False, server_default=""))
            batch.add_column(sa.Column("currency_category_id", sa.Integer()))
            batch.add_column(sa.Column("supporting_participants_allowed", sa.Boolean(), nullable=False, server_default=sa.true()))
            batch.add_column(sa.Column("multiple_supporting_participants_allowed", sa.Boolean(), nullable=False, server_default=sa.true()))
            batch.add_column(sa.Column("training_supported", sa.Boolean(), nullable=False, server_default=sa.true()))
            batch.add_column(sa.Column("assessment_supported", sa.Boolean(), nullable=False, server_default=sa.true()))
            batch.create_foreign_key(
                "fk_operational_position_currency_category",
                "position_currency_category", ["currency_category_id"], ["id"],
            )

    op.create_table(
        "position_participant_role", *_tenant_columns(),
        sa.Column("code", sa.String(30), nullable=False),
        sa.Column("label", sa.String(80), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("counts_for_currency", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint(
            "unit_id", "code", name="uq_position_participant_role_code"
        ),
    )
    op.create_table(
        "position_status_event", *_tenant_columns(),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id"), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False),
        sa.Column("reason", sa.String(250), nullable=False, server_default=""),
        sa.Column("transaction_key", sa.String(64), nullable=False, unique=True),
        sa.CheckConstraint("status IN ('closed', 'open')", name="ck_position_status_event_status"),
    )
    op.create_table(
        "position_session", *_tenant_columns(),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id"), nullable=False, index=True),
        sa.Column("primary_person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("session_type", sa.String(20), nullable=False, server_default="operational"),
        sa.Column("started_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("ended_at", sa.DateTime(), index=True),
        sa.Column("ended_reason", sa.String(40), nullable=False, server_default=""),
        sa.Column("maximum_duration_seconds", sa.Integer()),
        sa.Column("warning_threshold_seconds", sa.Integer()),
        sa.Column("due_off_at", sa.DateTime()),
        sa.Column("currency_category_id", sa.Integer(), sa.ForeignKey("position_currency_category.id")),
        sa.Column("created_by_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False),
        sa.Column("corrected_at", sa.DateTime()),
        sa.Column("corrected_by_id", sa.Integer(), sa.ForeignKey("staff.id")),
        sa.Column("correction_reason", sa.String(500), nullable=False, server_default=""),
        sa.Column("is_void", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("transaction_key", sa.String(64), nullable=False, unique=True),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_position_session_time_order"),
    )
    op.create_index(
        "uq_position_session_open_position", "position_session", ["unit_id", "position_id"],
        unique=True, postgresql_where=sa.text("ended_at IS NULL AND is_void = false"),
        sqlite_where=sa.text("ended_at IS NULL AND is_void = 0"),
    )
    op.create_index(
        "uq_position_session_open_controller", "position_session", ["unit_id", "primary_person_id"],
        unique=True, postgresql_where=sa.text("ended_at IS NULL AND is_void = false"),
        sqlite_where=sa.text("ended_at IS NULL AND is_void = 0"),
    )
    op.create_table(
        "position_session_participant", *_tenant_columns(),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("position_session.id"), nullable=False, index=True),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, index=True),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("position_participant_role.id"), nullable=False, index=True),
        sa.Column("started_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("ended_at", sa.DateTime(), index=True),
        sa.Column("ended_reason", sa.String(40), nullable=False, server_default=""),
        sa.Column("transaction_key", sa.String(64), nullable=False, unique=True),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_position_participant_time_order"),
    )
    op.create_index(
        "uq_position_participant_open_person", "position_session_participant",
        ["unit_id", "person_id"], unique=True,
        postgresql_where=sa.text("ended_at IS NULL"),
        sqlite_where=sa.text("ended_at IS NULL"),
    )
    op.create_table(
        "controller_kiosk_credential", *_tenant_columns(),
        sa.Column("person_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False, unique=True),
        sa.Column("pin_hash", sa.String(255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime()),
        sa.Column("changed_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "position_session_audit", *_tenant_columns(),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("position_session.id"), index=True),
        sa.Column("position_id", sa.Integer(), sa.ForeignKey("operational_position.id"), index=True),
        sa.Column("actor_id", sa.Integer(), sa.ForeignKey("staff.id"), nullable=False),
        sa.Column("action", sa.String(40), nullable=False, index=True),
        sa.Column("occurred_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("old_value_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("new_value_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("reason", sa.String(500), nullable=False, server_default=""),
        sa.Column("transaction_key", sa.String(64), nullable=False, index=True),
    )

    roles = sa.table(
        "position_participant_role",
        sa.column("unit_id", sa.Integer), sa.column("code", sa.String),
        sa.column("label", sa.String), sa.column("is_primary", sa.Boolean),
        sa.column("counts_for_currency", sa.Boolean), sa.column("is_active", sa.Boolean),
    )
    # Airport databases intentionally do not contain the control-plane Unit
    # table. Staff is authoritative for the tenant identifier on this physical
    # database.
    unit_ids = [
        row[0] for row in op.get_bind().execute(
            sa.text("SELECT DISTINCT unit_id FROM staff")
        )
    ]
    op.bulk_insert(roles, [
        {"unit_id": unit_id, "code": code, "label": label, "is_primary": primary,
         "counts_for_currency": counts, "is_active": True}
        for unit_id in unit_ids
        for code, label, primary, counts in (
            ("primary", "Primary controller", True, True),
            ("ojti", "OJTI", False, False),
            ("assessor", "Assessor", False, False),
            ("secondary", "Secondary controller", False, False),
        )
    ])


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    for table in (
        "position_session_audit", "controller_kiosk_credential",
        "position_session_participant", "position_session",
        "position_status_event", "position_participant_role",
    ):
        op.drop_table(table)
    with op.batch_alter_table("operational_position") as batch:
        batch.drop_constraint("fk_operational_position_currency_category", type_="foreignkey")
        for column in (
            "assessment_supported", "training_supported",
            "multiple_supporting_participants_allowed",
            "supporting_participants_allowed", "currency_category_id",
            "group_name", "display_order",
        ):
            batch.drop_column(column)
    op.drop_table("position_currency_category")
    with op.batch_alter_table("staff") as batch:
        batch.alter_column(
            "role", existing_type=sa.String(32), type_=sa.String(10),
            existing_nullable=False,
        )
