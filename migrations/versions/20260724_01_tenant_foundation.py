"""tenant foundation and request lifecycle

Revision ID: 20260724_01
"""
from alembic import op
import sqlalchemy as sa

revision = "20260724_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "staff" not in inspector.get_table_names():
        # Fresh production install uses explicit Alembic operations. It must
        # never import Flask or let application metadata create the schema.
        from migrations.fresh_schema import create_fresh_schema
        create_fresh_schema()
        return
    existing_tables = set(inspector.get_table_names())
    if "unit" not in existing_tables:
        op.create_table(
            "unit",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("code", sa.String(12), nullable=False, unique=True),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("timezone", sa.String(64), nullable=False, server_default="Europe/London"),
            sa.Column("locale", sa.String(20), nullable=False, server_default="en-GB"),
            sa.Column("date_format", sa.String(30), nullable=False, server_default="%d/%m/%Y"),
            sa.Column("branding_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("status", sa.String(20), nullable=False, server_default="active"),
            sa.Column("plan", sa.String(40), nullable=False, server_default="starter"),
            sa.Column("request_months_ahead", sa.Integer(), nullable=False, server_default="3"),
            sa.Column("request_lock_day", sa.Integer(), nullable=False, server_default="20"),
            sa.Column("active_user_limit", sa.Integer(), nullable=False, server_default="10"),
            sa.Column("onboarding_step", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("trial_ends_at", sa.DateTime()),
            sa.Column("renews_at", sa.DateTime()),
            sa.Column("suspended_at", sa.DateTime()),
            sa.Column("last_active_at", sa.DateTime()),
        )
    op.execute(
        "INSERT INTO unit (id, code, name, created_at) "
        "SELECT 1, 'FIRST', 'First airport unit', CURRENT_TIMESTAMP "
        "WHERE NOT EXISTS (SELECT 1 FROM unit WHERE id=1)"
    )
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    for table in (
        "staff", "watch", "shift_type", "assignment", "shift_request",
        "annotation_type", "requirement", "leave", "sickness", "ai_rule_set",
        "change_log", "staff_watch_history",
    ):
        if table not in existing_tables:
            continue
        columns = {
            column["name"] for column in inspector.get_columns(table)
        }
        with op.batch_alter_table(table) as batch:
            if "unit_id" not in columns:
                batch.add_column(sa.Column(
                    "unit_id", sa.Integer(), nullable=False,
                    server_default="1",
                ))
                batch.create_index(f"ix_{table}_unit_id", ["unit_id"])
    inspector = sa.inspect(bind)
    if "shift_type" in existing_tables:
        columns = {c["name"] for c in inspector.get_columns("shift_type")}
        with op.batch_alter_table("shift_type") as batch:
            if "is_active" not in columns:
                batch.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
            if "is_requestable" not in columns:
                batch.add_column(sa.Column("is_requestable", sa.Boolean(), nullable=False, server_default=sa.false()))
            if "required_qualification" not in columns:
                batch.add_column(sa.Column("required_qualification", sa.String(40), nullable=False, server_default=""))
    if "shift_request" in existing_tables:
        columns = {c["name"] for c in inspector.get_columns("shift_request")}
        additions = (
            ("requester_comment", sa.Column("requester_comment", sa.String(500), nullable=False, server_default="")),
            ("created_at", sa.Column("created_at", sa.DateTime())),
            ("updated_at", sa.Column("updated_at", sa.DateTime())),
            ("fulfilled_at", sa.Column("fulfilled_at", sa.DateTime())),
            ("cancelled_at", sa.Column("cancelled_at", sa.DateTime())),
            ("resulting_assignment_id", sa.Column("resulting_assignment_id", sa.Integer())),
        )
        with op.batch_alter_table("shift_request") as batch:
            for name, column in additions:
                if name not in columns:
                    batch.add_column(column)
    if "annotation_type" in existing_tables:
        columns = {c["name"] for c in inspector.get_columns("annotation_type")}
        additions = (
            ("colour", sa.Column("colour", sa.String(20), nullable=False, server_default="#6c757d")),
            ("description", sa.Column("description", sa.Text(), nullable=False, server_default="")),
            ("note_required", sa.Column("note_required", sa.Boolean(), nullable=False, server_default=sa.false())),
            ("admin_only", sa.Column("admin_only", sa.Boolean(), nullable=False, server_default=sa.false())),
            ("has_been_used", sa.Column("has_been_used", sa.Boolean(), nullable=False, server_default=sa.false())),
        )
        with op.batch_alter_table("annotation_type") as batch:
            for name, column in additions:
                if name not in columns:
                    batch.add_column(column)


def downgrade():
    raise RuntimeError("Tenant-boundary migrations are intentionally irreversible")
