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
        # Fresh production install: create the complete metadata baseline.
        # Subsequent revisions remain explicit incremental migrations.
        from app import db
        db.metadata.create_all(bind=bind)
        return
    op.create_table(
        "unit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(12), nullable=False, unique=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="Europe/London"),
        sa.Column("locale", sa.String(20), nullable=False, server_default="en-GB"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("plan", sa.String(40), nullable=False, server_default="starter"),
        sa.Column("request_months_ahead", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("request_lock_day", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("active_user_limit", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.execute(
        "INSERT INTO unit (id, code, name, created_at) "
        "VALUES (1, 'FIRST', 'First airport unit', CURRENT_TIMESTAMP)"
    )
    for table in (
        "staff", "watch", "shift_type", "assignment", "shift_request",
        "annotation_type", "requirement", "leave", "sickness", "ai_rule_set",
        "change_log", "staff_watch_history",
    ):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("unit_id", sa.Integer(), nullable=False, server_default="1"))
            batch.create_index(f"ix_{table}_unit_id", ["unit_id"])
    with op.batch_alter_table("shift_type") as batch:
        batch.add_column(sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch.add_column(sa.Column("is_requestable", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("required_qualification", sa.String(40), nullable=False, server_default=""))
    with op.batch_alter_table("shift_request") as batch:
        batch.add_column(sa.Column("requester_comment", sa.String(500), nullable=False, server_default=""))
        batch.add_column(sa.Column("created_at", sa.DateTime()))
        batch.add_column(sa.Column("updated_at", sa.DateTime()))
        batch.add_column(sa.Column("fulfilled_at", sa.DateTime()))
        batch.add_column(sa.Column("cancelled_at", sa.DateTime()))
        batch.add_column(sa.Column("resulting_assignment_id", sa.Integer()))
    with op.batch_alter_table("annotation_type") as batch:
        batch.add_column(sa.Column("colour", sa.String(20), nullable=False, server_default="#6c757d"))
        batch.add_column(sa.Column("description", sa.Text(), nullable=False, server_default=""))
        batch.add_column(sa.Column("note_required", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("admin_only", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("has_been_used", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade():
    raise RuntimeError("Tenant-boundary migrations are intentionally irreversible")
