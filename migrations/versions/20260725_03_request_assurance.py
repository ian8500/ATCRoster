"""complete the shift-request lifecycle and evidence tables

Revision ID: 20260725_03
Revises: 20260724_02
"""
from alembic import op
import sqlalchemy as sa


revision = "20260725_03"
down_revision = "20260724_02"
branch_labels = None
depends_on = None


def _column_names(inspector, table):
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "shift_request" in tables:
        columns = _column_names(inspector, "shift_request")
        additions = (
            ("submitted_at", sa.Column("submitted_at", sa.DateTime())),
            ("created_at", sa.Column("created_at", sa.DateTime())),
            ("updated_at", sa.Column("updated_at", sa.DateTime())),
            ("fulfilled_at", sa.Column("fulfilled_at", sa.DateTime())),
            ("cancelled_at", sa.Column("cancelled_at", sa.DateTime())),
            (
                "resulting_assignment_id",
                sa.Column(
                    "resulting_assignment_id",
                    sa.Integer(),
                    sa.ForeignKey("assignment.id"),
                ),
            ),
        )
        with op.batch_alter_table("shift_request") as batch:
            for name, column in additions:
                if name not in columns:
                    batch.add_column(column)
        op.execute(
            sa.text(
                "UPDATE shift_request SET "
                "created_at = COALESCE(created_at, CURRENT_TIMESTAMP), "
                "updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP), "
                "submitted_at = COALESCE(submitted_at, created_at, CURRENT_TIMESTAMP)"
            )
        )

    if "request_audit" not in tables:
        op.create_table(
            "request_audit",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "request_id", sa.Integer(), sa.ForeignKey("shift_request.id"),
                nullable=False, index=True,
            ),
            sa.Column("actor_id", sa.Integer(), nullable=False),
            sa.Column(
                "occurred_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("transition", sa.String(30), nullable=False),
            sa.Column("old_value", sa.Text(), nullable=False, server_default=""),
            sa.Column("new_value", sa.Text(), nullable=False, server_default=""),
            sa.Column("reason", sa.String(500), nullable=False, server_default=""),
        )

    if "notification" not in tables:
        op.create_table(
            "notification",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "recipient_id", sa.Integer(), sa.ForeignKey("staff.id"),
                nullable=False, index=True,
            ),
            sa.Column("kind", sa.String(40), nullable=False),
            sa.Column("message", sa.String(500), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("read_at", sa.DateTime()),
        )


def downgrade():
    raise RuntimeError(
        "Shift-request assurance records are business evidence and are not "
        "removed automatically."
    )
