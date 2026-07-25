"""preserve tenant annotation history and idempotency evidence

Revision ID: 20260725_04
Revises: 20260725_03
"""
from alembic import op
import sqlalchemy as sa


revision = "20260725_04"
down_revision = "20260725_03"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "annotation_audit" not in tables:
        op.create_table(
            "annotation_audit",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "annotation_type_id", sa.Integer(),
                sa.ForeignKey("annotation_type.id"), index=True,
            ),
            sa.Column(
                "assignment_id", sa.Integer(),
                sa.ForeignKey("assignment.id"), index=True,
            ),
            sa.Column("actor_id", sa.Integer(), nullable=False),
            sa.Column("action", sa.String(30), nullable=False),
            sa.Column("old_value", sa.Text(), nullable=False, server_default=""),
            sa.Column("new_value", sa.Text(), nullable=False, server_default=""),
            sa.Column(
                "occurred_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("transaction_key", sa.String(64), unique=True),
        )


def downgrade():
    raise RuntimeError(
        "Annotation history is assurance evidence and is not removed "
        "automatically."
    )
