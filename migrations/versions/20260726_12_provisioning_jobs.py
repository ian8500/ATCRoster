"""Add durable airport provisioning jobs.

Revision ID: 20260726_12
Revises: 20260726_11
"""
import os

from alembic import op
import sqlalchemy as sa


revision = "20260726_12"
down_revision = "20260726_11"
branch_labels = None
depends_on = None


def upgrade():
    role = os.environ.get("ATCROSTER_SCHEMA_ROLE", "combined")
    if role == "operational":
        return
    inspector = sa.inspect(op.get_bind())
    if "provisioning_job" in inspector.get_table_names():
        return
    op.create_table(
        "provisioning_job",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("state", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("active_key", sa.String(20)),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=False),
        sa.Column("locked_at", sa.DateTime()),
        sa.Column("worker_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_error_code", sa.String(80), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["unit_id"], ["unit.id"]),
        sa.UniqueConstraint("idempotency_key"),
        sa.UniqueConstraint(
            "unit_id", "active_key", name="uq_active_provisioning_job"
        ),
    )
    op.create_index("ix_provisioning_job_unit_id", "provisioning_job", ["unit_id"])
    current_tables = set(sa.inspect(op.get_bind()).get_table_names())
    invitation_columns = (
        {
            column["name"]
            for column in sa.inspect(op.get_bind()).get_columns(
                "secure_invitation"
            )
        }
        if "secure_invitation" in current_tables
        else set()
    )
    if "secure_invitation" in current_tables and (
        "active_bootstrap_key" not in invitation_columns
    ):
        with op.batch_alter_table("secure_invitation") as batch:
            batch.add_column(sa.Column("active_bootstrap_key", sa.String(20)))
            batch.create_unique_constraint(
                "uq_active_bootstrap_invitation",
                ["unit_id", "role", "active_bootstrap_key"],
            )


def downgrade():
    raise RuntimeError("Provisioning job history cannot be safely removed.")
