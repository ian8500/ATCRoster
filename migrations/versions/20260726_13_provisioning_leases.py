"""Add renewable provisioning leases and worker heartbeat.

Revision ID: 20260726_13
Revises: 20260726_12
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260726_13"
down_revision = "20260726_12"
branch_labels = None
depends_on = None


def upgrade():
    role = os.environ.get("ATCROSTER_SCHEMA_ROLE", "combined")
    if role == "operational":
        return
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "provisioning_job" in tables:
        columns = {
            column["name"] for column in inspector.get_columns("provisioning_job")
        }
        with op.batch_alter_table("provisioning_job") as batch:
            if "lease_owner" not in columns:
                batch.add_column(
                    sa.Column(
                        "lease_owner", sa.String(64), nullable=False, server_default=""
                    )
                )
            if "lease_expires_at" not in columns:
                batch.add_column(sa.Column("lease_expires_at", sa.DateTime()))
                batch.create_index(
                    "ix_provisioning_job_lease_expires_at", ["lease_expires_at"]
                )
    if "worker_heartbeat" not in tables:
        op.create_table(
            "worker_heartbeat",
            sa.Column("worker_id", sa.String(64), primary_key=True),
            sa.Column(
                "process_type",
                sa.String(30),
                nullable=False,
                server_default="provisioning",
            ),
            sa.Column(
                "state", sa.String(30), nullable=False, server_default="starting"
            ),
            sa.Column("last_seen_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=False),
        )


def downgrade():
    raise RuntimeError("Provisioning lease history cannot be safely removed.")
