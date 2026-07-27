"""Add email-backed, manager-approved account recovery.

Revision ID: 20260727_14
Revises: 20260726_13
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260727_14"
down_revision = "20260726_13"
branch_labels = None
depends_on = None


def _tables():
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table):
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table)
    }


def upgrade():
    role = os.environ.get("ATCROSTER_SCHEMA_ROLE", "combined")
    tables = _tables()
    if role in {"combined", "operational"} and "staff" in tables:
        if "email" not in _columns("staff"):
            with op.batch_alter_table("staff") as batch:
                batch.add_column(
                    sa.Column(
                        "email",
                        sa.String(254),
                        nullable=False,
                        server_default="",
                    )
                )
    if role in {"combined", "control"} and "platform_identity" in tables:
        if "email" not in _columns("platform_identity"):
            with op.batch_alter_table("platform_identity") as batch:
                batch.add_column(
                    sa.Column(
                        "email",
                        sa.String(254),
                        nullable=False,
                        server_default="",
                    )
                )
        if "recovery_request" not in tables:
            op.create_table(
                "recovery_request",
                sa.Column("id", sa.Integer(), primary_key=True),
                sa.Column("unit_id", sa.Integer(), sa.ForeignKey("unit.id")),
                sa.Column(
                    "identity_id",
                    sa.Integer(),
                    sa.ForeignKey("platform_identity.id"),
                ),
                sa.Column("person_id", sa.Integer()),
                sa.Column(
                    "approval_token_digest",
                    sa.String(64),
                    nullable=False,
                    unique=True,
                ),
                sa.Column(
                    "reset_token_digest",
                    sa.String(64),
                    unique=True,
                ),
                sa.Column(
                    "state",
                    sa.String(24),
                    nullable=False,
                    server_default="pending_approval",
                ),
                sa.Column("expires_at", sa.DateTime(), nullable=False),
                sa.Column("approved_at", sa.DateTime()),
                sa.Column("completed_at", sa.DateTime()),
                sa.Column(
                    "created_at",
                    sa.DateTime(),
                    nullable=False,
                    server_default=sa.func.now(),
                ),
            )
            op.create_index(
                "ix_recovery_request_unit_id",
                "recovery_request",
                ["unit_id"],
            )
            op.create_index(
                "ix_recovery_request_identity_id",
                "recovery_request",
                ["identity_id"],
            )


def downgrade():
    raise RuntimeError("Account recovery audit history cannot be safely removed.")
