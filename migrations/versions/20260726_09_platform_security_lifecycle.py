"""Add central platform security and lifecycle state.

Revision ID: 20260726_09
Revises: 20260725_08
"""
from alembic import op
import sqlalchemy as sa

revision = "20260726_09"
down_revision = "20260725_08"
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
    tables = _tables()
    if "platform_identity" in tables and "platform_mfa_credential" not in tables:
        op.create_table(
            "platform_mfa_credential",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("identity_id", sa.Integer(), sa.ForeignKey("platform_identity.id"), nullable=False, unique=True),
            sa.Column("encrypted_secret", sa.Text(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("enrolled_at", sa.DateTime()),
            sa.Column("last_used_step", sa.BigInteger()),
            sa.Column("recovery_codes_digest", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("reset_required", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    if "secure_invitation" in tables and "issued_at" not in _columns("secure_invitation"):
        with op.batch_alter_table("secure_invitation") as batch:
            batch.add_column(sa.Column("issued_at", sa.DateTime()))
        op.execute("UPDATE secure_invitation SET issued_at=CURRENT_TIMESTAMP WHERE issued_at IS NULL")
    if {"secure_invitation", "platform_identity", "unit_membership"} <= tables and "signup_workflow" not in tables:
        op.create_table(
            "signup_workflow",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("invitation_id", sa.Integer(), sa.ForeignKey("secure_invitation.id"), nullable=False, unique=True),
            sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
            sa.Column("state", sa.String(40), nullable=False),
            sa.Column("normalized_username", sa.String(120), nullable=False),
            sa.Column("identity_id", sa.Integer(), sa.ForeignKey("platform_identity.id")),
            sa.Column("operational_person_id", sa.Integer()),
            sa.Column("membership_id", sa.Integer(), sa.ForeignKey("unit_membership.id")),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_error_code", sa.String(80), nullable=False, server_default=""),
            sa.Column("compensation_state", sa.String(40), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
    if "database_routing_metadata" in tables:
        existing = _columns("database_routing_metadata")
        additions = (
            ("provisioning_state", sa.String(40), "pending"),
            ("last_error_code", sa.String(80), ""),
            ("attempt_count", sa.Integer(), "0"),
            ("last_attempt_at", sa.DateTime(), None),
            ("ready_at", sa.DateTime(), None),
        )
        with op.batch_alter_table("database_routing_metadata") as batch:
            for name, column_type, default in additions:
                if name not in existing:
                    batch.add_column(sa.Column(name, column_type, server_default=default))
    if "platform_identity" in tables and "central_security_audit" not in tables:
        op.create_table(
            "central_security_audit",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("identity_id", sa.Integer(), sa.ForeignKey("platform_identity.id")),
            sa.Column("event_type", sa.String(80), nullable=False),
            sa.Column("principal_digest", sa.String(32), nullable=False, server_default=""),
            sa.Column("outcome", sa.String(20), nullable=False),
            sa.Column("safe_detail", sa.String(200), nullable=False, server_default=""),
            sa.Column("occurred_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )


def downgrade():
    raise RuntimeError("Platform security lifecycle state is not safely reversible.")
