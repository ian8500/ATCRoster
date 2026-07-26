"""scope roster settings to an airport unit

Revision ID: 20260725_05
Revises: 20260725_04
"""
import os

from alembic import op
import sqlalchemy as sa


revision = "20260725_05"
down_revision = "20260725_04"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "roster_setting" not in inspector.get_table_names():
        op.create_table(
            "roster_setting",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
                nullable=False, index=True,
            ),
            sa.Column("key", sa.String(50), nullable=False),
            sa.Column("value", sa.Text(), nullable=False, server_default=""),
            sa.UniqueConstraint(
                "unit_id", "key", name="uq_roster_setting_unit_key"
            ),
        )
        return
    columns = {column["name"] for column in inspector.get_columns("roster_setting")}
    if {"id", "unit_id"}.issubset(columns):
        return
    op.create_table(
        "roster_setting_scoped",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "unit_id", sa.Integer(), sa.ForeignKey("unit.id"),
            nullable=False, index=True,
        ),
        sa.Column("key", sa.String(50), nullable=False),
        sa.Column("value", sa.Text(), nullable=False, server_default=""),
        sa.UniqueConstraint(
            "unit_id", "key", name="uq_roster_setting_unit_key"
        ),
    )
    op.execute(
        "INSERT INTO roster_setting_scoped (unit_id, key, value) "
        "SELECT 1, key, value FROM roster_setting"
    )
    op.drop_table("roster_setting")
    op.rename_table("roster_setting_scoped", "roster_setting")


def downgrade():
    raise RuntimeError(
        "Per-airport settings cannot be collapsed without losing tenant data."
    )
