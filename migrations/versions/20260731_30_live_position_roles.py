"""add standard live-position supporting roles

Revision ID: 20260731_30
Revises: 20260731_29
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260731_30"
down_revision = "20260731_29"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    connection = op.get_bind()
    unit_ids = [
        row[0]
        for row in connection.execute(sa.text("SELECT DISTINCT unit_id FROM staff"))
    ]
    roles = sa.table(
        "position_participant_role",
        sa.column("unit_id", sa.Integer()),
        sa.column("code", sa.String(30)),
        sa.column("label", sa.String(80)),
        sa.column("is_primary", sa.Boolean()),
        sa.column("counts_for_currency", sa.Boolean()),
        sa.column("is_active", sa.Boolean()),
    )
    for unit_id in unit_ids:
        existing = {
            row[0]
            for row in connection.execute(
                sa.select(roles.c.code).where(roles.c.unit_id == unit_id)
            )
        }
        rows = [
            {
                "unit_id": unit_id,
                "code": code,
                "label": label,
                "is_primary": False,
                "counts_for_currency": False,
                "is_active": True,
            }
            for code, label in (
                ("examiner", "Examiner"),
                ("safety_controller", "Safety controller"),
                ("observer", "Observer"),
            )
            if code not in existing
        ]
        if rows:
            connection.execute(roles.insert(), rows)


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    op.execute(
        sa.text(
            "DELETE FROM position_participant_role "
            "WHERE code IN ('examiner', 'safety_controller', 'observer')"
        )
    )
