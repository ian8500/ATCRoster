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
        row[0] for row in connection.execute(
            sa.text("SELECT DISTINCT unit_id FROM staff")
        )
    ]
    for unit_id in unit_ids:
        for code, label in (
            ("examiner", "Examiner"),
            ("safety_controller", "Safety controller"),
            ("observer", "Observer"),
        ):
            connection.execute(sa.text(
                "INSERT INTO position_participant_role "
                "(unit_id, code, label, is_primary, counts_for_currency, is_active) "
                "SELECT :unit_id, :code, :label, false, false, true "
                "WHERE NOT EXISTS (SELECT 1 FROM position_participant_role "
                "WHERE unit_id = :unit_id AND code = :code)"
            ), {"unit_id": unit_id, "code": code, "label": label})


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    op.execute(sa.text(
        "DELETE FROM position_participant_role "
        "WHERE code IN ('examiner', 'safety_controller', 'observer')"
    ))
