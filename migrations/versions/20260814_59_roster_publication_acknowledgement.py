"""Prevent duplicate acknowledgements of a roster publication version.

Revision ID: 20260814_59
Revises: 20260813_58
"""
import os

from alembic import op
import sqlalchemy as sa

revision = "20260814_59"
down_revision = "20260813_58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "roster_acknowledgement" not in inspector.get_table_names():
        return
    names = {item["name"] for item in inspector.get_unique_constraints("roster_acknowledgement")}
    if "uq_roster_acknowledgement_publication_person" not in names:
        with op.batch_alter_table("roster_acknowledgement") as batch:
            batch.create_unique_constraint(
                "uq_roster_acknowledgement_publication_person",
                ["unit_id", "publication_id", "person_id"],
            )


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    inspector = sa.inspect(op.get_bind())
    if "roster_acknowledgement" not in inspector.get_table_names():
        return
    names = {item["name"] for item in inspector.get_unique_constraints("roster_acknowledgement")}
    if "uq_roster_acknowledgement_publication_person" in names:
        with op.batch_alter_table("roster_acknowledgement") as batch:
            batch.drop_constraint("uq_roster_acknowledgement_publication_person", type_="unique")
