"""Add complete effective-dated competency fields.

Revision ID: 20260803_45
Revises: 20260803_44
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_45"
down_revision = "20260803_44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "person_qualification" not in inspector.get_table_names():
        return
    existing = {
        column["name"] for column in inspector.get_columns("person_qualification")
    }
    columns = (
        sa.Column("valid_to", sa.Date()),
        sa.Column("suspended_from", sa.Date()),
        sa.Column("suspended_to", sa.Date()),
        sa.Column("evidence_reference", sa.String(500), nullable=False, server_default=""),
        sa.Column("created_by_user_id", sa.Integer()),
    )
    with op.batch_alter_table("person_qualification") as batch:
        for column in columns:
            if column.name not in existing:
                batch.add_column(column)
    constraints = {
        row.get("name") for row in sa.inspect(bind).get_check_constraints(
            "person_qualification"
        )
    }
    with op.batch_alter_table("person_qualification") as batch:
        if "ck_person_qualification_valid_range" not in constraints:
            batch.create_check_constraint(
                "ck_person_qualification_valid_range",
                "valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from",
            )
        if "ck_person_qualification_suspension_range" not in constraints:
            batch.create_check_constraint(
                "ck_person_qualification_suspension_range",
                "suspended_to IS NULL OR suspended_from IS NULL OR "
                "suspended_to >= suspended_from",
            )


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if "person_qualification" not in sa.inspect(bind).get_table_names():
        return
    with op.batch_alter_table("person_qualification") as batch:
        batch.drop_constraint("ck_person_qualification_suspension_range", type_="check")
        batch.drop_constraint("ck_person_qualification_valid_range", type_="check")
        for name in (
            "created_by_user_id", "evidence_reference", "suspended_to",
            "suspended_from", "valid_to",
        ):
            batch.drop_column(name)
