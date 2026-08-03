"""Add dated workforce fields required by the joiner workflow.

Revision ID: 20260803_44
Revises: 20260803_43
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_44"
down_revision = "20260803_43"
branch_labels = None
depends_on = None


COLUMNS = (
    sa.Column("employment_start_date", sa.Date()),
    sa.Column("unit_join_date", sa.Date()),
    sa.Column("roster_start_date", sa.Date()),
    sa.Column(
        "employment_type", sa.String(20), nullable=False,
        server_default="FULL_TIME",
    ),
    sa.Column("contracted_minutes_per_week", sa.Integer()),
    sa.Column("workforce_notes", sa.Text(), nullable=False, server_default=""),
)


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "staff" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("staff")}
    missing = [column for column in COLUMNS if column.name not in existing]
    if missing:
        with op.batch_alter_table("staff") as batch:
            for column in missing:
                batch.add_column(column)
    constraints = {
        item.get("name") for item in sa.inspect(bind).get_check_constraints("staff")
    }
    with op.batch_alter_table("staff") as batch:
        if "ck_staff_employment_type" not in constraints:
            batch.create_check_constraint(
                "ck_staff_employment_type",
                "employment_type IN ('FULL_TIME','PART_TIME')",
            )
        if "ck_staff_contracted_minutes_nonnegative" not in constraints:
            batch.create_check_constraint(
                "ck_staff_contracted_minutes_nonnegative",
                "contracted_minutes_per_week IS NULL OR "
                "contracted_minutes_per_week >= 0",
            )


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "staff" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("staff")}
    constraints = {
        item.get("name") for item in inspector.get_check_constraints("staff")
    }
    with op.batch_alter_table("staff") as batch:
        for name in (
            "ck_staff_contracted_minutes_nonnegative",
            "ck_staff_employment_type",
        ):
            if name in constraints:
                batch.drop_constraint(name, type_="check")
        for column in reversed(COLUMNS):
            if column.name in existing:
                batch.drop_column(column.name)
