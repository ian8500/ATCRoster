"""remove impossible control-to-operational person foreign key

Revision ID: 20260725_07
Revises: 20260725_06
"""
import sqlalchemy as sa
from alembic import op

revision = "20260725_07"
down_revision = "20260725_06"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "unit_membership" not in inspector.get_table_names():
        return
    foreign_keys = inspector.get_foreign_keys("unit_membership")
    person_keys = [
        key for key in foreign_keys
        if key.get("constrained_columns") == ["person_id"]
    ]
    if not person_keys:
        return
    with op.batch_alter_table("unit_membership") as batch:
        for key in person_keys:
            if key.get("name"):
                batch.drop_constraint(key["name"], type_="foreignkey")


def downgrade():
    raise RuntimeError(
        "A control database cannot reference an operational database row."
    )
