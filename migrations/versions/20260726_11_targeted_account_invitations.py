"""Link access invitations to configured roster people.

Revision ID: 20260726_11
Revises: 20260726_10
"""
from alembic import op
import sqlalchemy as sa


revision = "20260726_11"
down_revision = "20260726_10"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if "secure_invitation" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("secure_invitation")
    }
    if "target_person_id" not in columns:
        with op.batch_alter_table("secure_invitation") as batch:
            batch.add_column(sa.Column("target_person_id", sa.Integer()))
            batch.create_index(
                "ix_secure_invitation_target_person_id",
                ["target_person_id"],
                unique=False,
            )


def downgrade():
    raise RuntimeError(
        "Targeted access invitations cannot be safely removed."
    )
