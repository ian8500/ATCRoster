"""Store user-facing annotation detail separately from assignment notes."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "20260727_16"
down_revision = "20260727_15"
branch_labels = None
depends_on = None


def upgrade():
    inspector = inspect(op.get_bind())
    if "assignment" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("assignment")}
    if "annotation_note" not in columns:
        with op.batch_alter_table("assignment") as batch:
            batch.add_column(
                sa.Column(
                    "annotation_note",
                    sa.String(length=140),
                    nullable=False,
                    server_default="",
                )
            )


def downgrade():
    inspector = inspect(op.get_bind())
    if "assignment" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("assignment")}
    if "annotation_note" in columns:
        with op.batch_alter_table("assignment") as batch:
            batch.drop_column("annotation_note")
