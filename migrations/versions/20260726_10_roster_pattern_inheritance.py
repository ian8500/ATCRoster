"""Add unit and watch roster-pattern inheritance.

Revision ID: 20260726_10
Revises: 20260726_09
"""
from alembic import op
import sqlalchemy as sa


revision = "20260726_10"
down_revision = "20260726_09"
branch_labels = None
depends_on = None


def _columns(table):
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table)
    }


def upgrade():
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "watch" in tables:
        watch_columns = _columns("watch")
        with op.batch_alter_table("watch") as batch:
            if "pattern_csv" not in watch_columns:
                batch.add_column(sa.Column(
                    "pattern_csv", sa.String(500),
                    nullable=False, server_default="",
                ))
            if "pattern_anchor" not in watch_columns:
                batch.add_column(sa.Column("pattern_anchor", sa.Date()))

    if "staff" not in tables:
        return
    staff_columns = _columns("staff")
    if "pattern_override" not in staff_columns:
        with op.batch_alter_table("staff") as batch:
            batch.add_column(sa.Column(
                "pattern_override", sa.Boolean(),
                nullable=False, server_default=sa.false(),
            ))
        if "pattern_csv" in staff_columns:
            staff = sa.table(
                "staff",
                sa.column("pattern_override", sa.Boolean()),
                sa.column("pattern_csv", sa.String()),
            )
            op.execute(
                staff.update()
                .where(sa.func.coalesce(staff.c.pattern_csv, "") != "")
                .values(pattern_override=sa.true())
            )


def downgrade():
    raise RuntimeError(
        "Roster-pattern inheritance cannot be safely removed."
    )
