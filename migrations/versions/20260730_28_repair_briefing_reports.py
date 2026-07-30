"""ensure the briefing reports table exists in operational databases

Revision ID: 20260730_28
Revises: 20260730_27
"""
import sqlalchemy as sa
from alembic import op

revision = "20260730_28"
down_revision = "20260730_27"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    # Control databases do not contain airport staff or briefing reports.
    if "staff" not in tables or "briefing_assurance_run" in tables:
        return
    op.create_table(
        "briefing_assurance_run",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
        sa.Column("operational_date", sa.Date(), nullable=False, index=True),
        sa.Column("run_by_id", sa.Integer(), nullable=False),
        sa.Column("run_by_name", sa.String(80), nullable=False),
        sa.Column(
            "run_at", sa.DateTime(), nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("result_json", sa.Text(), nullable=False, server_default="{}"),
    )


def downgrade():
    # Revision 21 owns this table. This repair is intentionally non-destructive.
    pass
