"""Complete the roster-impact queue and recalculation audit.

Revision ID: 20260803_49
Revises: 20260803_48
"""

import os

from alembic import op
import sqlalchemy as sa


revision = "20260803_49"
down_revision = "20260803_48"
branch_labels = None
depends_on = None


AUDIT_COLUMNS = (
    sa.Column("started_at", sa.DateTime()),
    sa.Column("affected_dates", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("assignments_created", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("baselines_changed", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("baselines_removed", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("overrides_retained", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("redundant_overrides_found", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("exceptions_created", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("warnings_created", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("error_message", sa.String(2000), nullable=False, server_default=""),
)


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "roster_impact_event" in tables:
        op.execute("UPDATE roster_impact_event SET status='RUNNING' WHERE status='PROCESSING'")
        existing = {c["name"] for c in sa.inspect(bind).get_columns("roster_impact_event")}
        with op.batch_alter_table("roster_impact_event") as batch:
            batch.drop_constraint("ck_roster_impact_event_status", type_="check")
            batch.alter_column("status", type_=sa.String(30), existing_type=sa.String(20))
            for column in AUDIT_COLUMNS:
                if column.name not in existing:
                    batch.add_column(column)
            batch.create_check_constraint(
                "ck_roster_impact_event_status",
                "status IN ('PENDING','RUNNING','COMPLETED','COMPLETED_WITH_WARNINGS','FAILED')",
            )
    if "roster_impact_exception" in tables:
        temporary_watch = "watch" not in tables
        if temporary_watch:
            # Some supported legacy fixtures predate the watch table, while
            # migration 43 still leaves a forward FK reference to it. SQLite
            # batch reflection needs the referenced table to exist briefly.
            op.create_table("watch", sa.Column("id", sa.Integer(), primary_key=True))
        op.execute("UPDATE roster_impact_exception SET status='NOT_APPLICABLE' WHERE status='DISMISSED'")
        with op.batch_alter_table("roster_impact_exception") as batch:
            batch.drop_constraint("ck_roster_impact_exception_status", type_="check")
            batch.create_check_constraint(
                "ck_roster_impact_exception_status",
                "status IN ('OPEN','ACKNOWLEDGED','RESOLVED','NOT_APPLICABLE')",
            )
        if temporary_watch:
            op.drop_table("watch")


def downgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = sa.inspect(bind).get_table_names()
    if "roster_impact_exception" in tables:
        op.execute("UPDATE roster_impact_exception SET status='DISMISSED' WHERE status='NOT_APPLICABLE'")
        with op.batch_alter_table("roster_impact_exception") as batch:
            batch.drop_constraint("ck_roster_impact_exception_status", type_="check")
            batch.create_check_constraint(
                "ck_roster_impact_exception_status",
                "status IN ('OPEN','ACKNOWLEDGED','RESOLVED','DISMISSED')",
            )
    if "roster_impact_event" in tables:
        op.execute("UPDATE roster_impact_event SET status='COMPLETED' WHERE status='COMPLETED_WITH_WARNINGS'")
        op.execute("UPDATE roster_impact_event SET status='PROCESSING' WHERE status IN ('PENDING','RUNNING','FAILED')")
        existing = {c["name"] for c in sa.inspect(bind).get_columns("roster_impact_event")}
        with op.batch_alter_table("roster_impact_event") as batch:
            batch.drop_constraint("ck_roster_impact_event_status", type_="check")
            batch.alter_column("status", type_=sa.String(20), existing_type=sa.String(30))
            batch.create_check_constraint(
                "ck_roster_impact_event_status", "status IN ('PROCESSING','COMPLETED')",
            )
            for column in reversed(AUDIT_COLUMNS):
                if column.name in existing:
                    batch.drop_column(column.name)
