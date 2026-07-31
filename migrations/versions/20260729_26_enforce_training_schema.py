"""enforce training schema by physical database contents

Revision ID: 20260729_26
Revises: 20260729_25
"""
import sqlalchemy as sa
from alembic import op

revision = "20260729_26"
down_revision = "20260729_25"
branch_labels = None
depends_on = None


def _table_names(bind):
    return set(sa.inspect(bind).get_table_names())


def upgrade():
    bind = op.get_bind()
    tables = _table_names(bind)
    # Control databases never contain staff. Determine the database role from
    # its physical schema so this repair cannot be skipped by stale process
    # environment or route metadata.
    if "staff" not in tables:
        return
    staff_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("staff")
    }
    if "caa_license_number" not in staff_columns:
        with op.batch_alter_table("staff") as batch:
            batch.add_column(sa.Column(
                "caa_license_number", sa.String(40), nullable=False,
                server_default="",
            ))
    if "training_level" not in tables:
        op.create_table(
            "training_level",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column("name", sa.String(80), nullable=False),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.UniqueConstraint(
                "unit_id", "name", name="uq_training_level_unit_name"
            ),
        )
    tables = _table_names(bind)
    if "training_objective" not in tables:
        op.create_table(
            "training_objective",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column(
                "level_id", sa.Integer(), sa.ForeignKey("training_level.id"),
                nullable=False, index=True,
            ),
            sa.Column("position", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(100), nullable=False),
            sa.Column("description", sa.Text(), nullable=False, server_default=""),
            sa.UniqueConstraint(
                "level_id", "position", name="uq_training_objective_position"
            ),
        )
    if "training_session" not in tables:
        op.create_table(
            "training_session",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column(
                "trainee_id", sa.Integer(), sa.ForeignKey("staff.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "ojti_id", sa.Integer(), sa.ForeignKey("staff.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "level_id", sa.Integer(), sa.ForeignKey("training_level.id"),
                nullable=False, index=True,
            ),
            sa.Column("training_date", sa.Date(), nullable=False, index=True),
            sa.Column("duration_minutes", sa.Integer(), nullable=False),
            sa.Column("summary", sa.Text(), nullable=False, server_default=""),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
        )
    tables = _table_names(bind)
    if "training_score" not in tables:
        op.create_table(
            "training_score",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column(
                "session_id", sa.Integer(), sa.ForeignKey("training_session.id"),
                nullable=False, index=True,
            ),
            sa.Column(
                "objective_id", sa.Integer(),
                sa.ForeignKey("training_objective.id"),
                nullable=False, index=True,
            ),
            sa.Column("attainment", sa.Integer(), nullable=False),
            sa.Column("assistance", sa.Integer(), nullable=False),
            sa.Column(
                "safety_critical", sa.Boolean(), nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("note", sa.Text(), nullable=False, server_default=""),
            sa.UniqueConstraint(
                "session_id", "objective_id",
                name="uq_training_score_objective",
            ),
        )


def downgrade():
    pass
