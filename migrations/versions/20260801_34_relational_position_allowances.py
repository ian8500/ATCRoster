"""Normalize weekly position time allowances.

Revision ID: 20260801_34
Revises: 20260801_33
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from alembic import op
import sqlalchemy as sa


revision = "20260801_34"
down_revision = "20260801_33"
branch_labels = None
depends_on = None


def _validated_matrix(raw_value: object, position_id: int) -> dict[str, int]:
    try:
        document = json.loads(str(raw_value or "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Position {position_id} has an invalid maximum-time matrix; "
            "repair it before upgrading."
        ) from exc
    if not isinstance(document, dict):
        raise RuntimeError(
            f"Position {position_id} maximum-time matrix must be an object."
        )
    validated: dict[str, int] = {}
    for raw_key, raw_minutes in document.items():
        try:
            parts = str(raw_key).split(":")
            if len(parts) != 2:
                raise ValueError
            weekday, start_hour = (int(part) for part in parts)
            if isinstance(raw_minutes, bool):
                raise ValueError
            minutes = int(raw_minutes)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Position {position_id} has an invalid maximum-time slot."
            ) from exc
        if not 0 <= weekday <= 6 or not 0 <= start_hour <= 23:
            raise RuntimeError(
                f"Position {position_id} has a maximum-time slot outside the weekly matrix."
            )
        if not 1 <= minutes <= 1440:
            raise RuntimeError(
                f"Position {position_id} maximum time must be 1 to 1,440 minutes."
            )
        canonical_key = f"{weekday}:{start_hour}"
        if canonical_key in validated:
            raise RuntimeError(
                f"Position {position_id} has duplicate maximum-time slots."
            )
        validated[canonical_key] = minutes
    return validated


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, unit_id, maximum_session_duration_matrix_json "
            "FROM operational_position"
        )
    ).mappings()
    matrices = [
        (
            int(row["id"]),
            int(row["unit_id"]),
            _validated_matrix(
                row["maximum_session_duration_matrix_json"], int(row["id"])
            ),
        )
        for row in rows
    ]

    with op.batch_alter_table("operational_position") as batch:
        batch.create_unique_constraint("uq_position_unit_id", ["unit_id", "id"])

    op.create_table(
        "operational_position_time_allowance",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=False),
        sa.Column("weekday", sa.Integer(), nullable=False),
        sa.Column("start_hour", sa.Integer(), nullable=False),
        sa.Column("maximum_duration_minutes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["unit_id", "position_id"],
            ["operational_position.unit_id", "operational_position.id"],
            name="fk_position_allowance_position_unit",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "unit_id",
            "position_id",
            "weekday",
            "start_hour",
            name="uq_position_allowance_slot",
        ),
        sa.CheckConstraint(
            "weekday >= 0 AND weekday <= 6",
            name="ck_position_allowance_weekday",
        ),
        sa.CheckConstraint(
            "start_hour >= 0 AND start_hour <= 23",
            name="ck_position_allowance_start_hour",
        ),
        sa.CheckConstraint(
            "maximum_duration_minutes >= 1 AND maximum_duration_minutes <= 1440",
            name="ck_position_allowance_duration",
        ),
    )
    op.create_index(
        "ix_position_allowance_lookup",
        "operational_position_time_allowance",
        ["unit_id", "position_id", "weekday", "start_hour"],
    )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    allowance_table = sa.table(
        "operational_position_time_allowance",
        sa.column("unit_id", sa.Integer()),
        sa.column("position_id", sa.Integer()),
        sa.column("weekday", sa.Integer()),
        sa.column("start_hour", sa.Integer()),
        sa.column("maximum_duration_minutes", sa.Integer()),
        sa.column("created_at", sa.DateTime()),
        sa.column("updated_at", sa.DateTime()),
    )
    converted = []
    for position_id, unit_id, matrix in matrices:
        for key, minutes in matrix.items():
            weekday, start_hour = (int(part) for part in key.split(":"))
            converted.append(
                {
                    "unit_id": unit_id,
                    "position_id": position_id,
                    "weekday": weekday,
                    "start_hour": start_hour,
                    "maximum_duration_minutes": minutes,
                    "created_at": now,
                    "updated_at": now,
                }
            )
    if converted:
        op.bulk_insert(allowance_table, converted)

    with op.batch_alter_table("operational_position") as batch:
        batch.drop_column("maximum_session_duration_matrix_json")


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    with op.batch_alter_table("operational_position") as batch:
        batch.add_column(
            sa.Column(
                "maximum_session_duration_matrix_json",
                sa.Text(),
                nullable=False,
                server_default="{}",
            )
        )
    rows = bind.execute(
        sa.text(
            "SELECT position_id, weekday, start_hour, maximum_duration_minutes "
            "FROM operational_position_time_allowance ORDER BY position_id, weekday, start_hour"
        )
    ).mappings()
    matrices: dict[int, dict[str, int]] = {}
    for row in rows:
        matrices.setdefault(int(row["position_id"]), {})[
            f"{int(row['weekday'])}:{int(row['start_hour'])}"
        ] = int(row["maximum_duration_minutes"])
    for position_id, matrix in matrices.items():
        bind.execute(
            sa.text(
                "UPDATE operational_position "
                "SET maximum_session_duration_matrix_json=:matrix WHERE id=:position_id"
            ),
            {"matrix": json.dumps(matrix, sort_keys=True), "position_id": position_id},
        )
    op.drop_table("operational_position_time_allowance")
    with op.batch_alter_table("operational_position") as batch:
        batch.drop_constraint("uq_position_unit_id", type_="unique")
