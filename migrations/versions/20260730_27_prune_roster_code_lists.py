"""remove roster-code list entries without a matching shift type

Revision ID: 20260730_27
Revises: 20260729_26
"""
import json

import sqlalchemy as sa
from alembic import op

revision = "20260730_27"
down_revision = "20260729_26"
branch_labels = None
depends_on = None

CODE_LIST_KEYS = {
    "working_codes",
    "banned_codes",
    "exclude_from_counters",
    "non_working_codes",
}


def upgrade():
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if not {"roster_setting", "shift_type"}.issubset(tables):
        return
    settings = sa.table(
        "roster_setting",
        sa.column("id", sa.Integer()),
        sa.column("unit_id", sa.Integer()),
        sa.column("key", sa.String()),
        sa.column("value", sa.Text()),
    )
    shifts = sa.table(
        "shift_type",
        sa.column("unit_id", sa.Integer()),
        sa.column("code", sa.String()),
    )
    valid_by_unit = {}
    for unit_id, code in bind.execute(
        sa.select(shifts.c.unit_id, shifts.c.code)
    ):
        valid_by_unit.setdefault(unit_id, set()).add(
            str(code or "").strip().upper()
        )
    rows = bind.execute(
        sa.select(
            settings.c.id,
            settings.c.unit_id,
            settings.c.value,
        ).where(settings.c.key.in_(CODE_LIST_KEYS))
    ).all()
    for row_id, unit_id, raw_value in rows:
        try:
            values = json.loads(raw_value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            values = []
        valid = valid_by_unit.get(unit_id, set())
        cleaned = []
        for value in values:
            code = str(value or "").strip().upper()
            if code and code in valid and code not in cleaned:
                cleaned.append(code)
        bind.execute(
            settings.update().where(settings.c.id == row_id).values(
                value=json.dumps(cleaned)
            )
        )


def downgrade():
    # Removed references pointed to codes that did not exist, so they cannot
    # be reconstructed safely.
    pass
