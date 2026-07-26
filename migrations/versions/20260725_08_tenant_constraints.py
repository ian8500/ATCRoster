"""enforce tenant foreign keys and unit-scoped uniqueness

Revision ID: 20260725_08
Revises: 20260725_07
"""
import os

import sqlalchemy as sa
from alembic import op

revision = "20260725_08"
down_revision = "20260725_07"
branch_labels = None
depends_on = None

NAMING = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
}

TENANT_TABLES = (
    "staff", "watch", "shift_type", "assignment", "shift_request",
    "annotation_type", "requirement", "leave", "sickness", "ai_rule_set",
    "change_log", "staff_watch_history", "roster_setting",
)

SCOPED_UNIQUES = {
    "shift_type": (
        ("code",),
        ("unit_id", "code"),
        "uq_shift_unit_code",
    ),
    "watch": (
        ("name",),
        ("unit_id", "name"),
        "uq_watch_unit_name",
    ),
    "annotation_type": (
        ("code",),
        ("unit_id", "code"),
        "uq_annotation_unit_code",
    ),
    "staff": (
        ("staff_no",),
        ("unit_id", "staff_no"),
        "uq_staff_unit_number",
    ),
    "requirement": (
        ("year", "month"),
        ("unit_id", "year", "month"),
        "uniq_unit_year_month",
    ),
    "roster_setting": (
        ("key",),
        ("unit_id", "key"),
        "uq_roster_setting_unit_key",
    ),
    "ai_rule_set": (
        ("year", "month"),
        ("unit_id", "year", "month"),
        "uniq_ai_ruleset_unit_month",
    ),
    "assignment": (
        ("staff_id", "day"),
        ("unit_id", "staff_id", "day"),
        "uniq_unit_staff_day",
    ),
    "shift_request": (
        ("staff_id", "day"),
        ("unit_id", "staff_id", "day"),
        "uniq_shift_request_unit_staff_day",
    ),
}


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    schema_role = os.environ.get(
        "ATCROSTER_SCHEMA_ROLE", "combined"
    ).lower()
    for table in (() if schema_role == "operational" else TENANT_TABLES):
        if table not in tables:
            continue
        inspector = sa.inspect(bind)
        columns = {
            column["name"] for column in inspector.get_columns(table)
        }
        if "unit_id" not in columns:
            continue
        unit_fks = [
            key for key in inspector.get_foreign_keys(table)
            if key.get("constrained_columns") == ["unit_id"]
        ]
        if not unit_fks:
            with op.batch_alter_table(
                table, naming_convention=NAMING
            ) as batch:
                batch.create_foreign_key(
                    f"fk_{table}_unit_id_unit",
                    "unit", ["unit_id"], ["id"],
                )

    for table, (old_columns, new_columns, new_name) in SCOPED_UNIQUES.items():
        if table not in tables:
            continue
        inspector = sa.inspect(bind)
        uniques = inspector.get_unique_constraints(table)
        indexes = inspector.get_indexes(table)
        existing_sets = {
            tuple(item.get("column_names") or ())
            for item in (*uniques, *indexes)
            if item.get("unique", True)
        }
        if tuple(new_columns) in existing_sets:
            continue
        old_constraints = [
            item for item in uniques
            if tuple(item.get("column_names") or ()) == tuple(old_columns)
        ]
        with op.batch_alter_table(
            table, naming_convention=NAMING
        ) as batch:
            for constraint in old_constraints:
                name = constraint.get("name") or (
                    f"uq_{table}_{old_columns[0]}"
                )
                batch.drop_constraint(name, type_="unique")
            batch.create_unique_constraint(new_name, list(new_columns))


def downgrade():
    raise RuntimeError(
        "Tenant constraints are an irreversible security boundary."
    )
