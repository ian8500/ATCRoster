"""Add configurable watch handover module.

Revision ID: 20260808_54
Revises: 20260808_53
"""

import os
from datetime import datetime, timezone
import json

from alembic import op
import sqlalchemy as sa


revision = "20260808_54"
down_revision = "20260808_53"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        bind = op.get_bind()
        existing = set(sa.inspect(bind).get_table_names())
        if {"unit", "feature_flag"} <= existing:
            unit = sa.table(
                "unit", sa.column("id", sa.Integer()),
                sa.column("status", sa.String()),
            )
            feature = sa.table(
                "feature_flag", sa.column("unit_id", sa.Integer()),
                sa.column("key", sa.String()), sa.column("enabled", sa.Boolean()),
            )
            enabled_units = set(bind.execute(sa.select(feature.c.unit_id).where(
                feature.c.key == "handover_module"
            )).scalars())
            unit_ids = bind.execute(sa.select(unit.c.id).where(
                unit.c.status != "platform_control"
            )).scalars()
            for unit_id in unit_ids:
                if unit_id not in enabled_units:
                    bind.execute(feature.insert().values(
                        unit_id=unit_id, key="handover_module", enabled=True,
                    ))
        return
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "handover_field" not in existing:
        op.create_table(
            "handover_field",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("section_name", sa.String(80), nullable=False, server_default="Operational overview"),
            sa.Column("label", sa.String(120), nullable=False),
            sa.Column("field_type", sa.String(20), nullable=False, server_default="text"),
            sa.Column("options_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("help_text", sa.String(240), nullable=False, server_default=""),
            sa.Column("placeholder", sa.String(160), nullable=False, server_default=""),
            sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("display_order", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_handover_field_unit_id", "handover_field", ["unit_id"])
    if "handover_record" not in existing:
        op.create_table(
            "handover_record",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="published"),
            sa.Column("created_by_id", sa.Integer(), nullable=False),
            sa.Column("created_by_name", sa.String(80), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("target_shift_day", sa.Date()),
            sa.Column("target_shift_code", sa.String(10), nullable=False, server_default=""),
            sa.Column("target_shift_name", sa.String(80), nullable=False, server_default=""),
            sa.Column("target_shift_start", sa.DateTime()),
            sa.Column("next_shift_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("responses_json", sa.Text(), nullable=False, server_default="[]"),
        )
        op.create_index("ix_handover_record_unit_id", "handover_record", ["unit_id"])
        op.create_index("ix_handover_record_status", "handover_record", ["status"])
        op.create_index("ix_handover_record_created_by_id", "handover_record", ["created_by_id"])
        op.create_index("ix_handover_record_created_at", "handover_record", ["created_at"])
        op.create_index("ix_handover_record_target_shift_day", "handover_record", ["target_shift_day"])
    bind = op.get_bind()
    if {"unit", "handover_field"} <= set(sa.inspect(bind).get_table_names()):
        metadata = sa.MetaData()
        unit_table = sa.Table("unit", metadata, autoload_with=bind)
        field_table = sa.Table("handover_field", metadata, autoload_with=bind)
        configured = set(bind.execute(sa.select(field_table.c.unit_id)).scalars())
        unit_ids = bind.execute(sa.select(unit_table.c.id).where(
            unit_table.c.status != "platform_control"
        )).scalars()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        defaults = (
            ("Operational overview", "Operational status", "select", ["Normal", "Degraded", "Contingency"], "Select the overall operational state.", True),
            ("Operational overview", "Key operational information", "text", [], "Record constraints, coordination or events the incoming watch must know.", True),
            ("Equipment & systems", "Equipment and service status", "text", [], "Include outages, restrictions and outstanding actions.", False),
            ("Incoming priorities", "Priorities for the incoming watch", "text", [], "List the most important actions in priority order.", False),
        )
        for unit_id in unit_ids:
            if unit_id in configured:
                continue
            for order, values in enumerate(defaults, start=1):
                section, label, field_type, options, help_text, required = values
                bind.execute(field_table.insert().values(
                    unit_id=unit_id, section_name=section, label=label,
                    field_type=field_type, options_json=json.dumps(options),
                    help_text=help_text, placeholder="", required=required,
                    active=True, display_order=order * 10,
                    created_at=now, updated_at=now,
                ))


def downgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        bind = op.get_bind()
        if "feature_flag" in sa.inspect(bind).get_table_names():
            feature = sa.table(
                "feature_flag", sa.column("key", sa.String()),
            )
            bind.execute(feature.delete().where(feature.c.key == "handover_module"))
        return
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    for table in ("handover_record", "handover_field"):
        if table in existing:
            op.drop_table(table)
