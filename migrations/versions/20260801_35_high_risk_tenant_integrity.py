"""Enforce high-risk operational tenant relationships.

Revision ID: 20260801_35
Revises: 20260801_34

PostgreSQL is the production database. SQLite remains a development/test
compatibility target and cannot add these constraints without destructive table
rebuilds, so the production boundary is deliberately PostgreSQL-only.
"""

from __future__ import annotations

import os

from alembic import op
import sqlalchemy as sa


revision = "20260801_35"
down_revision = "20260801_34"
branch_labels = None
depends_on = None


# Parent tables referenced by composite (unit_id, id) foreign keys.
PARENTS = (
    "annotation_type",
    "assignment",
    "operational_position",
    "person_qualification",
    "position_currency_category",
    "position_participant_role",
    "position_session",
    "qualification_type",
    "roster_publication",
    "shift_request",
    "staff",
    "training_level",
    "training_objective",
    "training_session",
    "watch",
)


# (child table, child reference column, parent table)
RELATIONSHIPS = (
    ("assignment", "staff_id", "staff"),
    ("leave", "staff_id", "staff"),
    ("sickness", "staff_id", "staff"),
    ("shift_request", "staff_id", "staff"),
    ("shift_request", "resulting_assignment_id", "assignment"),
    ("request_audit", "request_id", "shift_request"),
    ("notification", "recipient_id", "staff"),
    ("annotation_audit", "annotation_type_id", "annotation_type"),
    ("annotation_audit", "assignment_id", "assignment"),
    ("staff_watch_history", "staff_id", "staff"),
    ("staff_watch_history", "watch_id", "watch"),
    ("person_qualification", "person_id", "staff"),
    ("person_qualification", "qualification_type_id", "qualification_type"),
    ("person_qualification_history", "person_qualification_id", "person_qualification"),
    ("roster_acknowledgement", "publication_id", "roster_publication"),
    ("roster_acknowledgement", "person_id", "staff"),
    ("training_objective", "level_id", "training_level"),
    ("training_session", "trainee_id", "staff"),
    ("training_session", "ojti_id", "staff"),
    ("training_session", "level_id", "training_level"),
    ("training_score", "session_id", "training_session"),
    ("training_score", "objective_id", "training_objective"),
    ("position_status_event", "position_id", "operational_position"),
    ("position_status_event", "actor_id", "staff"),
    ("position_session", "position_id", "operational_position"),
    ("position_session", "primary_person_id", "staff"),
    ("position_session", "currency_category_id", "position_currency_category"),
    ("position_session", "created_by_id", "staff"),
    ("position_session", "corrected_by_id", "staff"),
    ("position_session_participant", "session_id", "position_session"),
    ("position_session_participant", "person_id", "staff"),
    ("position_session_participant", "role_id", "position_participant_role"),
    ("position_session_audit", "session_id", "position_session"),
    ("position_session_audit", "position_id", "operational_position"),
    ("position_session_audit", "actor_id", "staff"),
    ("position_endorsement", "person_id", "staff"),
    ("position_endorsement", "position_id", "operational_position"),
    ("position_requirement", "position_id", "operational_position"),
    ("break_plan", "person_id", "staff"),
    ("break_plan", "position_id", "operational_position"),
    ("achieved_duty", "person_id", "staff"),
    ("achieved_duty", "planned_assignment_id", "assignment"),
    ("fatigue_report", "person_id", "staff"),
    ("controller_kiosk_credential", "person_id", "staff"),
    ("mfa_credential", "person_id", "staff"),
)


def _columns(inspector: sa.Inspector, table: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table)}


def _constraint_for_columns(
    inspector: sa.Inspector, table: str, columns: tuple[str, ...]
) -> str | None:
    for constraint in inspector.get_unique_constraints(table):
        if tuple(constraint.get("column_names") or ()) == columns:
            return constraint.get("name")
    return None


def _preflight_relationship(
    bind: sa.Connection, child: str, reference: str, parent: str
) -> None:
    diagnostic = sa.text(
        f'SELECT c.id, c.unit_id, c."{reference}" AS referenced_id '
        f'FROM "{child}" c LEFT JOIN "{parent}" p '
        f'ON p.id = c."{reference}" AND p.unit_id = c.unit_id '
        f'WHERE c."{reference}" IS NOT NULL AND p.id IS NULL '
        "ORDER BY c.id LIMIT 10"
    )
    invalid = [dict(row._mapping) for row in bind.execute(diagnostic)]
    if invalid:
        raise RuntimeError(
            f"Cannot enforce {child}(unit_id, {reference}) -> "
            f"{parent}(unit_id, id): cross-unit or orphaned rows {invalid}. "
            "Correct the referenced ID or tenant ownership under an approved "
            "data-repair change, then rerun the migration."
        )


def _scope_live_transaction_key(inspector: sa.Inspector, table: str) -> None:
    global_name = _constraint_for_columns(inspector, table, ("transaction_key",))
    scoped_name = _constraint_for_columns(
        inspector, table, ("unit_id", "transaction_key")
    )
    if global_name:
        op.drop_constraint(global_name, table, type_="unique")
    if not scoped_name:
        op.create_unique_constraint(
            f"uq_{table}_unit_transaction_key",
            table,
            ["unit_id", "transaction_key"],
        )


def upgrade() -> None:
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") == "control":
        return
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    for parent in PARENTS:
        if parent not in tables or not {"unit_id", "id"} <= _columns(inspector, parent):
            continue
        if not _constraint_for_columns(inspector, parent, ("unit_id", "id")):
            op.create_unique_constraint(
                f"uq_{parent}_unit_id", parent, ["unit_id", "id"]
            )
            inspector = sa.inspect(bind)

    for child, reference, parent in RELATIONSHIPS:
        if child not in tables or parent not in tables:
            continue
        if not {"unit_id", reference} <= _columns(inspector, child):
            continue
        _preflight_relationship(bind, child, reference, parent)
        constrained = ("unit_id", reference)
        existing = {
            tuple(constraint.get("constrained_columns") or ())
            for constraint in inspector.get_foreign_keys(child)
        }
        if constrained not in existing:
            op.create_foreign_key(
                f"fk_{child}_{reference}_unit",
                child,
                parent,
                ["unit_id", reference],
                ["unit_id", "id"],
            )
            inspector = sa.inspect(bind)

    for table in (
        "position_status_event",
        "position_session",
        "position_session_participant",
    ):
        if table in tables:
            _scope_live_transaction_key(inspector, table)
            inspector = sa.inspect(bind)


def downgrade() -> None:
    raise RuntimeError(
        "High-risk tenant constraints are an irreversible security boundary."
    )
