"""Canonical extension ownership and tenant-aware database routing."""

from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy.session import Session as FlaskSqlAlchemySession

from tenancy import operational_engine_for_authenticated_unit


OPERATIONAL_TABLE_NAMES = frozenset({
    "roster_setting", "annotation_type", "watch", "staff", "shift_type",
    "requirement", "special_requirement", "leave", "sickness", "assignment",
    "shift_request", "sms_audit", "request_audit", "notification",
    "annotation_audit", "ai_rule_set", "change_log", "staff_watch_history",
    "qualification_type", "person_qualification", "person_qualification_history",
    "roster_publication", "roster_acknowledgement", "scenario",
    "operational_position", "operational_position_group",
    "operational_position_time_allowance", "position_endorsement",
    "position_requirement", "break_plan", "achieved_duty", "fatigue_report",
    "roster_rule_version", "mfa_credential", "briefing_item", "briefing_delivery",
    "briefing_audit", "briefing_assurance_run", "briefing_message_type",
    "handover_field", "handover_record", "handover_operational_state",
    "handover_equipment", "training_level", "training_objective",
    "training_session", "training_score", "position_currency_category",
    "position_participant_role", "position_status_event", "position_session",
    "position_session_participant", "controller_kiosk_credential",
    "position_session_audit", "toil_transaction", "work_pattern",
    "work_pattern_day", "work_pattern_day_allowed_shift", "staff_pattern_assignment",
    "staff_rule", "bank_holiday", "roster_proposal", "roster_proposal_assignment",
    "roster_period", "roster_impact_event", "roster_impact_exception",
})

db: SQLAlchemy | None = None


def create_tenant_database(app, deployment_environment: str) -> SQLAlchemy:
    """Create the single database extension with operational bind routing."""

    class TenantRoutedSession(FlaskSqlAlchemySession):
        def get_bind(self, mapper=None, clause=None, bind=None, **kwargs):
            if bind is not None:
                return bind
            table_name = None
            if mapper is not None:
                selectable = getattr(mapper, "persist_selectable", None)
                table_name = getattr(selectable, "name", None)
                if table_name is None:
                    table_name = getattr(getattr(mapper, "__table__", None), "name", None)
            if table_name in OPERATIONAL_TABLE_NAMES:
                try:
                    return operational_engine_for_authenticated_unit()
                except RuntimeError:
                    if deployment_environment == "production":
                        raise RuntimeError(
                            "Operational database access requires an authenticated airport route."
                        )
            return super().get_bind(mapper=mapper, clause=clause, bind=bind, **kwargs)

    global db
    db = SQLAlchemy(
        app,
        session_options={"expire_on_commit": False, "class_": TenantRoutedSession},
    )
    return db
