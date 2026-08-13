"""Composition of roster-owned route registrations."""

from __future__ import annotations

from typing import Any

from roster_blueprint import create_roster_blueprint, create_roster_dependencies

from .overtime import create_overtime_blueprint, create_overtime_dependencies


def register_roster_blueprints(
    app: Any, *, db: Any, operational_models: Any, saas_models: Any,
    publication_service: Any, services: Any,
) -> None:
    """Register roster and overtime routes from roster-domain dependencies."""
    app.register_blueprint(create_roster_blueprint(create_roster_dependencies(
        db=db, operational_models=operational_models, saas_models=saas_models,
        publication_service=publication_service,
        validate_csrf=services.validate_csrf,
        parse_year_month=services.parse_year_month,
        current_unit_id=services.current_unit_id,
        roster_month_service=services.roster_month_service,
        assignment_runtime=services.assignment_runtime, utcnow=services.now,
        log_change=services.log_change, consume_rate_limit=services.consume_rate_limit,
        requirements_for_day=services.requirements_for_day,
        staff_is_countable_on=services.staff_is_countable_on,
        operational_capability_matrix=services.operational_capability_matrix,
        exclude_from_counters=services.exclude_from_counters,
        get_shift=services.get_shift,
        shift_counter_group_for_day=services.shift_counter_group_for_day,
        night_active_on=services.night_active_on,
        can_edit_roster=services.can_edit_roster,
        banned_roster_codes=services.banned_roster_codes,
        can_apply_annotations=services.can_apply_annotations,
        parse_annotation=services.parse_annotation,
        is_admin_user=services.is_admin_user,
        apply_toil_annotation_delta=services.apply_toil_annotation_delta,
        load_month_roster=services.load_month_roster,
        add_months=services.add_months, shift_groups=services.shift_groups,
        watch_ids_for_staff_on=services.watch_ids_for_staff_on,
        roster_fatigue_flags=services.roster_fatigue_flags,
        roster_fatigue_matrix=services.roster_fatigue_matrix,
        roster_validation=services.roster_validation,
        roster_month_cache=services.roster_month_cache, metrics=services.metrics,
        roster_proposal_service=services.roster_proposal_service,
        get_annotation_groups=services.get_annotation_groups,
    )))
    app.register_blueprint(create_overtime_blueprint(create_overtime_dependencies(
        operational_models=operational_models,
        current_unit_id=services.current_unit_id,
        consume_rate_limit=services.consume_rate_limit,
        is_editor_user=services.is_editor_user, validate_csrf=services.validate_csrf,
        parse_date=services.parse_date,
        compute_candidates=services.compute_overtime_candidates,
        can_send_messages=services.can_send_messages, send_sms=services.send_sms,
        default_sms_body=services.default_sms_body,
        sms_configured=services.sms_configured,
    )))
