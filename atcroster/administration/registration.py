"""Composition of administration-owned route registrations."""

from __future__ import annotations

from typing import Any

from .blueprint import (
    create_admin_dashboard_blueprint,
    create_admin_dashboard_dependencies,
    create_administration_blueprint,
    create_administration_dependencies,
)
from .context import create_admin_context_dependencies
from .lifecycle import (
    create_staff_lifecycle_blueprint,
    create_staff_lifecycle_dependencies,
)
from .onboarding import create_onboarding_blueprint, create_onboarding_dependencies
from .reference import create_reference_data_blueprint, create_reference_data_dependencies
from .staff_edit import create_staff_edit_blueprint, create_staff_edit_dependencies
from .toil import (
    create_toil_administration_blueprint,
    create_toil_administration_dependencies,
)
from .watch_moves import create_watch_move_blueprint, create_watch_move_dependencies


def register_administration_blueprints(
    app: Any, *, db: Any, operational_models: Any, saas_models: Any,
    roster_impact_event_type: Any, services: Any,
) -> None:
    """Register the routes owned by administration with explicit runtime services."""
    app.register_blueprint(create_administration_blueprint(
        create_administration_dependencies(
            is_admin_user=services.is_admin_user,
            live_position_enabled=services.live_position_enabled,
        )
    ))
    app.register_blueprint(create_admin_dashboard_blueprint(
        create_admin_dashboard_dependencies(
            is_admin_user=services.is_admin_user,
            actions=services.admin_actions(),
            context=create_admin_context_dependencies(
                db=db, operational_models=operational_models, saas_models=saas_models,
                current_unit_id=services.current_unit_id,
                roster_settings_snapshot=services.roster_settings_snapshot,
                validate_pattern=services.validate_pattern,
                shift_counter_group=services.shift_counter_group,
                sms_number_options=services.sms_number_options,
                sms_operational_options=services.sms_operational_options,
                sms_default_number=services.sms_default_number,
                absence_types=services.absence_types,
                default_base_pattern=services.default_base_pattern,
                pattern_codes=services.pattern_codes,
            ),
        )
    ))
    app.register_blueprint(create_staff_edit_blueprint(create_staff_edit_dependencies(
        db=db, operational_models=operational_models, saas_models=saas_models,
        roster_impact_event_type=roster_impact_event_type,
        current_unit_id=services.current_unit_id, parse_date=services.parse_date,
        valid_email=services.valid_email, normalise_phone=services.normalise_phone,
        validate_pattern=services.validate_pattern, now=services.now,
        record_qualification_history=services.record_qualification_history,
        record_roster_impact=services.record_roster_impact,
        user_permissions=services.user_permissions,
        admin_required=services.admin_required, pattern_codes=services.pattern_codes,
    )))
    app.register_blueprint(create_onboarding_blueprint(create_onboarding_dependencies(
        db=db, operational_models=operational_models, saas_models=saas_models,
        current_unit_id=services.current_unit_id,
        is_admin_user=services.is_admin_user, validate_csrf=services.validate_csrf,
    )))
    app.register_blueprint(create_reference_data_blueprint(
        create_reference_data_dependencies(
            db=db, operational_models=operational_models,
            current_unit_id=services.current_unit_id, validate_csrf=services.validate_csrf,
            refresh_annotation_cache=services.refresh_annotation_cache,
            normalise_codes=services.normalise_codes,
            save_codes_setting=services.save_codes_setting,
            prune_roster_code_settings=services.prune_roster_code_settings,
            working_codes=services.working_codes,
            banned_codes=services.banned_codes,
            excluded_codes=services.excluded_codes,
            non_working_codes=services.non_working_codes,
            admin_required=services.admin_required,
        )
    ))
    app.register_blueprint(create_staff_lifecycle_blueprint(
        create_staff_lifecycle_dependencies(
            db=db, operational_models=operational_models,
            roster_impact_event_type=roster_impact_event_type,
            current_unit_id=services.current_unit_id, parse_date=services.parse_date,
            record_roster_impact=services.record_roster_impact,
            admin_required=services.admin_required,
        )
    ))
    app.register_blueprint(create_watch_move_blueprint(create_watch_move_dependencies(
        db=db, operational_models=operational_models,
        roster_impact_event_type=roster_impact_event_type,
        current_unit_id=services.current_unit_id,
        is_admin_user=services.is_admin_user,
        record_roster_impact=services.record_roster_impact,
        log_change=services.log_change,
    )))
    app.register_blueprint(create_toil_administration_blueprint(
        create_toil_administration_dependencies(
            db=db, operational_models=operational_models,
            current_unit_id=services.current_unit_id,
            is_admin_user=services.is_admin_user, validate_csrf=services.validate_csrf,
            record_toil_transaction=services.record_toil_transaction,
        )
    ))
