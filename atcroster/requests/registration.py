"""Composition of absence and shift-request routes."""

from __future__ import annotations

from typing import Any

from absence_requests_blueprint import (
    create_absence_request_dependencies,
    create_absence_requests_blueprint,
)


def register_request_blueprints(
    app: Any, *, db: Any, operational_models: Any, services: Any
) -> None:
    """Register request routes with request-workflow runtime services."""
    app.register_blueprint(create_absence_requests_blueprint(
        create_absence_request_dependencies(
            db=db, operational_models=operational_models,
            is_admin_user=services.is_admin_user,
            parse_year_month=services.parse_year_month,
            month_range=services.month_range,
            clamp_prev_next=services.clamp_prev_next,
            validate_csrf=services.validate_csrf,
            get_absence_types=services.get_absence_types,
            save_absence_types=services.save_absence_types,
            tenant_get=services.tenant_get,
            current_unit_id=services.current_unit_id,
            refresh_day_from_pattern_and_leave=services.refresh_day,
            group_sickness_instances=services.group_sickness_instances,
            workflow=services.workflow, utcnow=services.now,
            request_statuses=services.request_statuses,
            request_transitions=services.request_transitions,
            would_create_new_fatigue_issues=services.new_fatigue_issues,
            staff_has_shift_qualification=services.staff_has_shift_qualification,
            can_override_roster_conflicts=services.can_override_conflicts,
            lock_roster_month=services.lock_roster_month,
            record_toil_transaction=services.record_toil_transaction,
        )
    ))
