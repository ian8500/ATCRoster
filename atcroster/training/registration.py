"""Composition of the legacy training route surface."""

from __future__ import annotations

from typing import Any

from training_blueprint import create_training_blueprint, create_training_dependencies


def register_training_blueprint(
    app: Any, *, db: Any, operational_models: Any, saas_models: Any,
    services: Any,
) -> None:
    """Register training routes from training-owned runtime dependencies."""
    app.register_blueprint(create_training_blueprint(create_training_dependencies(
        db=db, operational_models=operational_models, saas_models=saas_models,
        current_unit_id=services.current_unit_id,
        training_enabled=services.training_enabled,
        is_editor_user=services.is_editor_user,
        can_manage_training=services.can_manage_training,
        can_record_training=services.can_record_training,
        is_under_training=services.is_under_training,
        training_profile_allowed=services.training_profile_allowed,
        validate_csrf=services.validate_csrf,
        competency_enabled=services.competency_enabled,
        is_admin_user=services.is_admin_user, utcnow=services.now,
        record_qualification_history=services.record_qualification_history,
        sync_qualification_to_roster_profile=services.sync_qualification_to_roster_profile,
        record_qualification_roster_impact=services.record_qualification_roster_impact,
    )))
