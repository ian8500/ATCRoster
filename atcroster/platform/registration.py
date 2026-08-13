"""Composition of platform administration routes."""

from __future__ import annotations

from typing import Any

from .admin import create_platform_admin_blueprint, create_platform_admin_dependencies
from .worker_health import create_worker_health_blueprint, WorkerHealthDependencies


def register_platform_blueprints(
    app: Any, *, db: Any, operational_models: Any, saas_models: Any,
    application_module: Any, services: Any,
) -> None:
    """Register privileged platform routes from platform-owned dependencies."""
    app.register_blueprint(create_worker_health_blueprint(WorkerHealthDependencies(
        application_module=application_module, metrics=services.metrics,
        worker_health_snapshot=services.worker_health_snapshot,
    )))
    app.register_blueprint(create_platform_admin_blueprint(
        create_platform_admin_dependencies(
            db=db, operational_models=operational_models, saas_models=saas_models,
            now=services.now, validate_csrf=services.validate_csrf,
            consume_rate_limit=services.consume_rate_limit,
            security_event=services.security_event,
            feature_flags=services.feature_flags,
            module_feature_flags=services.module_feature_flags,
        )
    ))
