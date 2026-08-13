"""Platform-level administration routes."""

from .bootstrap import LegacyBootstrapService
from .worker_health import (
    WorkerHealthDependencies,
    create_worker_health_blueprint,
    load_worker_health_snapshot,
    operational_routes_ready,
)
from .legacy_migrations import (
    add_assignment_annotation,
    add_columns_if_missing,
    add_performance_indexes,
    add_role_and_calendar_token,
    add_unique_assignment_key,
    add_invitation_target,
    add_toil_and_leave_fields,
    add_watch_pattern_configuration,
    upgrade_tenant_foundation,
)

__all__ = (
    "WorkerHealthDependencies",
    "LegacyBootstrapService",
    "add_assignment_annotation",
    "add_columns_if_missing",
    "add_performance_indexes",
    "add_role_and_calendar_token",
    "add_unique_assignment_key",
    "add_invitation_target",
    "add_toil_and_leave_fields",
    "add_watch_pattern_configuration",
    "upgrade_tenant_foundation",
    "create_worker_health_blueprint",
    "load_worker_health_snapshot",
    "operational_routes_ready",
)
