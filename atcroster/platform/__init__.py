"""Platform-level administration routes."""

from .worker_health import WorkerHealthDependencies, create_worker_health_blueprint
from .legacy_migrations import add_role_and_calendar_token

__all__ = (
    "WorkerHealthDependencies",
    "add_role_and_calendar_token",
    "create_worker_health_blueprint",
)
