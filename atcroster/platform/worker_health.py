"""Platform provisioning worker health route."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, jsonify
from flask_login import current_user, login_required


def load_worker_health_snapshot(
    application_module: Any, *, stale_after_seconds: int
) -> dict[str, Any]:
    """Load provisioning code only when the privileged probe is requested."""
    from platform_provisioning import worker_health_snapshot

    return worker_health_snapshot(
        application_module,
        stale_after_seconds=stale_after_seconds,
    )


def operational_routes_ready(
    *, db: Any, Unit: Any, DatabaseRoutingMetadata: Any
) -> bool:
    """Check that every active operational unit has usable database routing."""
    active_units = Unit.query.filter(Unit.status == "active", Unit.code != "CTRL").all()
    for unit in active_units:
        routing = db.session.get(DatabaseRoutingMetadata, unit.id)
        if (
            not routing
            or not routing.secret_name
            or not os.environ.get(routing.secret_name)
        ):
            return False
    return True


@dataclass(frozen=True)
class WorkerHealthDependencies:
    application_module: Any
    metrics: Any
    worker_health_snapshot: Callable[..., dict[str, Any]]


def create_worker_health_blueprint(dependencies: WorkerHealthDependencies) -> Blueprint:
    blueprint = Blueprint("platform_worker_health", __name__)

    @login_required
    def platform_worker_health():
        if getattr(current_user, "role", "") != "superadmin":
            abort(403)
        snapshot = dependencies.worker_health_snapshot(
            dependencies.application_module,
            stale_after_seconds=int(
                os.environ.get("ATCROSTER_PROVISIONING_LEASE_SECONDS", "120")
            )
            * 2,
        )
        dependencies.metrics.set("worker_queue_depth", snapshot["queue_depth"])
        dependencies.metrics.set(
            "worker_oldest_queued_age_seconds", snapshot["oldest_queued_age_seconds"]
        )
        dependencies.metrics.set("stale_workers", snapshot["stale_workers"])
        return jsonify(snapshot), 200 if snapshot["status"] == "ready" else 503

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/platform/worker-health",
            "platform_worker_health",
            platform_worker_health,
            methods=("GET",),
        )

    return blueprint
