"""Production logging, metrics, and health primitives.

This module deliberately has no dependency on the legacy application module.
The Flask integration passes explicit callbacks and database objects at the edge.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
import secrets
import threading
import time
from typing import Any

from flask import Response, abort, g, jsonify, request
from flask_login import current_user
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text


SAFE_EVENT_FIELDS = frozenset(
    {
        "actor_id",
        "duration_ms",
        "environment",
        "event",
        "exception_class",
        "http_status",
        "job_type",
        "outcome",
        "request_id",
        "reason",
        "retry_count",
        "route",
        "service",
        "scope",
        "principal",
        "unit_id",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit a stable, single-line production log envelope."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname.lower(),
            "service": "atcroster-web",
            "version": os.environ.get("ATCROSTER_COMMIT_SHA", "unknown")[:64],
            "environment": os.environ.get("ATCROSTER_ENVIRONMENT", "development")[:32],
            "message": record.getMessage(),
        }
        fields = getattr(record, "structured_fields", None)
        if isinstance(fields, dict):
            payload.update(
                {key: fields[key] for key in SAFE_EVENT_FIELDS if key in fields}
            )
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_class"] = record.exc_info[0].__name__
        return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)


def configure_production_logging(app: Any, environment: str) -> None:
    if environment != "production":
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    app.logger.propagate = False


def structured_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    safe = {key: value for key, value in fields.items() if key in SAFE_EVENT_FIELDS}
    safe["event"] = event[:80]
    logger.warning(event[:80], extra={"structured_fields": safe})


class MetricsRegistry:
    """Small provider-neutral Prometheus text collector for one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = (
            defaultdict(float)
        )

    def add(self, name: str, value: float = 1, **labels: str) -> None:
        safe_labels = tuple(
            sorted((key, str(item)[:80]) for key, item in labels.items())
        )
        with self._lock:
            self._values[(name, safe_labels)] += value

    def set(self, name: str, value: float, **labels: str) -> None:
        safe_labels = tuple(
            sorted((key, str(item)[:80]) for key, item in labels.items())
        )
        with self._lock:
            self._values[(name, safe_labels)] = value

    def render(self) -> str:
        with self._lock:
            rows = sorted(self._values.items())
        lines = []
        for (name, labels), value in rows:
            suffix = ""
            if labels:
                encoded = ",".join(
                    f'{key}="{item.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                    for key, item in labels
                )
                suffix = "{" + encoded + "}"
            lines.append(f"atcroster_{name}{suffix} {value:g}")
        return "\n".join(lines) + "\n"


def begin_request(registry: MetricsRegistry) -> float:
    registry.add("active_web_requests", 1)
    return time.monotonic()


def finish_request(
    registry: MetricsRegistry,
    started_at: float,
    *,
    route: str,
    method: str,
    status: int,
) -> float:
    duration = max(0.0, time.monotonic() - started_at)
    labels = {"route": route[:100], "method": method[:10], "status": str(status)}
    registry.add("http_requests_total", 1, **labels)
    registry.add("http_request_duration_seconds_sum", duration, **labels)
    registry.add("http_request_duration_seconds_count", 1, **labels)
    if status >= 400:
        registry.add("http_errors_total", 1, route=route[:100], status=str(status))
    registry.add("active_web_requests", -1)
    return duration


def readiness_snapshot(
    *,
    db: Any,
    alembic_path: str,
    required_tables: set[str] | frozenset[str],
    production: bool,
    redis_limiter: Any | None,
    additional_check: Any | None = None,
) -> tuple[dict[str, str], int]:
    """Return a deliberately terse public readiness result."""
    try:
        connection = db.session.connection()
        connection.execute(text("SELECT 1"))
        present = set(inspect(connection).get_table_names())
        revision = MigrationContext.configure(connection).get_current_revision()
        expected = ScriptDirectory.from_config(
            AlembicConfig(alembic_path)
        ).get_current_head()
        if not required_tables.issubset(present):
            return {"status": "not_ready"}, 503
        if production and revision != expected:
            return {"status": "not_ready"}, 503
        if production and redis_limiter is not None:
            redis_limiter.verify()
        if production and additional_check is not None and not additional_check():
            return {"status": "not_ready"}, 503
        return {"status": "ready"}, 200
    except Exception:
        return {"status": "not_ready"}, 503


def register_operations_routes(
    app: Any,
    *,
    db: Any,
    environment: str,
    limiter: Any,
    metrics: MetricsRegistry,
    required_tables: set[str] | frozenset[str],
    additional_readiness_check: Any | None = None,
    alembic_path: str = "alembic.ini",
) -> None:
    """Register stable global endpoint names around extracted operations logic."""

    def health_live():
        return jsonify(
            {"status": "ok", "service": "atcroster", "environment": environment}
        )

    def health_ready():
        payload, status = readiness_snapshot(
            db=db,
            alembic_path=alembic_path,
            required_tables=required_tables,
            production=environment == "production",
            redis_limiter=limiter,
            additional_check=additional_readiness_check,
        )
        if status != 200:
            metrics.add("readiness_failures_total")
            app.logger.error(
                "readiness_check_failed request_id=%s",
                getattr(g, "request_id", ""),
            )
        return jsonify(payload), status

    def monitoring_allowed() -> bool:
        if (
            getattr(current_user, "is_authenticated", False)
            and getattr(current_user, "role", "") == "superadmin"
        ):
            return True
        configured = str(app.config.get("ATCROSTER_INTERNAL_METRICS_TOKEN", ""))
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        return bool(
            configured and supplied and secrets.compare_digest(configured, supplied)
        )

    def internal_metrics():
        if not monitoring_allowed():
            abort(403)
        return Response(
            metrics.render(),
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )

    def internal_health():
        if not monitoring_allowed():
            abort(403)
        payload, status = readiness_snapshot(
            db=db,
            alembic_path=alembic_path,
            required_tables=frozenset(),
            production=environment == "production",
            redis_limiter=limiter,
            additional_check=additional_readiness_check,
        )
        return (
            jsonify(
                {
                    **payload,
                    "database": "reachable" if status == 200 else "unavailable",
                    "redis_required": environment == "production",
                }
            ),
            status,
        )

    app.add_url_rule("/health/live", "health_live", health_live, methods=["GET"])
    app.add_url_rule("/health/ready", "health_ready", health_ready, methods=["GET"])
    app.add_url_rule(
        "/internal/metrics", "internal_metrics", internal_metrics, methods=["GET"]
    )
    app.add_url_rule(
        "/internal/health", "internal_health", internal_health, methods=["GET"]
    )
