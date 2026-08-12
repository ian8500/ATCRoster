"""Response security headers, CSP nonce access and request completion hooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flask import Flask, Response, g, request
from flask_login import current_user


@dataclass(frozen=True)
class SecurityHeaderDependencies:
    deployment_environment: str
    metrics: Any
    finish_request: Callable[..., float]
    slow_roster_seconds: float = 2.0


def csp_nonce() -> str:
    """Return the per-request nonce without creating or persisting one."""
    return getattr(g, "csp_nonce", "")


def content_security_policy(nonce: str, *, production: bool) -> str:
    """Build the established policy as a directly testable pure function."""
    return (
        "default-src 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        f"style-src 'self' 'nonce-{nonce}' "
        "https://fonts.googleapis.com "
        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "style-src-attr 'none'; "
        f"script-src 'self' 'nonce-{nonce}' "
        "https://cdn.jsdelivr.net; "
        "connect-src 'self'; worker-src 'self'; manifest-src 'self'"
        + ("; upgrade-insecure-requests" if production else "")
    )


def register_security_headers(
    app: Flask,
    dependencies: SecurityHeaderDependencies,
) -> Callable[[Response], Response]:
    """Register the explicit Jinja nonce and after-request security boundary."""
    production = dependencies.deployment_environment == "production"
    app.jinja_env.globals["csp_nonce"] = csp_nonce

    def security_headers(response: Response) -> Response:
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            content_security_policy(csp_nonce(), production=production),
        )
        if request.is_secure or production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        # Static URLs emitted by ``asset_url`` have an mtime version query
        # parameter. They are safe to cache even for an authenticated user;
        # dynamic responses remain private below.
        if request.endpoint == "static" and request.args.get("v"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif current_user.is_authenticated:
            response.headers.setdefault("Cache-Control", "no-store, private")
        started_at = getattr(g, "metrics_started_at", None)
        if started_at is not None:
            route = request.endpoint or "unmatched"
            duration = dependencies.finish_request(
                dependencies.metrics,
                started_at,
                route=route,
                method=request.method,
                status=response.status_code,
            )
            if route == "roster_month":
                dependencies.metrics.add("roster_page_requests_total")
                dependencies.metrics.add(
                    "roster_page_duration_seconds_sum", duration
                )
                dependencies.metrics.add("roster_page_duration_seconds_count")
                response.headers["Server-Timing"] = (
                    f"roster;dur={duration * 1000:.2f}"
                )
                if duration >= dependencies.slow_roster_seconds:
                    dependencies.metrics.add("roster_page_slow_requests_total")
                    app.logger.warning(
                        "slow_roster_request",
                        extra={"structured_fields": {
                            "request_id": getattr(g, "request_id", ""),
                            "route": route,
                            "unit_id": getattr(current_user, "unit_id", None),
                            "duration_ms": round(duration * 1000, 2),
                            "threshold_ms": round(
                                dependencies.slow_roster_seconds * 1000, 2
                            ),
                        }},
                    )
            g.metrics_started_at = None
            if production:
                app.logger.info(
                    "request_completed",
                    extra={
                        "structured_fields": {
                            "request_id": getattr(g, "request_id", ""),
                            "route": route,
                            "unit_id": getattr(current_user, "unit_id", None),
                            "actor_id": getattr(current_user, "id", None),
                            "outcome": (
                                "success" if response.status_code < 400 else "error"
                            ),
                            "http_status": response.status_code,
                            "duration_ms": round(duration * 1000, 2),
                        }
                    },
                )
        return response

    security_headers.__name__ = "_security_headers"
    app.after_request(security_headers)
    return security_headers
