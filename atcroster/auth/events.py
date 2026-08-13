"""Authentication security-event telemetry."""

from __future__ import annotations

from typing import Any, Callable


def record_security_event(
    *,
    metrics: Any,
    logger: Any,
    request_id: str,
    structured_event: Callable[..., None],
    event: str,
    **safe_fields: Any,
) -> None:
    """Emit privacy-safe auth telemetry and structured event data."""
    if "login_failed" in event or event == "mfa_login_failed":
        metrics.add("login_failures_total", event=event)
    if "rate_limit" in event:
        metrics.add("rate_limit_events_total", event=event)
    if event == "rate_limiter_unavailable":
        metrics.add("redis_failures_total", operation="rate_limit")
    structured_event(logger, event, request_id=request_id, **safe_fields)
