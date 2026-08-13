"""Privacy-preserving authentication rate-limit helpers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

from flask import abort


def privacy_rate_limit_key(
    secret_key: str,
    scope: str,
    remote_address: str,
    subject: object,
    privacy_key: Callable[..., str],
) -> str:
    return privacy_key(secret_key, scope, remote_address or "unknown", subject)


def consume_rate_limit(
    *,
    limiter: Any,
    key: str,
    limit: int,
    window: timedelta,
    unavailable: type[Exception],
    security_event: Callable[..., None],
    scope: str,
    fail_closed: bool = True,
) -> bool:
    try:
        return limiter.consume(key, limit, max(1, int(window.total_seconds())))
    except unavailable:
        security_event("rate_limiter_unavailable", scope=scope)
        if fail_closed:
            abort(503, "Security service is temporarily unavailable.")
        return True


def reset_rate_limit(
    *,
    limiter: Any,
    key: str,
    unavailable: type[Exception],
    security_event: Callable[..., None],
    scope: str,
) -> None:
    try:
        limiter.reset(key)
    except unavailable:
        security_event("rate_limiter_unavailable", scope=scope)
