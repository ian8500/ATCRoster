"""Account-recovery token validation and lifecycle helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

from flask import abort


def active_recovery_from_digest(
    RecoveryRequest: Any,
    field_name: str,
    raw_token: str,
    expected_state: str,
    now: Callable[[], Any],
) -> Any:
    """Resolve one unexpired recovery token without exposing token state."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", raw_token or ""):
        abort(404)
    field = getattr(RecoveryRequest, field_name)
    row = RecoveryRequest.query.filter(
        field == hashlib.sha256(raw_token.encode()).hexdigest(),
        RecoveryRequest.state == expected_state,
    ).first_or_404()
    comparison_now = now()
    if row.expires_at.tzinfo is None:
        comparison_now = comparison_now.replace(tzinfo=None)
    if row.expires_at <= comparison_now:
        abort(410, "This recovery link has expired.")
    return row
