"""Central security-audit event creation."""

from __future__ import annotations

from typing import Any


def record_central_security_event(
    db: Any,
    CentralSecurityAudit: Any,
    event_type: str,
    outcome: str,
    identity_id: int | None = None,
    principal: str = "",
    detail: str = "",
) -> None:
    """Append a bounded, non-sensitive central security event."""
    db.session.add(CentralSecurityAudit(
        identity_id=identity_id,
        event_type=event_type[:80],
        outcome=outcome[:20],
        principal_digest=principal[:32],
        safe_detail=detail[:200],
    ))
