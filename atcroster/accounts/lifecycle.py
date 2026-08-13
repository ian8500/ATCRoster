"""Account-login lifecycle persistence."""

from __future__ import annotations

from typing import Any, Callable


def record_successful_login(
    *,
    db: Any,
    PlatformIdentity: Any,
    Unit: Any,
    AggregateUsageEvent: Any,
    user: Any,
    now: Callable[[], Any],
) -> None:
    """Update account and unit activity, then persist the usage event."""
    occurred_at = now()
    identity = PlatformIdentity.query.filter_by(username=user.username).first()
    if identity:
        identity.last_active_at = occurred_at
    unit = db.session.get(Unit, user.unit_id)
    if unit:
        unit.last_active_at = occurred_at
    if user.role != "superadmin":
        db.session.add(AggregateUsageEvent(unit_id=user.unit_id, event_type="login", count=1))
    db.session.commit()
