"""Flask-Login identity decoding with explicit tenant binding."""

from __future__ import annotations

from typing import Any, Callable


def load_identity(
    user_id: object,
    *,
    db: Any,
    UnitMembership: Any,
    DatabaseRoutingMetadata: Any,
    PlatformIdentity: Any,
    Staff: Any,
    deployment_environment: str,
    bind_authenticated_unit: Callable[..., Any],
    remember_tenant_token: Callable[[Any], None],
):
    """Resolve membership, platform, or development-only legacy identities."""
    value = str(user_id or "")
    if value.startswith("membership:"):
        membership_id = _suffix_int(value)
        if membership_id is None:
            return None
        membership = db.session.get(UnitMembership, membership_id)
        if not membership or membership.status != "active":
            return None
        routing = db.session.get(DatabaseRoutingMetadata, membership.unit_id)
        if deployment_environment == "production" and not routing:
            return None
        token = bind_authenticated_unit(
            membership.unit_id, routing.secret_name if routing else None
        )
        remember_tenant_token(token)
        return db.session.get(Staff, membership.person_id)
    if value.startswith("platform-identity:"):
        identity_id = _suffix_int(value)
        return (
            db.session.get(PlatformIdentity, identity_id)
            if identity_id is not None
            else None
        )
    if value.startswith("legacy:") and deployment_environment != "production":
        try:
            _, raw_unit_id, raw_person_id = value.split(":", 2)
            token = bind_authenticated_unit(int(raw_unit_id))
            remember_tenant_token(token)
            return db.session.get(Staff, int(raw_person_id))
        except ValueError:
            return None
    return None


def _suffix_int(value: str) -> int | None:
    try:
        return int(value.split(":", 1)[1])
    except (IndexError, ValueError):
        return None
