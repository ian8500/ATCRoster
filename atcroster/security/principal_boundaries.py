"""Role and MFA boundaries enforced before operational request handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Flask, request, session
from flask_login import current_user

PLATFORM_ENDPOINTS = frozenset(
    {
        "platform_admin",
        "logout",
        "password_change",
        "platform_worker_health",
        "internal_metrics",
        "internal_health",
        "static",
        "favicon",
        "health_live",
        "health_ready",
    }
)
KIOSK_ENDPOINTS = frozenset(
    {
        "live_position.kiosk_hmi",
        "live_position.live_state",
        "live_position.controllers",
        "live_position.open_position",
        "live_position.live_events",
        "live_position.close_position",
        "live_position.logon",
        "live_position.logoff",
        "live_position.handover",
        "live_position.add_participant",
        "live_position.remove_participant",
        "logout",
        "static",
        "favicon",
        "health_live",
        "health_ready",
    }
)
MFA_EXEMPT_ENDPOINTS = frozenset(
    {"mfa_setup", "logout", "static", "favicon", "health_live", "health_ready"}
)


@dataclass(frozen=True)
class PrincipalBoundaryDependencies:
    UnitMembership: Any
    MfaCredential: Any
    deployment_environment: str
    logout_user: Callable[[], None]
    redirect: Callable[[str], Any]
    url_for: Callable[[str], str]
    abort: Callable[[int], None]


def enforce_principal_boundaries(
    user: Any,
    session: Any,
    endpoint: str | None,
    method: str,
    dependencies: PrincipalBoundaryDependencies,
):
    """Enforce role-specific route allowlists and mandatory MFA enrollment."""
    if not getattr(user, "is_authenticated", False):
        return None
    deps = dependencies
    role = getattr(user, "role", "")
    if role == "superadmin":
        if not session.get("_platform_mfa_verified"):
            deps.logout_user()
            session.clear()
            return deps.redirect(deps.url_for("login"))
        if endpoint == "index":
            return deps.redirect(deps.url_for("platform_admin"))
        if endpoint not in PLATFORM_ENDPOINTS:
            deps.abort(403)
        return None
    if role == "position_monitor":
        if endpoint not in KIOSK_ENDPOINTS:
            if method == "GET":
                return deps.redirect(deps.url_for("live_position.kiosk_hmi"))
            deps.abort(403)
        return None
    unit_admin = deps.UnitMembership.query.filter_by(
        person_id=user.id,
        unit_id=getattr(user, "unit_id", 0),
        role="UnitAdmin",
        status="active",
    ).first()
    if (
        deps.deployment_environment == "production" or unit_admin is not None
    ) and endpoint not in MFA_EXEMPT_ENDPOINTS:
        credential = deps.MfaCredential.query.filter_by(
            person_id=user.id, enabled=True
        ).first()
        if not credential:
            return deps.redirect(deps.url_for("mfa_setup"))
    return None


def register_principal_boundaries(
    app: Flask,
    dependencies: PrincipalBoundaryDependencies,
) -> Callable[[], Any]:
    """Register role and MFA enforcement on the application request boundary."""

    def enforce_request_boundaries():
        return enforce_principal_boundaries(
            current_user,
            session,
            request.endpoint,
            request.method,
            dependencies,
        )

    enforce_request_boundaries.__name__ = "_enforce_principal_boundaries"
    app.before_request(enforce_request_boundaries)
    return enforce_request_boundaries
