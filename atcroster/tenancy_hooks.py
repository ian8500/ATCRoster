"""Flask request hooks for verified tenant binding and guaranteed cleanup."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flask import Flask, abort, g, request


@dataclass(frozen=True)
class TenantHookDependencies:
    deployment_environment: str
    current_user: Callable[[], Any]
    enforce_session: Callable[[Any], Any]
    routing_for_unit: Callable[[int], Any | None]
    clear_context: Callable[[], None]
    bind_authenticated_unit: Callable[[int, str | None], Any]
    reset_authenticated_unit: Callable[[Any], None]
    bind_platform_control: Callable[[], Any]
    reset_platform_control: Callable[[Any], None]


def register_tenant_hooks(
    app: Flask,
    dependencies: TenantHookDependencies,
) -> tuple[Callable[[], Any], Callable[[BaseException | None], None]]:
    """Register request binding and teardown with explicit infrastructure edges."""

    def bind_tenant_context():
        dependencies.clear_context()
        g.request_id = request.headers.get("X-Request-ID") or secrets.token_hex(12)
        g.csp_nonce = secrets.token_urlsafe(18)
        g.tenant_context_token = None
        g.platform_control_token = None

        user = dependencies.current_user()
        session_response = dependencies.enforce_session(user)
        if session_response is not None:
            return session_response
        if not user.is_authenticated:
            return None
        if getattr(user, "role", "") == "superadmin":
            g.platform_control_token = dependencies.bind_platform_control()
            return None

        unit_id = int(getattr(user, "unit_id", 0) or 0)
        if unit_id and g.tenant_context_token is None:
            routing = dependencies.routing_for_unit(unit_id)
            if dependencies.deployment_environment == "production" and not routing:
                abort(503, "Operational database routing is unavailable.")
            g.tenant_context_token = dependencies.bind_authenticated_unit(
                unit_id,
                routing.secret_name if routing else None,
            )
        return None

    def reset_tenant_context(_error: BaseException | None = None) -> None:
        token = getattr(g, "tenant_context_token", None)
        g.tenant_context_token = None
        if token is not None:
            try:
                dependencies.reset_authenticated_unit(token)
            except RuntimeError:
                # Flask test/request contexts may invoke teardown more than once.
                pass
        platform_token = getattr(g, "platform_control_token", None)
        g.platform_control_token = None
        if platform_token is not None:
            try:
                dependencies.reset_platform_control(platform_token)
            except RuntimeError:
                pass
        dependencies.clear_context()

    bind_tenant_context.__name__ = "_bind_tenant_context"
    reset_tenant_context.__name__ = "_reset_tenant_context"
    app.before_request(bind_tenant_context)
    app.teardown_request(reset_tenant_context)
    return bind_tenant_context, reset_tenant_context
