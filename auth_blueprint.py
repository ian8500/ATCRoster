"""Authentication routes extracted from the legacy application module.

The blueprint registers the historical global endpoint names so existing
templates, redirects, LoginManager configuration and external URLs remain
compatible while authentication can evolve outside ``app.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash


@dataclass(frozen=True)
class AuthDependencies:
    """The deliberately small application surface required by login routes."""

    db: Any
    PlatformIdentity: Any
    UnitMembership: Any
    DatabaseRoutingMetadata: Any
    Staff: Any
    Unit: Any
    PlatformMfaCredential: Any
    MfaCredential: Any
    validate_csrf: Callable[[], None]
    normalized_login: Callable[[str], str]
    login_rate_key: Callable[[str], str]
    consume_rate_limit: Callable[[str, str], bool]
    reset_rate_limit: Callable[[str, str], None]
    security_event: Callable[..., None]
    central_security_event: Callable[..., None]
    bind_authenticated_unit: Callable[[int, str | None], Any]
    canonical_login_redirect: Callable[..., str]
    airport_login_endpoint: Callable[[Any], str]
    initialize_authenticated_session: Callable[[Any], None]
    record_successful_login: Callable[[Any], None]


@dataclass(frozen=True)
class OperationalPrincipal:
    user: Any
    identity: Any | None


@dataclass(frozen=True)
class PlatformPrincipal:
    user: Any
    identity: Any


LoginPrincipal = OperationalPrincipal | PlatformPrincipal


def create_auth_blueprint(dependencies: AuthDependencies) -> Blueprint:
    blueprint = Blueprint("authentication", __name__)

    @login_required
    def logout():
        session.pop("reports_sensitive_data_ack", None)
        session.pop("reports_sensitive_data_hub_entry", None)
        logout_user()
        flash("Logged out", "ok")
        return redirect(url_for("login"))

    def signin_form():
        if request.method == "POST":
            dependencies.validate_csrf()
            username = dependencies.normalized_login(request.form.get("username") or "")
            # Passwords are opaque secrets. Trimming changes valid credentials
            # and can unexpectedly authenticate a different submitted value.
            password = request.form.get("password") or ""
            rate_key = dependencies.login_rate_key(username)
            if not dependencies.consume_rate_limit("password-login", username):
                dependencies.security_event(
                    "login_rate_limited", principal=rate_key[-16:]
                )
                abort(429, "Too many login attempts. Try again later.")
            identity = dependencies.PlatformIdentity.query.filter_by(
                username=username
            ).first()
            principal: LoginPrincipal | None = None
            credentials_valid = False
            if identity:
                credentials_valid = check_password_hash(
                    identity.password_hash, password
                )
                if credentials_valid:
                    membership = dependencies.UnitMembership.query.filter_by(
                        identity_id=identity.id, status="active"
                    ).first()
                    if membership and membership.person_id:
                        routing = dependencies.db.session.get(
                            dependencies.DatabaseRoutingMetadata,
                            membership.unit_id,
                        )
                        if (
                            current_app.config["ATCROSTER_ENVIRONMENT"] == "production"
                            and not routing
                        ):
                            dependencies.security_event(
                                "operational_route_missing",
                                unit_id=membership.unit_id,
                            )
                            abort(
                                503,
                                "Operational database routing is unavailable.",
                            )
                        g.tenant_context_token = dependencies.bind_authenticated_unit(
                            membership.unit_id,
                            routing.secret_name if routing else None,
                        )
                        user = dependencies.db.session.get(
                            dependencies.Staff, membership.person_id
                        )
                        if user:
                            principal = OperationalPrincipal(user, identity)
                    else:
                        if identity.public_id.startswith("platform-"):
                            principal = PlatformPrincipal(identity, identity)
            else:
                # Authentication always begins in control. Unknown
                # principals never trigger an operational database query.
                credentials_valid = False
            if (
                identity
                and credentials_valid
                and not principal
                and not identity.public_id.startswith("platform-")
            ):
                credentials_valid = False
            if principal and credentials_valid:
                user = principal.user
                dependencies.reset_rate_limit("password-login", username)
                if user.membership_status != "active":
                    flash("This account is not active.", "error")
                    return render_template("login.html"), 403
                login_unit = dependencies.db.session.get(
                    dependencies.Unit, user.unit_id
                )
                if user.role != "superadmin" and (
                    not login_unit or login_unit.status != "active"
                ):
                    dependencies.security_event(
                        "suspended_unit_login_blocked",
                        principal=rate_key[-16:],
                        unit_id=user.unit_id,
                    )
                    flash("This airport account is not active.", "error")
                    return render_template("login.html"), 403
                session.clear()
                if isinstance(principal, PlatformPrincipal):
                    credential = dependencies.PlatformMfaCredential.query.filter_by(
                        identity_id=identity.id,
                        enabled=True,
                        reset_required=False,
                    ).first()
                    session["_platform_mfa_identity_id"] = identity.id
                    session["_platform_mfa_user_id"] = user.id
                    session["_platform_mfa_rate_key"] = rate_key
                    session["_platform_mfa_next"] = (
                        dependencies.canonical_login_redirect(
                            request.args.get("next"),
                            default_endpoint="platform_admin",
                            user_id=user.id,
                        )
                    )
                    dependencies.central_security_event(
                        "platform_password_verified",
                        "challenge",
                        identity.id,
                        rate_key[-16:],
                    )
                    dependencies.db.session.commit()
                    return redirect(
                        url_for(
                            "platform_mfa_challenge"
                            if credential
                            else "platform_mfa_setup"
                        )
                    )
                if user.role == "position_monitor":
                    # Password-only authentication is deliberately confined to
                    # the locked-down unit kiosk role. Ordinary airport users
                    # continue through the existing MFA flow below.
                    login_user(user, remember=True)
                    session.permanent = True
                    dependencies.initialize_authenticated_session(user)
                    dependencies.security_event(
                        "position_monitor_login_succeeded",
                        principal=rate_key[-16:],
                        unit_id=user.unit_id,
                    )
                    dependencies.record_successful_login(user)
                    return redirect(url_for(dependencies.airport_login_endpoint(user)))
                credential = dependencies.MfaCredential.query.filter_by(
                    person_id=user.id, enabled=True
                ).first()
                if credential:
                    session["_mfa_user_id"] = user.id
                    session["_mfa_unit_id"] = user.unit_id
                    session["_mfa_rate_key"] = rate_key
                    session["_mfa_next"] = dependencies.canonical_login_redirect(
                        request.args.get("next"),
                        default_endpoint=dependencies.airport_login_endpoint(user),
                        user_id=user.id,
                    )
                    return redirect(url_for("mfa_challenge"))
                login_user(user)
                dependencies.initialize_authenticated_session(user)
                dependencies.security_event(
                    "login_succeeded",
                    principal=rate_key[-16:],
                    unit_id=user.unit_id,
                )
                dependencies.record_successful_login(user)
                flash("Logged in successfully", "ok")
                return redirect(
                    dependencies.canonical_login_redirect(
                        request.args.get("next"),
                        default_endpoint=dependencies.airport_login_endpoint(user),
                        user_id=user.id,
                    )
                )
            if identity:
                dependencies.central_security_event(
                    "platform_login_failed",
                    "denied",
                    identity.id,
                    rate_key[-16:],
                )
                dependencies.db.session.commit()
            dependencies.security_event("login_failed", principal=rate_key[-16:])
            flash("Invalid username or password.", "error")
        return render_template("login.html")

    @blueprint.record_once
    def register_routes(state):
        # Register global endpoints deliberately: changing ``login`` or
        # ``logout`` would break LoginManager and existing template contracts.
        state.app.add_url_rule(
            "/login",
            endpoint="login",
            view_func=signin_form,
            methods=["GET", "POST"],
        )
        state.app.add_url_rule(
            "/logout",
            endpoint="logout",
            view_func=logout,
            methods=["POST"],
        )

    return blueprint
