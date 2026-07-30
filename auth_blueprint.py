"""Authentication routes extracted from the legacy application module.

The blueprint registers the historical global endpoint names so existing
templates, redirects, LoginManager configuration and external URLs remain
compatible while authentication can evolve outside ``app.py``.
"""

from __future__ import annotations

from types import ModuleType

from flask import Blueprint, abort, flash, g, redirect, render_template, request
from flask import session, url_for
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash


def create_auth_blueprint(core: ModuleType) -> Blueprint:
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
            core._validate_csrf()
            username = core._normalized_login(request.form.get("username") or "")
            password = (request.form.get("password") or "").strip()
            rate_key = core._login_rate_key(username)
            if not core._consume_rate_limit("password-login", username):
                core._security_event("login_rate_limited", principal=rate_key[-16:])
                abort(429, "Too many login attempts. Try again later.")
            identity = core.PlatformIdentity.query.filter_by(username=username).first()
            user = None
            platform_login = False
            credentials_valid = False
            if identity:
                credentials_valid = check_password_hash(
                    identity.password_hash, password
                )
                if credentials_valid:
                    membership = core.UnitMembership.query.filter_by(
                        identity_id=identity.id, status="active"
                    ).first()
                    if membership and membership.person_id:
                        routing = core.db.session.get(
                            core.DatabaseRoutingMetadata, membership.unit_id
                        )
                        if core.DEPLOYMENT_ENV == "production" and not routing:
                            core._security_event(
                                "operational_route_missing",
                                unit_id=membership.unit_id,
                            )
                            abort(
                                503,
                                "Operational database routing is unavailable.",
                            )
                        g.tenant_context_token = core.bind_authenticated_unit(
                            membership.unit_id,
                            routing.secret_name if routing else None,
                        )
                        user = core.db.session.get(core.Staff, membership.person_id)
                    else:
                        user = identity
                        platform_login = identity.public_id.startswith("platform-")
            elif core.DEPLOYMENT_ENV != "production":
                user = core.Staff.query.filter_by(username=username).first()
                credentials_valid = bool(user and user.check_password(password))
            else:
                # Production authentication always begins in control. Unknown
                # principals never trigger an operational database query.
                credentials_valid = False
            if (
                identity
                and credentials_valid
                and not user
                and not identity.public_id.startswith("platform-")
            ):
                credentials_valid = False
            if user and credentials_valid:
                core._reset_rate_limit("password-login", username)
                if user.membership_status != "active":
                    flash("This account is not active.", "error")
                    return render_template("login.html"), 403
                login_unit = core.db.session.get(core.Unit, user.unit_id)
                if user.role != "superadmin" and (
                    not login_unit or login_unit.status != "active"
                ):
                    core._security_event(
                        "suspended_unit_login_blocked",
                        principal=rate_key[-16:],
                        unit_id=user.unit_id,
                    )
                    flash("This airport account is not active.", "error")
                    return render_template("login.html"), 403
                session.clear()
                if platform_login:
                    credential = core.PlatformMfaCredential.query.filter_by(
                        identity_id=identity.id,
                        enabled=True,
                        reset_required=False,
                    ).first()
                    session["_platform_mfa_identity_id"] = identity.id
                    session["_platform_mfa_user_id"] = user.id
                    session["_platform_mfa_rate_key"] = rate_key
                    session["_platform_mfa_next"] = core._canonical_login_redirect(
                        request.args.get("next"),
                        default_endpoint="platform_admin",
                        user_id=user.id,
                    )
                    core._central_security_event(
                        "platform_password_verified",
                        "challenge",
                        identity.id,
                        rate_key[-16:],
                    )
                    core.db.session.commit()
                    return redirect(
                        url_for(
                            "platform_mfa_challenge"
                            if credential
                            else "platform_mfa_setup"
                        )
                    )
                credential = core.MfaCredential.query.filter_by(
                    person_id=user.id, enabled=True
                ).first()
                if credential:
                    session["_mfa_user_id"] = user.id
                    session["_mfa_unit_id"] = user.unit_id
                    session["_mfa_rate_key"] = rate_key
                    session["_mfa_next"] = core._canonical_login_redirect(
                        request.args.get("next"),
                        default_endpoint=core._airport_login_endpoint(user),
                        user_id=user.id,
                    )
                    return redirect(url_for("mfa_challenge"))
                login_user(user)
                core._initialize_authenticated_session(user)
                core._security_event(
                    "login_succeeded",
                    principal=rate_key[-16:],
                    unit_id=user.unit_id,
                )
                core._record_successful_login(user)
                flash("Logged in successfully", "ok")
                return redirect(
                    core._canonical_login_redirect(
                        request.args.get("next"),
                        default_endpoint=core._airport_login_endpoint(user),
                        user_id=user.id,
                    )
                )
            if identity:
                core._central_security_event(
                    "platform_login_failed",
                    "denied",
                    identity.id,
                    rate_key[-16:],
                )
                core.db.session.commit()
            core._security_event("login_failed", principal=rate_key[-16:])
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
