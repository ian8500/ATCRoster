"""MFA enrolment and challenge routes for platform and airport identities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import pyotp
from flask import (
    Blueprint,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_user


@dataclass(frozen=True)
class MfaRouteDependencies:
    db: Any
    PlatformIdentity: Any
    PlatformMfaCredential: Any
    Staff: Any
    MfaCredential: Any
    DatabaseRoutingMetadata: Any
    deployment_environment: str
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    decrypt_secret: Callable[[Any], str]
    matching_totp_step: Callable[[str, str], int | None]
    encrypt_field: Callable[[str], str]
    now: Callable[[], Any]
    central_security_event: Callable[..., None]
    bind_authenticated_unit: Callable[[int, str | None], Any]
    initialize_authenticated_session: Callable[..., None]
    security_event: Callable[..., None]
    record_successful_login: Callable[[Any], None]
    canonical_login_redirect: Callable[..., str]
    current_unit_id: Callable[[], int]
    current_auth_stamp: Callable[[Any], str]
    totp_qr_data_uri: Callable[[str], str]


def create_mfa_blueprint(dependencies: MfaRouteDependencies) -> Blueprint:
    """Build the compatibility routes while keeping MFA behaviour domain-owned."""
    blueprint = Blueprint("mfa", __name__)

    def pending_platform_login() -> tuple[Any | None, Any | None]:
        identity_id = int(session.get("_platform_mfa_identity_id") or 0)
        user_id = int(session.get("_platform_mfa_user_id") or 0)
        if not identity_id or user_id != identity_id:
            return None, None
        identity = dependencies.db.session.get(
            dependencies.PlatformIdentity, identity_id
        )
        if not identity or identity.role != "superadmin":
            return None, None
        return identity, identity

    def complete_platform_login(identity: Any, user: Any):
        next_url = session.get("_platform_mfa_next", "")
        session.clear()
        login_user(user)
        dependencies.initialize_authenticated_session(user, platform_mfa=True)
        identity.last_active_at = dependencies.now()
        dependencies.central_security_event(
            "platform_mfa_verified",
            "success",
            identity.id,
            hashlib.sha256(identity.username.lower().encode()).hexdigest()[:16],
        )
        dependencies.db.session.commit()
        return redirect(
            dependencies.canonical_login_redirect(
                next_url,
                default_endpoint="platform_admin",
                user_id=user.id,
            )
        )

    def platform_mfa_setup():
        identity, user = pending_platform_login()
        if not identity or not user:
            session.clear()
            return redirect(url_for("login"))
        existing = dependencies.PlatformMfaCredential.query.filter_by(
            identity_id=identity.id,
            enabled=True,
            reset_required=False,
        ).first()
        if existing:
            return redirect(url_for("platform_mfa_challenge"))
        pending = session.get("_pending_platform_mfa_secret")
        if not pending:
            pending = pyotp.random_base32()
            session["_pending_platform_mfa_secret"] = pending
        provisioning_uri = pyotp.TOTP(pending).provisioning_uri(
            name=identity.username,
            issuer_name="ATCRoster Platform",
        )
        if request.method == "POST":
            dependencies.validate_csrf()
            if not dependencies.consume_rate_limit(
                "platform-mfa-enrolment",
                identity.id,
                limit=10,
                window=timedelta(minutes=15),
            ):
                abort(429)
            code = re.sub(r"\s", "", request.form.get("code") or "")
            if not pyotp.TOTP(pending).verify(code, valid_window=1):
                dependencies.central_security_event(
                    "platform_mfa_enrolment", "denied", identity.id
                )
                dependencies.db.session.commit()
                flash("The verification code is not valid.", "error")
                return redirect(url_for("platform_mfa_setup"))
            credential = dependencies.PlatformMfaCredential.query.filter_by(
                identity_id=identity.id
            ).first()
            if not credential:
                credential = dependencies.PlatformMfaCredential(
                    identity_id=identity.id, encrypted_secret=""
                )
                dependencies.db.session.add(credential)
            credential.encrypted_secret = dependencies.encrypt_field(pending)
            credential.enabled, credential.reset_required = True, False
            credential.enrolled_at = dependencies.now()
            credential.recovery_codes_digest = "[]"
            dependencies.central_security_event(
                "platform_mfa_enrolment", "success", identity.id
            )
            dependencies.db.session.commit()
            session.pop("_pending_platform_mfa_secret", None)
            return redirect(url_for("platform_mfa_challenge"))
        return render_template(
            "mfa_setup.html",
            enabled=False,
            secret=pending,
            provisioning_uri=provisioning_uri,
            qr_data_uri=dependencies.totp_qr_data_uri(provisioning_uri),
            platform_enrolment=True,
        )

    def platform_mfa_challenge():
        identity, user = pending_platform_login()
        if not identity or not user:
            session.clear()
            return redirect(url_for("login"))
        credential = dependencies.PlatformMfaCredential.query.filter_by(
            identity_id=identity.id, enabled=True, reset_required=False
        ).first()
        if not credential:
            return redirect(url_for("platform_mfa_setup"))
        if request.method == "POST":
            dependencies.validate_csrf()
            if not dependencies.consume_rate_limit("platform-mfa", identity.id):
                dependencies.central_security_event(
                    "platform_mfa_rate_limited", "denied", identity.id
                )
                dependencies.db.session.commit()
                abort(429, "Too many verification attempts. Try again later.")
            code = re.sub(r"[\s-]", "", request.form.get("code") or "").upper()
            accepted = False
            if re.fullmatch(r"\d{6}", code):
                step = dependencies.matching_totp_step(
                    dependencies.decrypt_secret(credential), code
                )
                if step is not None and (
                    credential.last_used_step is None
                    or step > credential.last_used_step
                ):
                    credential.last_used_step, accepted = step, True
            if accepted:
                return complete_platform_login(identity, user)
            dependencies.central_security_event(
                "platform_mfa_verification", "denied", identity.id
            )
            dependencies.db.session.commit()
            flash("Invalid, expired or already-used verification code.", "error")
        return render_template("mfa_challenge.html", platform_challenge=True)

    def mfa_challenge():
        user_id, unit_id = (
            int(session.get("_mfa_user_id") or 0),
            int(session.get("_mfa_unit_id") or 0),
        )
        if not user_id or not unit_id:
            return redirect(url_for("login"))
        routing = dependencies.db.session.get(
            dependencies.DatabaseRoutingMetadata, unit_id
        )
        if dependencies.deployment_environment == "production" and not routing:
            session.clear()
            abort(503, "Operational database routing is unavailable.")
        g.tenant_context_token = dependencies.bind_authenticated_unit(
            unit_id, routing.secret_name if routing else None
        )
        user = dependencies.Staff.query.filter_by(id=user_id, unit_id=unit_id).first()
        credential = dependencies.MfaCredential.query.filter_by(
            person_id=user_id, enabled=True
        ).first()
        if not user or not credential:
            session.clear()
            return redirect(url_for("login"))
        if request.method == "POST":
            dependencies.validate_csrf()
            if not dependencies.consume_rate_limit(
                "airport-mfa", f"{unit_id}:{user_id}"
            ):
                abort(429, "Too many verification attempts. Try again later.")
            code = re.sub(r"[\s-]", "", request.form.get("code") or "").upper()
            accepted = False
            if re.fullmatch(r"\d{6}", code):
                step = dependencies.matching_totp_step(
                    dependencies.decrypt_secret(credential), code
                )
                if step is not None and (
                    credential.last_used_step is None
                    or step > credential.last_used_step
                ):
                    credential.last_used_step, accepted = step, True
            if accepted:
                next_url = session.get("_mfa_next", "")
                session.clear()
                login_user(user)
                dependencies.initialize_authenticated_session(user)
                dependencies.security_event(
                    "mfa_login_succeeded",
                    principal=hashlib.sha256(
                        user.username.lower().encode()
                    ).hexdigest()[:16],
                    unit_id=user.unit_id,
                )
                dependencies.record_successful_login(user)
                dependencies.db.session.commit()
                return redirect(
                    dependencies.canonical_login_redirect(next_url, user_id=user.id)
                )
            flash("Invalid, expired or already-used verification code.", "error")
        return render_template("mfa_challenge.html")

    def mfa_setup():
        pending_user_id = int(session.get("_mfa_user_id") or 0)
        pending_unit_id = int(session.get("_mfa_unit_id") or 0)
        pending_enrolment = bool(pending_user_id and pending_unit_id)
        if pending_enrolment:
            routing = dependencies.db.session.get(
                dependencies.DatabaseRoutingMetadata, pending_unit_id
            )
            g.tenant_context_token = dependencies.bind_authenticated_unit(
                pending_unit_id, routing.secret_name if routing else None
            )
            enrolment_user = dependencies.Staff.query.filter_by(
                id=pending_user_id, unit_id=pending_unit_id, membership_status="active"
            ).first()
            if not enrolment_user:
                session.clear()
                return redirect(url_for("login"))
        elif current_user.is_authenticated:
            enrolment_user = current_user
        else:
            return redirect(url_for("login"))
        if getattr(enrolment_user, "role", "") == "superadmin":
            abort(403)
        credential = dependencies.MfaCredential.query.filter_by(
            person_id=enrolment_user.id
        ).first()
        if credential and credential.enabled and not credential.reset_required:
            return render_template("mfa_setup.html", enabled=True)
        pending = session.get("_pending_mfa_secret")
        if not pending:
            pending = pyotp.random_base32()
            session["_pending_mfa_secret"] = pending
        provisioning_uri = pyotp.TOTP(pending).provisioning_uri(
            name=enrolment_user.username, issuer_name="ATCRoster"
        )
        if request.method == "POST":
            dependencies.validate_csrf()
            if not dependencies.consume_rate_limit(
                "airport-mfa-enrolment",
                f"{enrolment_user.unit_id}:{enrolment_user.id}",
                limit=10,
                window=timedelta(minutes=15),
            ):
                abort(429)
            code = re.sub(r"\s", "", request.form.get("code") or "")
            if not pyotp.TOTP(pending).verify(code, valid_window=1):
                flash("The verification code is not valid.", "error")
                return redirect(url_for("mfa_setup"))
            if not credential:
                credential = dependencies.MfaCredential(
                    unit_id=enrolment_user.unit_id,
                    person_id=enrolment_user.id,
                    encrypted_secret="",
                )
                dependencies.db.session.add(credential)
            credential.encrypted_secret, credential.enabled, credential.reset_required = (
                dependencies.encrypt_field(pending),
                True,
                False,
            )
            credential.enrolled_at = dependencies.now()
            credential.recovery_codes_digest = "[]"
            dependencies.db.session.commit()
            session.pop("_pending_mfa_secret", None)
            if pending_enrolment:
                next_url = session.get("_mfa_next", "")
                session.clear()
                login_user(enrolment_user)
                dependencies.initialize_authenticated_session(enrolment_user)
                dependencies.security_event(
                    "mfa_reenrolment_succeeded",
                    principal=hashlib.sha256(enrolment_user.username.lower().encode()).hexdigest()[:16],
                    unit_id=enrolment_user.unit_id,
                )
                dependencies.record_successful_login(enrolment_user)
                return redirect(dependencies.canonical_login_redirect(next_url, user_id=enrolment_user.id))
            session["_auth_stamp"] = dependencies.current_auth_stamp(enrolment_user)
            return redirect(url_for("staff_profile", sid=enrolment_user.id) + "#mfa")
        return render_template(
            "mfa_setup.html",
            enabled=False,
            secret=pending,
            provisioning_uri=provisioning_uri,
            qr_data_uri=dependencies.totp_qr_data_uri(provisioning_uri),
            reset_enrolment=pending_enrolment and bool(credential and credential.reset_required),
        )

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule(
            "/login/platform-mfa/setup",
            "platform_mfa_setup",
            platform_mfa_setup,
            methods=("GET", "POST"),
        )
        state.app.add_url_rule(
            "/login/platform-mfa",
            "platform_mfa_challenge",
            platform_mfa_challenge,
            methods=("GET", "POST"),
        )
        state.app.add_url_rule(
            "/login/mfa", "mfa_challenge", mfa_challenge, methods=("GET", "POST")
        )
        state.app.add_url_rule(
            "/security/mfa", "mfa_setup", mfa_setup, methods=("GET", "POST")
        )

    return blueprint
