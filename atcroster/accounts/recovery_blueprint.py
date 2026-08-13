"""Account-recovery request route."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import (
    Blueprint,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required


@dataclass(frozen=True)
class RecoveryRequestDependencies:
    db: Any
    PlatformIdentity: Any
    UnitMembership: Any
    RecoveryRequest: Any
    Unit: Any
    Staff: Any
    DatabaseRoutingMetadata: Any
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    valid_email: Callable[[str], str]
    normalized_login: Callable[[str], str]
    platform_support_emails: Callable[[], list[str]]
    unit_admin_emails: Callable[[int], list[str]]
    send_email: Callable[[str, str, str], bool]
    now: Callable[[], Any]
    active_recovery: Callable[[str, str, str], Any]
    is_admin_user: Callable[[Any], bool]
    bind_authenticated_unit: Callable[[int, str | None], Any]
    generate_password_hash: Callable[[str], str]


def create_recovery_request_blueprint(
    dependencies: RecoveryRequestDependencies,
) -> Blueprint:
    blueprint = Blueprint("account_recovery", __name__)

    def account_recovery():
        mode = (request.values.get("mode") or "password").strip()
        if mode not in {"password", "username"}:
            mode = "password"
        if request.method == "POST":
            dependencies.validate_csrf()
            subject = hashlib.sha256(
                f"{request.remote_addr}:{mode}".encode()
            ).hexdigest()
            if not dependencies.consume_rate_limit(
                "account-recovery", subject, limit=8, window=timedelta(hours=1)
            ):
                abort(429, "Too many recovery attempts. Try again later.")
            if mode == "username":
                email = dependencies.valid_email(request.form.get("email") or "")
                airport_code = (request.form.get("airport_code") or "").strip().upper()
                if email:
                    identities = (
                        dependencies.PlatformIdentity.query.filter(
                            dependencies.db.func.lower(
                                dependencies.PlatformIdentity.email
                            )
                            == email,
                        )
                        .order_by(dependencies.PlatformIdentity.username)
                        .all()
                    )
                    if identities:
                        usernames = "\n".join(
                            f"- {identity.username}" for identity in identities
                        )
                        dependencies.send_email(
                            email,
                            "Your ATCRoster username",
                            "The username(s) registered to this email address are:\n"
                            + usernames
                            + "\n\nIf you did not request this, contact your Unit Administrator.",
                        )
                    elif airport_code and (
                        unit := dependencies.Unit.query.filter_by(
                            code=airport_code
                        ).first()
                    ):
                        for admin_email in dependencies.unit_admin_emails(unit.id):
                            dependencies.send_email(
                                admin_email,
                                f"Username recovery assistance for {unit.code}",
                                f"A user requested username assistance for {email}. Verify their identity using your approved local process before disclosing or updating account details.",
                            )
            else:
                username = dependencies.normalized_login(
                    request.form.get("username") or ""
                )
                identity = dependencies.PlatformIdentity.query.filter_by(
                    username=username
                ).first()
                if identity:
                    membership = dependencies.UnitMembership.query.filter_by(
                        identity_id=identity.id, status="active"
                    ).first()
                    unit_id = membership.unit_id if membership else None
                    approvers = (
                        dependencies.platform_support_emails()
                        if identity.role == "superadmin"
                        or (membership and membership.role == "UnitAdmin")
                        else dependencies.unit_admin_emails(unit_id)
                    )
                    approvers = approvers or dependencies.platform_support_emails()
                    token = secrets.token_urlsafe(32)
                    dependencies.db.session.add(
                        dependencies.RecoveryRequest(
                            unit_id=unit_id,
                            identity_id=identity.id,
                            person_id=membership.person_id if membership else None,
                            approval_token_digest=hashlib.sha256(
                                token.encode()
                            ).hexdigest(),
                            state="pending_approval",
                            expires_at=dependencies.now() + timedelta(hours=24),
                        )
                    )
                    dependencies.db.session.commit()
                    approval_url = url_for(
                        "approve_account_recovery", token=token, _external=True
                    )
                    for approver in approvers:
                        dependencies.send_email(
                            approver,
                            "ATCRoster password reset approval required",
                            f"A password reset was requested for {username}.\n\nReview and approve it here:\n{approval_url}\n\nThe link expires in 24 hours. Do not forward it.",
                        )
            flash(
                "If the supplied details match an account, the recovery process has started. Check email or contact your administrator.",
                "ok",
            )
            return redirect(url_for("account_recovery", mode=mode))
        return render_template("account_recovery.html", mode=mode)

    @login_required
    def approve_account_recovery(token: str):
        row = dependencies.active_recovery(
            "approval_token_digest", token, "pending_approval"
        )
        permitted = getattr(current_user, "role", "") == "superadmin" or (
            dependencies.is_admin_user(current_user)
            and int(getattr(current_user, "unit_id", 0) or 0) == int(row.unit_id or 0)
        )
        if not permitted:
            abort(403)
        identity = dependencies.db.session.get(
            dependencies.PlatformIdentity, row.identity_id
        )
        if not identity:
            abort(410, "The account is no longer available.")
        if request.method == "POST":
            dependencies.validate_csrf()
            if not identity.email:
                flash(
                    "This account has no registered email. Add and verify an email address before approving the reset.",
                    "error",
                )
                return redirect(request.url)
            raw_reset = secrets.token_urlsafe(32)
            row.reset_token_digest = hashlib.sha256(raw_reset.encode()).hexdigest()
            row.state, row.approved_at, row.expires_at = (
                "reset_sent",
                dependencies.now(),
                dependencies.now() + timedelta(hours=1),
            )
            reset_url = url_for(
                "complete_account_recovery", token=raw_reset, _external=True
            )
            if not dependencies.send_email(
                identity.email,
                "Reset your ATCRoster password",
                f"Your password reset was approved.\n\nChoose a new password here:\n{reset_url}\n\nThe link expires in one hour and can only be used once.",
            ):
                dependencies.db.session.rollback()
                flash(
                    "The reset email could not be delivered. Check the SMTP configuration and the registered email address.",
                    "error",
                )
                return redirect(request.url)
            dependencies.db.session.commit()
            flash("Reset approved and emailed to the account holder.", "ok")
            return redirect(url_for("index"))
        return render_template("recovery_approve.html", recovery=row, identity=identity)

    def complete_account_recovery(token: str):
        row = dependencies.active_recovery("reset_token_digest", token, "reset_sent")
        identity = dependencies.db.session.get(
            dependencies.PlatformIdentity, row.identity_id
        )
        if not identity:
            abort(410, "The account is no longer available.")
        if request.method == "POST":
            dependencies.validate_csrf()
            password, confirmation = (
                request.form.get("password") or "",
                request.form.get("password_confirmation") or "",
            )
            if len(password) < 12:
                flash("Use a password of at least 12 characters.", "error")
            elif password != confirmation:
                flash("The password confirmation does not match.", "error")
            else:
                password_hash = dependencies.generate_password_hash(password)
                identity.password_hash = password_hash
                membership = dependencies.UnitMembership.query.filter_by(
                    identity_id=identity.id, status="active"
                ).first()
                if membership and membership.person_id:
                    routing = dependencies.db.session.get(
                        dependencies.DatabaseRoutingMetadata, membership.unit_id
                    )
                    g.tenant_context_token = dependencies.bind_authenticated_unit(
                        membership.unit_id, routing.secret_name if routing else None
                    )
                    if person := dependencies.db.session.get(
                        dependencies.Staff, membership.person_id
                    ):
                        person.password_hash = password_hash
                row.state, row.completed_at, row.reset_token_digest = (
                    "completed",
                    dependencies.now(),
                    None,
                )
                dependencies.db.session.commit()
                flash("Password updated. Sign in with your new password.", "ok")
                return redirect(url_for("login"))
        return render_template("recovery_reset.html")

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/recover", "account_recovery", account_recovery, methods=("GET", "POST")
        )
        state.app.add_url_rule(
            "/recover/approve/<token>",
            "approve_account_recovery",
            approve_account_recovery,
            methods=("GET", "POST"),
        )
        state.app.add_url_rule(
            "/recover/reset/<token>",
            "complete_account_recovery",
            complete_account_recovery,
            methods=("GET", "POST"),
        )

    return blueprint
