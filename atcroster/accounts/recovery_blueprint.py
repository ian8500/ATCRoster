"""Account-recovery request route."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for


@dataclass(frozen=True)
class RecoveryRequestDependencies:
    db: Any
    PlatformIdentity: Any
    UnitMembership: Any
    RecoveryRequest: Any
    Unit: Any
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    valid_email: Callable[[str], str]
    normalized_login: Callable[[str], str]
    platform_support_emails: Callable[[], list[str]]
    unit_admin_emails: Callable[[int], list[str]]
    send_email: Callable[[str, str, str], bool]
    now: Callable[[], Any]


def create_recovery_request_blueprint(dependencies: RecoveryRequestDependencies) -> Blueprint:
    blueprint = Blueprint("account_recovery", __name__)

    def account_recovery():
        mode = (request.values.get("mode") or "password").strip()
        if mode not in {"password", "username"}:
            mode = "password"
        if request.method == "POST":
            dependencies.validate_csrf()
            subject = hashlib.sha256(f"{request.remote_addr}:{mode}".encode()).hexdigest()
            if not dependencies.consume_rate_limit("account-recovery", subject, limit=8, window=timedelta(hours=1)):
                abort(429, "Too many recovery attempts. Try again later.")
            if mode == "username":
                email = dependencies.valid_email(request.form.get("email") or "")
                airport_code = (request.form.get("airport_code") or "").strip().upper()
                if email:
                    identities = dependencies.PlatformIdentity.query.filter(
                        dependencies.db.func.lower(dependencies.PlatformIdentity.email) == email,
                    ).order_by(dependencies.PlatformIdentity.username).all()
                    if identities:
                        usernames = "\n".join(f"- {identity.username}" for identity in identities)
                        dependencies.send_email(email, "Your ATCRoster username", "The username(s) registered to this email address are:\n" + usernames + "\n\nIf you did not request this, contact your Unit Administrator.")
                    elif airport_code and (unit := dependencies.Unit.query.filter_by(code=airport_code).first()):
                        for admin_email in dependencies.unit_admin_emails(unit.id):
                            dependencies.send_email(admin_email, f"Username recovery assistance for {unit.code}", f"A user requested username assistance for {email}. Verify their identity using your approved local process before disclosing or updating account details.")
            else:
                username = dependencies.normalized_login(request.form.get("username") or "")
                identity = dependencies.PlatformIdentity.query.filter_by(username=username).first()
                if identity:
                    membership = dependencies.UnitMembership.query.filter_by(identity_id=identity.id, status="active").first()
                    unit_id = membership.unit_id if membership else None
                    approvers = dependencies.platform_support_emails() if identity.role == "superadmin" or (membership and membership.role == "UnitAdmin") else dependencies.unit_admin_emails(unit_id)
                    approvers = approvers or dependencies.platform_support_emails()
                    token = secrets.token_urlsafe(32)
                    dependencies.db.session.add(dependencies.RecoveryRequest(unit_id=unit_id, identity_id=identity.id, person_id=membership.person_id if membership else None, approval_token_digest=hashlib.sha256(token.encode()).hexdigest(), state="pending_approval", expires_at=dependencies.now() + timedelta(hours=24)))
                    dependencies.db.session.commit()
                    approval_url = url_for("approve_account_recovery", token=token, _external=True)
                    for approver in approvers:
                        dependencies.send_email(approver, "ATCRoster password reset approval required", f"A password reset was requested for {username}.\n\nReview and approve it here:\n{approval_url}\n\nThe link expires in 24 hours. Do not forward it.")
            flash("If the supplied details match an account, the recovery process has started. Check email or contact your administrator.", "ok")
            return redirect(url_for("account_recovery", mode=mode))
        return render_template("account_recovery.html", mode=mode)

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/recover", "account_recovery", account_recovery, methods=("GET", "POST"))

    return blueprint
