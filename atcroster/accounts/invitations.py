"""Public invitation acceptance route."""

from __future__ import annotations

import hashlib
import re
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


@dataclass(frozen=True)
class InvitationAcceptanceDependencies:
    db: Any
    SecureInvitation: Any
    Unit: Any
    DatabaseRoutingMetadata: Any
    Staff: Any
    deployment_environment: str
    consume_rate_limit: Callable[..., bool]
    now: Callable[[], Any]
    bind_authenticated_unit: Callable[[int, str | None], Any]
    validate_csrf: Callable[[], None]
    valid_email: Callable[[str], str]
    run_signup: Callable[..., Any]
    signup_error: type[Exception]


def create_invitation_acceptance_blueprint(
    dependencies: InvitationAcceptanceDependencies,
) -> Blueprint:
    blueprint = Blueprint("invitation_acceptance", __name__)
    db = dependencies.db
    SecureInvitation = dependencies.SecureInvitation
    Unit = dependencies.Unit
    DatabaseRoutingMetadata = dependencies.DatabaseRoutingMetadata
    Staff = dependencies.Staff
    DEPLOYMENT_ENV = dependencies.deployment_environment
    _consume_rate_limit = dependencies.consume_rate_limit
    utcnow = dependencies.now
    bind_authenticated_unit = dependencies.bind_authenticated_unit
    _validate_csrf = dependencies.validate_csrf
    _valid_email = dependencies.valid_email
    _run_invitation_signup = dependencies.run_signup
    SignupWorkflowError = dependencies.signup_error

    def accept_invitation(token):
        """Accept a one-time, expiring invitation without trusting tenant input."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token or ""):
            abort(404)
        digest = hashlib.sha256(token.encode()).hexdigest()
        if not _consume_rate_limit(
            "invitation-acceptance",
            digest,
            limit=20,
            window=timedelta(hours=1),
        ):
            abort(429, "Too many invitation attempts.")
        invitation = SecureInvitation.query.filter_by(
            token_digest=digest
        ).first_or_404()
        expiry_now = utcnow()
        if invitation.expires_at.tzinfo is None:
            expiry_now = expiry_now.replace(tzinfo=None)
        if (
            invitation.accepted_at
            or invitation.disabled_at
            or invitation.expires_at <= expiry_now
        ):
            abort(410, "This invitation has expired or has already been used.")
        unit = db.session.get(Unit, invitation.unit_id)
        routing = (
            db.session.get(DatabaseRoutingMetadata, invitation.unit_id)
            if unit
            else None
        )
        if (
            not unit
            or unit.status not in {"active", "provisioning"}
            or (
                unit.status == "provisioning"
                and (not routing or routing.provisioning_state != "invitation_issued")
            )
        ):
            abort(409, "This airport account is not accepting invitations.")
        if DEPLOYMENT_ENV == "production" and not routing:
            abort(503, "Operational database routing is unavailable.")
        # A targeted invitation refers to a person in the airport's operational
        # database. Establish that trusted route before resolving or displaying
        # the roster profile, including on the initial anonymous GET.
        g.tenant_context_token = bind_authenticated_unit(
            invitation.unit_id,
            routing.secret_name if routing else None,
        )
        target_person = None
        if invitation.target_person_id:
            target_person = Staff.query.filter_by(
                id=invitation.target_person_id,
                unit_id=invitation.unit_id,
            ).first()
            if not target_person:
                abort(410, "The linked roster person is no longer available.")
        if request.method == "POST":
            _validate_csrf()
            name = (
                target_person.name
                if target_person
                else (request.form.get("name") or "").strip()
            )
            username = (request.form.get("username") or "").strip().lower()
            email = _valid_email(request.form.get("email") or "")
            password = request.form.get("password") or ""
            if not name or not re.fullmatch(r"[a-z0-9._-]{3,120}", username):
                flash("Enter a name and a valid username.", "error")
                return render_template(
                    "invitation_accept.html",
                    invitation=invitation,
                    unit=unit,
                    target_person=target_person,
                ), 400
            if len(password) < 12:
                flash("Use a password of at least 12 characters.", "error")
                return render_template(
                    "invitation_accept.html",
                    invitation=invitation,
                    unit=unit,
                    target_person=target_person,
                ), 400
            if not email:
                flash("Enter a valid email address.", "error")
                return render_template(
                    "invitation_accept.html",
                    invitation=invitation,
                    unit=unit,
                    target_person=target_person,
                ), 400
            try:
                from signup_locking import invitation_signup_lock

                with invitation_signup_lock(db, invitation.id):
                    locked_invitation = (
                        SecureInvitation.query.filter_by(
                            id=invitation.id,
                            accepted_at=None,
                            disabled_at=None,
                        )
                        .with_for_update()
                        .first()
                    )
                    if not locked_invitation:
                        abort(410, "This invitation has already been used.")
                    _run_invitation_signup(
                        locked_invitation,
                        unit,
                        name,
                        username,
                        password,
                        email=email,
                    )
            except (SignupWorkflowError, ValueError) as exc:
                flash(str(exc), "error")
                return render_template(
                    "invitation_accept.html",
                    invitation=invitation,
                    unit=unit,
                    target_person=target_person,
                ), 409
            flash(
                "Kiosk account created. Sign in on the dedicated display."
                if invitation.role == "PositionMonitor"
                else "Account created. Sign in and configure MFA.",
                "ok",
            )
            return redirect(url_for("login"))
        return render_template(
            "invitation_accept.html",
            invitation=invitation,
            unit=unit,
            target_person=target_person,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/invite/<token>",
            "accept_invitation",
            accept_invitation,
            methods=("GET", "POST"),
        )

    return blueprint
