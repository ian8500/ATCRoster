"""Unit account administration workflow."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash


@dataclass(frozen=True)
class UnitAccountsDependencies:
    db: Any
    Unit: Any
    Staff: Any
    PlatformIdentity: Any
    UnitMembership: Any
    SecureInvitation: Any
    MfaCredential: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    normalized_login: Callable[[str], str]
    now: Callable[[], Any]
    tenant_get: Callable[[Any, int], Any]
    consume_rate_limit: Callable[..., bool]
    central_security_event: Callable[..., None]


def create_unit_accounts_blueprint(dependencies: UnitAccountsDependencies) -> Blueprint:
    """Create unit-account administration with explicit compensation boundaries."""
    blueprint = Blueprint("unit_accounts", __name__)
    db = dependencies.db
    Unit = dependencies.Unit
    Staff = dependencies.Staff
    PlatformIdentity = dependencies.PlatformIdentity
    UnitMembership = dependencies.UnitMembership
    SecureInvitation = dependencies.SecureInvitation
    MfaCredential = dependencies.MfaCredential
    _current_unit_id = dependencies.current_unit_id
    is_admin_user = dependencies.is_admin_user
    _validate_csrf = dependencies.validate_csrf
    _normalized_login = dependencies.normalized_login
    utcnow = dependencies.now
    tenant_get = dependencies.tenant_get

    @login_required
    def unit_accounts():
        if not is_admin_user(current_user):
            abort(403)
        unit_id = _current_unit_id()
        unit = db.session.get(Unit, unit_id)
        if not unit:
            abort(404)
        if request.method == "POST":
            _validate_csrf()
            action = (request.form.get("action") or "").strip()
            if action == "create_invitation":
                role = (request.form.get("role") or "StaffUser").strip()
                allowed_roles = {
                    "UnitAdmin",
                    "RosterEditor",
                    "WatchManager",
                    "StaffUser",
                    "ReadOnlyAuditor",
                }
                if role not in allowed_roles:
                    abort(400, "Invalid invitation role.")
                try:
                    person_id = int(request.form.get("person_id") or 0)
                except ValueError:
                    abort(400, "Invalid roster person.")
                person = Staff.query.filter_by(id=person_id, unit_id=unit_id).first()
                if not person:
                    flash(
                        "Select an existing roster person before issuing access.",
                        "error",
                    )
                    return redirect(url_for("unit_accounts"))
                if (
                    UnitMembership.query.filter_by(unit_id=unit_id, person_id=person.id)
                    .filter(UnitMembership.status.in_(("active", "invited")))
                    .first()
                ):
                    flash(
                        "That roster person already has account access or a pending membership.",
                        "error",
                    )
                    return redirect(url_for("unit_accounts"))
                existing_invitation = SecureInvitation.query.filter_by(
                    unit_id=unit_id,
                    target_person_id=person.id,
                    accepted_at=None,
                    disabled_at=None,
                ).first()
                if existing_invitation:
                    flash(
                        "That roster person already has a pending invitation. "
                        "Disable it before issuing another.",
                        "error",
                    )
                    return redirect(url_for("unit_accounts"))
                try:
                    from account_limits import lock_unit_capacity

                    lock_unit_capacity(db, Unit, UnitMembership, unit_id)
                    raw_token = secrets.token_urlsafe(32)
                    invitation = SecureInvitation(
                        unit_id=unit_id,
                        token_digest=hashlib.sha256(raw_token.encode()).hexdigest(),
                        role=role,
                        target_person_id=person.id,
                        expires_at=utcnow() + timedelta(days=7),
                    )
                    db.session.add(invitation)
                    db.session.commit()
                except ValueError as exc:
                    db.session.rollback()
                    flash(str(exc), "error")
                    return redirect(url_for("unit_accounts"))
                invite_url = url_for(
                    "accept_invitation", token=raw_token, _external=True
                )
                flash(
                    f"Invitation for {person.name} created. Copy this link now; it is shown only "
                    f"once: {invite_url}",
                    "ok",
                )
                return redirect(url_for("unit_accounts"))
            if action == "create_account":
                name = (request.form.get("name") or "").strip()
                username = _normalized_login(request.form.get("username") or "")
                password = request.form.get("password") or ""
                if not name or not username or len(password) < 12:
                    flash(
                        "Name, username, and a 12-character password are required.",
                        "error",
                    )
                    return redirect(url_for("unit_accounts"))
                central_duplicate = PlatformIdentity.query.filter(
                    db.func.lower(PlatformIdentity.username) == username
                ).first()
                local_duplicate = Staff.query.filter(
                    db.func.lower(Staff.username) == username
                ).first()
                if central_duplicate or local_duplicate:
                    flash("That login identifier is unavailable.", "error")
                    return redirect(url_for("unit_accounts"))
                identity = None
                staff = None
                try:
                    password_hash = generate_password_hash(password)
                    identity = PlatformIdentity(
                        public_id=f"member-{secrets.token_hex(12)}",
                        username=username,
                        password_hash=password_hash,
                    )
                    db.session.add(identity)
                    db.session.commit()
                    staff = Staff(
                        unit_id=unit_id,
                        username=username,
                        name=name,
                        staff_no=f"{unit.code}-LOGIN-{secrets.token_hex(3).upper()}",
                        role="user",
                        is_operational=False,
                        membership_status="pending",
                    )
                    staff.password_hash = password_hash
                    db.session.add(staff)
                    db.session.commit()
                    membership = UnitMembership(
                        identity_id=identity.id,
                        unit_id=unit_id,
                        person_id=staff.id,
                        role="StaffUser",
                        status="invited",
                    )
                    db.session.add(membership)
                    db.session.flush()
                    from account_limits import activate_membership

                    activate_membership(db, Unit, UnitMembership, membership.id)
                    membership.activated_at = utcnow()
                    staff.membership_status = "active"
                    db.session.commit()
                    flash("Account activated.", "ok")
                except (ValueError, IntegrityError) as exc:
                    db.session.rollback()
                    if staff and staff.id:
                        pending_staff = db.session.get(Staff, staff.id)
                        if (
                            pending_staff
                            and pending_staff.membership_status != "active"
                        ):
                            db.session.delete(pending_staff)
                            db.session.commit()
                    if identity and identity.id:
                        orphan = db.session.get(PlatformIdentity, identity.id)
                        has_membership = UnitMembership.query.filter_by(
                            identity_id=identity.id
                        ).first()
                        if orphan and not has_membership:
                            db.session.delete(orphan)
                            db.session.commit()
                    message = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else "That login identifier is unavailable."
                    )
                    flash(message, "error")
                return redirect(url_for("unit_accounts"))
            if action == "deactivate":
                membership_id = int(request.form.get("membership_id") or 0)
                membership = UnitMembership.query.filter_by(
                    id=membership_id, unit_id=unit_id, status="active"
                ).first_or_404()
                if membership.person_id == current_user.id:
                    flash("You cannot deactivate your own account.", "error")
                else:
                    membership.status = "suspended"
                    membership.suspended_at = utcnow()
                    linked = (
                        tenant_get(Staff, membership.person_id)
                        if membership.person_id
                        else None
                    )
                    if linked:
                        linked.membership_status = "suspended"
                    db.session.commit()
                    flash("Account deactivated.", "ok")
                return redirect(url_for("unit_accounts"))
            if action == "reset_mfa":
                if not dependencies.consume_rate_limit(
                    "unit-admin-mfa-reset", current_user.id, limit=5,
                    window=timedelta(minutes=15),
                ):
                    abort(429, "Too many MFA reset attempts. Try again later.")
                try:
                    membership_id = int(request.form.get("membership_id") or 0)
                except ValueError:
                    abort(400, "Invalid account.")
                reason = (request.form.get("reason") or "").strip()
                if len(reason) < 5 or len(reason) > 200:
                    abort(400, "Provide a reset reason between 5 and 200 characters.")
                membership = UnitMembership.query.filter_by(
                    id=membership_id, unit_id=unit_id, status="active"
                ).first_or_404()
                if membership.person_id == current_user.id:
                    abort(403, "You cannot reset your own MFA.")
                target = tenant_get(Staff, membership.person_id) if membership.person_id else None
                identity = db.session.get(PlatformIdentity, membership.identity_id)
                if not target or not identity or target.role == "superadmin" or identity.public_id.startswith("platform-"):
                    abort(403, "Platform administrator MFA can only be reset through the platform-security process.")
                # A UnitAdmin target can only be reset by a different UnitAdmin;
                # self-reset is blocked above, including for the last administrator.
                if membership.role == "UnitAdmin":
                    requester = UnitMembership.query.filter_by(
                        unit_id=unit_id, person_id=current_user.id, status="active", role="UnitAdmin"
                    ).first()
                    if not requester:
                        abort(403)
                credential = MfaCredential.query.filter_by(
                    unit_id=unit_id, person_id=target.id
                ).with_for_update().first()
                if not credential:
                    credential = MfaCredential(
                        unit_id=unit_id, person_id=target.id, encrypted_secret="", enabled=False,
                        reset_required=True, recovery_codes_digest="[]",
                    )
                    db.session.add(credential)
                else:
                    credential.encrypted_secret = ""
                    credential.enabled = False
                    credential.reset_required = True
                    credential.last_used_step = None
                    credential.recovery_codes_digest = "[]"
                dependencies.central_security_event(
                    "airport_mfa_reset", "success", identity.id,
                    hashlib.sha256(current_user.username.lower().encode()).hexdigest()[:16],
                    f"unit={unit_id};target={target.id};reason={reason}",
                )
                db.session.commit()
                flash(
                    "MFA reset. The existing authenticator was revoked and the user must enrol a new authenticator after their next password login.",
                    "ok",
                )
                return redirect(url_for("unit_accounts"))
            if action == "restore":
                try:
                    membership_id = int(request.form.get("membership_id") or 0)
                except ValueError:
                    abort(400)
                membership = UnitMembership.query.filter_by(
                    id=membership_id, unit_id=unit_id, status="suspended"
                ).first_or_404()
                try:
                    from account_limits import activate_membership

                    activate_membership(db, Unit, UnitMembership, membership.id)
                    membership.suspended_at = None
                    membership.activated_at = membership.activated_at or utcnow()
                    linked = (
                        tenant_get(Staff, membership.person_id)
                        if membership.person_id
                        else None
                    )
                    if linked:
                        linked.membership_status = "active"
                    db.session.commit()
                    flash("Account restored.", "ok")
                except ValueError as exc:
                    db.session.rollback()
                    flash(str(exc), "error")
                return redirect(url_for("unit_accounts"))
            if action == "disable_invitation":
                try:
                    invitation_id = int(request.form.get("invitation_id") or 0)
                except ValueError:
                    abort(400)
                invitation = SecureInvitation.query.filter_by(
                    id=invitation_id,
                    unit_id=unit_id,
                    accepted_at=None,
                    disabled_at=None,
                ).first_or_404()
                invitation.disabled_at = utcnow()
                invitation.active_bootstrap_key = None
                db.session.commit()
                flash("Invitation disabled.", "ok")
                return redirect(url_for("unit_accounts"))
            abort(400)
        memberships = (
            UnitMembership.query.filter_by(unit_id=unit_id)
            .order_by(UnitMembership.id)
            .all()
        )
        active_count = sum(1 for row in memberships if row.status == "active")
        current_time = utcnow()
        pending_invitations = (
            SecureInvitation.query.filter(
                SecureInvitation.unit_id == unit_id,
                SecureInvitation.accepted_at.is_(None),
                SecureInvitation.disabled_at.is_(None),
                SecureInvitation.expires_at > current_time,
            )
            .order_by(SecureInvitation.expires_at)
            .all()
        )
        unavailable_person_ids = {
            row.person_id
            for row in memberships
            if row.person_id and row.status in {"active", "invited"}
        } | {
            row.target_person_id for row in pending_invitations if row.target_person_id
        }
        roster_people = (
            Staff.query.filter_by(unit_id=unit_id)
            .filter(Staff.role != "position_monitor")
            .order_by(Staff.name)
            .all()
        )
        eligible_people = [
            person
            for person in roster_people
            if person.id not in unavailable_person_ids
        ]
        return render_template(
            "unit_accounts.html",
            unit=unit,
            memberships=memberships,
            active_count=active_count,
            pending_invitations=pending_invitations,
            eligible_people=eligible_people,
            staff_by_id={person.id: person for person in roster_people},
            mfa_by_person={
                credential.person_id: credential
                for credential in MfaCredential.query.filter_by(unit_id=unit_id).all()
            },
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/unit/accounts", "unit_accounts", unit_accounts, methods=("GET", "POST")
        )

    return blueprint
