"""Live Position kiosk-account administration route."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class KioskAccountDependencies:
    db: Any
    Unit: Any
    Staff: Any
    UnitMembership: Any
    SecureInvitation: Any
    current_unit_id: Callable[[], int]
    live_position_enabled: Callable[[int], bool]
    tenant_get: Callable[[Any, int], Any]
    utcnow: Callable[[], Any]
    validate_csrf: Callable[[], None]
    is_admin_user: Callable[[Any], bool]


def create_kiosk_account_blueprint(dependencies: KioskAccountDependencies) -> Blueprint:
    blueprint = Blueprint("kiosk_accounts", __name__)

    @login_required
    def kiosk_accounts():
        """Provision dedicated, module-scoped Live Position kiosk identities."""
        if not dependencies.is_admin_user(current_user):
            abort(403)
        unit_id = dependencies.current_unit_id()
        if not dependencies.live_position_enabled(unit_id):
            abort(404)
        unit = dependencies.db.session.get(dependencies.Unit, unit_id)
        if not unit:
            abort(404)
        if request.method == "POST":
            dependencies.validate_csrf()
            action = (request.form.get("action") or "").strip()
            if action == "create_invitation":
                from account_limits import lock_unit_capacity

                try:
                    lock_unit_capacity(
                        dependencies.db,
                        dependencies.Unit,
                        dependencies.UnitMembership,
                        unit_id,
                    )
                    raw_token = secrets.token_urlsafe(32)
                    invitation = dependencies.SecureInvitation(
                        unit_id=unit_id,
                        token_digest=hashlib.sha256(raw_token.encode()).hexdigest(),
                        role="PositionMonitor",
                        expires_at=dependencies.utcnow() + timedelta(days=7),
                    )
                    dependencies.db.session.add(invitation)
                    dependencies.db.session.commit()
                except ValueError as exc:
                    dependencies.db.session.rollback()
                    flash(str(exc), "error")
                    return redirect(url_for("kiosk_accounts"))
                invite_url = url_for("accept_invitation", token=raw_token, _external=True)
                flash(
                    "Kiosk setup link created. Copy it now; it is shown only "
                    f"once: {invite_url}",
                    "ok",
                )
                return redirect(url_for("kiosk_accounts"))
            if action == "disable_invitation":
                invitation_id = int(request.form.get("invitation_id") or 0)
                invitation = dependencies.SecureInvitation.query.filter_by(
                    id=invitation_id,
                    unit_id=unit_id,
                    role="PositionMonitor",
                    accepted_at=None,
                    disabled_at=None,
                ).first_or_404()
                invitation.disabled_at = dependencies.utcnow()
                dependencies.db.session.commit()
                flash("Kiosk setup link disabled.", "ok")
                return redirect(url_for("kiosk_accounts"))
            if action in {"deactivate", "restore"}:
                membership_id = int(request.form.get("membership_id") or 0)
                expected_status = "active" if action == "deactivate" else "suspended"
                membership = dependencies.UnitMembership.query.filter_by(
                    id=membership_id,
                    unit_id=unit_id,
                    role="PositionMonitor",
                    status=expected_status,
                ).first_or_404()
                linked = (
                    dependencies.tenant_get(dependencies.Staff, membership.person_id)
                    if membership.person_id
                    else None
                )
                if action == "deactivate":
                    membership.status = "suspended"
                    membership.suspended_at = dependencies.utcnow()
                    if linked:
                        linked.membership_status = "suspended"
                    message = "Kiosk account deactivated."
                else:
                    from account_limits import activate_membership

                    try:
                        activate_membership(
                            dependencies.db,
                            dependencies.Unit,
                            dependencies.UnitMembership,
                            membership.id,
                        )
                    except ValueError as exc:
                        dependencies.db.session.rollback()
                        flash(str(exc), "error")
                        return redirect(url_for("kiosk_accounts"))
                    membership.suspended_at = None
                    membership.activated_at = membership.activated_at or dependencies.utcnow()
                    if linked:
                        linked.membership_status = "active"
                    message = "Kiosk account restored."
                dependencies.db.session.commit()
                flash(message, "ok")
                return redirect(url_for("kiosk_accounts"))
            abort(400)
        invitations = dependencies.SecureInvitation.query.filter(
            dependencies.SecureInvitation.unit_id == unit_id,
            dependencies.SecureInvitation.role == "PositionMonitor",
            dependencies.SecureInvitation.accepted_at.is_(None),
            dependencies.SecureInvitation.disabled_at.is_(None),
            dependencies.SecureInvitation.expires_at > dependencies.utcnow(),
        ).order_by(dependencies.SecureInvitation.expires_at).all()
        memberships = dependencies.UnitMembership.query.filter_by(
            unit_id=unit_id, role="PositionMonitor"
        ).order_by(dependencies.UnitMembership.id).all()
        people = {
            row.id: row
            for row in dependencies.Staff.query.filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.id.in_([row.person_id for row in memberships if row.person_id]),
            ).all()
        }
        return render_template(
            "kiosk_accounts.html",
            invitations=invitations,
            memberships=memberships,
            people=people,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/administration/kiosk-accounts",
            "kiosk_accounts",
            kiosk_accounts,
            methods=("GET", "POST"),
        )

    return blueprint
