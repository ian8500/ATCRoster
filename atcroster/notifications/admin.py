"""SMS audit administration and authenticated provider webhook routes."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import ProgrammingError


@dataclass(frozen=True)
class SmsAdministrationDependencies:
    db: Any
    SmsAudit: Any
    SmsSenderRegistration: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    utcnow: Callable[[], Any]


def create_sms_administration_blueprint(dependencies: SmsAdministrationDependencies) -> Blueprint:
    """Create legacy endpoint-compatible SMS administration routes."""
    blueprint = Blueprint("sms_administration", __name__)

    def require_admin() -> None:
        if not dependencies.is_admin_user(current_user):
            abort(403)

    @login_required
    def admin_sms_audit():
        require_admin()
        rows = dependencies.SmsAudit.query.order_by(
            dependencies.SmsAudit.sent_at.desc(), dependencies.SmsAudit.id.desc(),
        ).limit(1000).all()
        try:
            registrations = dependencies.SmsSenderRegistration.query.filter_by(
                unit_id=dependencies.current_unit_id(), provider="clicksend",
            ).order_by(dependencies.SmsSenderRegistration.verification_requested_at.desc()).all()
        except ProgrammingError:
            dependencies.db.session.rollback()
            registrations = []
        return render_template("admin_sms_audit.html", rows=rows, registrations=registrations)

    @login_required
    def confirm_sms_sender(registration_id: int):
        require_admin()
        dependencies.validate_csrf()
        row = dependencies.SmsSenderRegistration.query.filter_by(
            id=registration_id, unit_id=dependencies.current_unit_id(), provider="clicksend",
        ).first_or_404()
        expiry = request.form.get("expires_at", "").strip()
        try:
            expires_at = datetime.strptime(expiry, "%Y-%m-%d") if expiry else dependencies.utcnow() + timedelta(days=365)
        except ValueError:
            flash("Enter a valid verification expiry date.", "error")
            return redirect(url_for("admin_sms_audit"))
        row.status = "verified"
        row.verified_at = dependencies.utcnow()
        row.expires_at = expires_at
        row.provider_identifier = (request.form.get("provider_identifier") or row.number).strip()[:120]
        dependencies.db.session.commit()
        flash("ClickSend sender verification recorded.", "ok")
        return redirect(url_for("admin_sms_audit"))

    def messagemedia_delivery_webhook():
        expected = os.getenv("MESSAGEMEDIA_WEBHOOK_TOKEN", "")
        supplied = request.headers.get("X-ATCRoster-Webhook-Token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            abort(403)
        payload = request.get_json(silent=True) or request.form.to_dict()
        message_id = str(payload.get("id") or payload.get("message_id") or payload.get("mtId") or "")
        status = str(payload.get("status") or payload.get("statusCode") or "submitted").lower()[:30]
        if message_id:
            row = dependencies.SmsAudit.query.filter_by(
                provider="messagemedia", provider_message_id=message_id,
            ).first()
            if row:
                row.delivery_status = status
                dependencies.db.session.commit()
        return Response(status=204)

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule("/admin/sms-audit", "admin_sms_audit", admin_sms_audit, methods=("GET",))
        state.app.add_url_rule("/admin/sms-senders/<int:registration_id>/confirm", "confirm_sms_sender", confirm_sms_sender, methods=("POST",))
        state.app.add_url_rule("/webhooks/messagemedia/delivery", "messagemedia_delivery_webhook", messagemedia_delivery_webhook, methods=("POST",))

    return blueprint
