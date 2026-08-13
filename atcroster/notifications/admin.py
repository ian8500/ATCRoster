"""Tenant-scoped SMS audit and a controlled administrator test send."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class SmsAdministrationDependencies:
    SmsAudit: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    configuration: Any
    normalise_sms_number: Callable[[str | None], str]
    send_sms: Callable[..., tuple[bool, str]]
    record_sms_audit: Callable[..., None]


def create_sms_administration_blueprint(dependencies: SmsAdministrationDependencies) -> Blueprint:
    blueprint = Blueprint("sms_administration", __name__)

    def require_admin() -> None:
        if not dependencies.is_admin_user(current_user):
            abort(403)

    @login_required
    def admin_sms_audit():
        require_admin()
        rows = dependencies.SmsAudit.query.filter_by(
            unit_id=dependencies.current_unit_id()
        ).order_by(dependencies.SmsAudit.sent_at.desc(), dependencies.SmsAudit.id.desc()).limit(1000).all()
        senders = dependencies.configuration.sender_options()
        return render_template("admin_sms_audit.html", rows=rows, senders=senders,
                               sms_ready=dependencies.configuration.service_configured())

    @login_required
    def send_test_sms():
        require_admin()
        dependencies.validate_csrf()
        if not dependencies.consume_rate_limit("sms-test", current_user.id, limit=3, window_seconds=3600):
            abort(429, "Too many test SMS requests. Try again later.")
        recipient = dependencies.normalise_sms_number(request.form.get("recipient_number"))
        senders = dependencies.configuration.sender_options()
        sender = dependencies.configuration.default_number("sms_default_sender", senders)
        if not recipient:
            flash("Enter a valid recipient number in E.164 format or a UK mobile number.", "error")
        elif not sender or not dependencies.configuration.service_configured():
            flash("ClickSend credentials or this unit's verified sender are not configured.", "error")
        else:
            body = "ATCO Roster SMS test. ClickSend messaging is configured correctly."
            ok, detail = dependencies.send_sms(recipient, body, sender)
            dependencies.record_sms_audit(sender_number=sender, recipient_number=recipient,
                                          recipient_label="SMS test", body=body,
                                          message_type="test", provider_message_id=detail,
                                          delivery_status="submitted" if ok else "failed")
            flash("Test SMS accepted by ClickSend." if ok else f"Test SMS was not accepted: {detail}", "ok" if ok else "error")
        return redirect(url_for("admin_sms_audit"))

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule("/admin/sms-audit", "admin_sms_audit", admin_sms_audit, methods=("GET",))
        state.app.add_url_rule("/admin/sms-test", "send_test_sms", send_test_sms, methods=("POST",))

    return blueprint
