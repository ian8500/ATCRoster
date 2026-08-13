"""Unit SMS messaging route extracted from the legacy application module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from flask import Blueprint, abort, flash, render_template, request
from flask_login import current_user, login_required


@dataclass(frozen=True)
class MessagingDependencies:
    db: Any
    Staff: Any
    Watch: Any
    Assignment: Any
    current_unit_id: Callable[[], int]
    utcnow: Callable[[], Any]
    can_send_unit_messages: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    sms_configuration: Any
    normalise_sms_number: Callable[[str | None], str]
    send_sms: Callable[..., tuple[bool, str]]
    record_sms_audit: Callable[..., None]
    flash_sms_result: Callable[..., None]


def create_messaging_blueprint(dependencies: MessagingDependencies) -> Blueprint:
    """Create the existing `/messages` endpoint without changing its contract."""
    blueprint = Blueprint("messaging", __name__)

    @login_required
    def unit_messages():
        if not dependencies.can_send_unit_messages(current_user):
            abort(403)
        people = dependencies.Staff.query.filter_by(membership_status="active").filter(
            dependencies.Staff.role != "position_monitor",
        ).order_by(dependencies.Staff.name).all()
        watches = dependencies.Watch.query.order_by(
            dependencies.Watch.order_index, dependencies.Watch.name,
        ).all()
        selected_scope = request.form.get("scope", "all")
        selected_recipient = request.form.get("recipient_id", "")
        selected_watch = request.form.get("watch_id", "")
        sender_options = dependencies.sms_configuration.sender_options()
        operational_options = dependencies.sms_configuration.operational_options()
        default_sender = dependencies.sms_configuration.default_number("sms_default_sender", sender_options)
        default_operational = dependencies.sms_configuration.default_number("sms_default_operational_number", operational_options)
        selected_sender = dependencies.normalise_sms_number(request.form.get("sender_number") or default_sender)
        selected_operational = dependencies.normalise_sms_number(request.form.get("operational_number") or default_operational)
        template = request.form.get("template", "custom")
        message = (request.form.get("message") or "").strip()
        preview: list[tuple[str, str]] = []

        if request.method == "POST":
            dependencies.validate_csrf()
            recipients: list[Any] = []
            direct_recipient = None
            allowed_senders = {item["number"] for item in sender_options}
            allowed_operational = {item["number"] for item in operational_options}
            if selected_sender not in allowed_senders:
                abort(400, "Choose your verified mobile sender, or the configured unit fallback.")
            if selected_scope == "all":
                recipients = people
            elif selected_scope == "watch" and selected_watch.isdigit():
                recipients = [person for person in people if person.watch_id == int(selected_watch)]
            elif selected_scope == "individual" and selected_recipient.isdigit():
                recipients = [person for person in people if person.id == int(selected_recipient)]
            elif selected_scope == "operational" and selected_operational in allowed_operational:
                direct_recipient = selected_operational
            if not recipients and not direct_recipient:
                flash("Choose at least one recipient.", "error")
            elif direct_recipient and template == "today_shift":
                flash("Shift reminders can only be sent to rostered people.", "error")
            elif template == "today_shift":
                today = date.today()
                assignments = dependencies.Assignment.query.filter(
                    dependencies.Assignment.day == today,
                    dependencies.Assignment.staff_id.in_([person.id for person in recipients]),
                ).all()
                assignment_map = {row.staff_id: row for row in assignments}
                sent, failures = 0, []
                for person in recipients:
                    assignment = assignment_map.get(person.id)
                    body = f"Hello {person.name}, you are rostered for {assignment.code if assignment else 'no assigned'} shift today ({today.strftime('%d %b %Y')})."
                    ok, detail = dependencies.send_sms(person.phone_number, body, from_number=selected_sender)
                    preview.append((person.name, body))
                    if ok:
                        dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=person.phone_number, recipient_label=person.name, body=body, message_type="shift_reminder", provider_message_id=detail)
                        sent += 1
                    else:
                        failures.append((person, detail))
                        dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=person.phone_number, recipient_label=person.name, body=body, message_type="shift_reminder", provider_message_id=detail, delivery_status="failed")
                dependencies.flash_sms_result(sent, failures)
            elif not message:
                flash("Enter a custom message.", "error")
            elif len(message) > 480:
                flash("Message is too long (limit 480 characters).", "error")
            elif direct_recipient:
                ok, detail = dependencies.send_sms(direct_recipient, message, from_number=selected_sender)
                label = next((item["label"] for item in operational_options if item["number"] == direct_recipient), direct_recipient)
                preview = [(label, message)]
                if ok:
                    dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=direct_recipient, recipient_label=label, body=message, message_type="operational", provider_message_id=detail)
                else:
                    dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=direct_recipient, recipient_label=label, body=message, message_type="operational", provider_message_id=detail, delivery_status="failed")
                dependencies.flash_sms_result(1 if ok else 0, [] if ok else [(None, detail)])
            else:
                sent, failures = 0, []
                for person in recipients:
                    ok, detail = dependencies.send_sms(person.phone_number, message, from_number=selected_sender)
                    if ok:
                        dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=person.phone_number, recipient_label=person.name, body=message, message_type="unit", provider_message_id=detail)
                        sent += 1
                    else:
                        failures.append((person, detail))
                        dependencies.record_sms_audit(sender_number=selected_sender, recipient_number=person.phone_number, recipient_label=person.name, body=message, message_type="unit", provider_message_id=detail, delivery_status="failed")
                preview = [(person.name, message) for person in recipients]
                dependencies.flash_sms_result(sent, failures)
        return render_template(
            "messages.html", people=people, watches=watches,
            sms_ready=dependencies.sms_configuration.service_configured(), template=template,
            message=message, selected_scope=selected_scope,
            selected_recipient=selected_recipient, selected_watch=selected_watch,
            sender_options=sender_options, operational_options=operational_options,
            selected_sender=selected_sender, selected_operational=selected_operational,
            preview=preview,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/messages", "unit_messages", unit_messages, methods=("GET", "POST"))

    return blueprint
