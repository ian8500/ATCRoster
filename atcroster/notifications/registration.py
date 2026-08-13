"""Notification-domain blueprint composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .admin import SmsAdministrationDependencies, create_sms_administration_blueprint
from .blueprint import NotificationDependencies, create_notification_blueprint
from .messaging import MessagingDependencies, create_messaging_blueprint


@dataclass(frozen=True)
class NotificationRegistrationDependencies:
    db: Any
    Notification: Any
    SmsAudit: Any
    SmsSenderRegistration: Any
    Staff: Any
    Watch: Any
    Assignment: Any
    current_unit_id: Callable[[], int]
    now: Callable[[], Any]
    validate_csrf: Callable[[], None]
    is_admin_user: Callable[[Any], bool]
    can_send_unit_messages: Callable[[Any], bool]
    sms_configuration: Any
    normalise_sms_number: Callable[[str | None], str]
    send_sms: Callable[..., tuple[bool, str]]
    record_sms_audit: Callable[..., None]
    flash_sms_result: Callable[..., None]


def register_notification_blueprints(
    app: Any, deps: NotificationRegistrationDependencies
) -> None:
    app.register_blueprint(create_notification_blueprint(NotificationDependencies(
        db=deps.db, Notification=deps.Notification,
        current_unit_id=deps.current_unit_id, utcnow=deps.now,
        validate_csrf=deps.validate_csrf,
    )))
    app.register_blueprint(create_sms_administration_blueprint(
        SmsAdministrationDependencies(
            db=deps.db, SmsAudit=deps.SmsAudit,
            SmsSenderRegistration=deps.SmsSenderRegistration,
            current_unit_id=deps.current_unit_id,
            is_admin_user=deps.is_admin_user,
            validate_csrf=deps.validate_csrf, utcnow=deps.now,
        )
    ))
    app.register_blueprint(create_messaging_blueprint(MessagingDependencies(
        db=deps.db, Staff=deps.Staff, Watch=deps.Watch,
        Assignment=deps.Assignment,
        SmsSenderRegistration=deps.SmsSenderRegistration,
        current_unit_id=deps.current_unit_id, utcnow=deps.now,
        can_send_unit_messages=deps.can_send_unit_messages,
        validate_csrf=deps.validate_csrf,
        sms_configuration=deps.sms_configuration,
        normalise_sms_number=deps.normalise_sms_number,
        send_sms=deps.send_sms, record_sms_audit=deps.record_sms_audit,
        flash_sms_result=deps.flash_sms_result,
    )))
