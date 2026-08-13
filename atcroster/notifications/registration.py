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
    notifications: Any


def create_notification_registration_dependencies(
    *, db: Any, operational_models: Any, **services: Any
) -> NotificationRegistrationDependencies:
    """Bind notification routes to canonical operational models."""
    return NotificationRegistrationDependencies(
        db=db,
        Notification=operational_models.Notification,
        SmsAudit=operational_models.SmsAudit,
        SmsSenderRegistration=operational_models.SmsSenderRegistration,
        Staff=operational_models.Staff,
        Watch=operational_models.Watch,
        Assignment=operational_models.Assignment,
        **services,
    )


def register_notification_runtime_blueprints(
    app: Any, *, db: Any, operational_models: Any, services: Any
) -> None:
    """Register notification routes from notification-owned runtime inputs."""
    register_notification_blueprints(
        app,
        create_notification_registration_dependencies(
            db=db, operational_models=operational_models,
            current_unit_id=services.current_unit_id, now=services.now,
            validate_csrf=services.validate_csrf,
            is_admin_user=services.is_admin_user,
            can_send_unit_messages=services.can_send_unit_messages,
            notifications=services.notifications,
        ),
    )


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
        notifications=deps.notifications,
    )))
