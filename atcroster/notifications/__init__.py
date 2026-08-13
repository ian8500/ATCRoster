"""Notification delivery and inbox presentation domain."""

from .blueprint import NotificationDependencies, create_notification_blueprint
from .sms import (
    normalise_sms_number, normalise_uk_mobile, parse_sms_number_lines,
    send_via_messagemedia,
)
from .email import email_service_configured, send_account_email, valid_email
from .configuration import SmsConfigurationService
from .audit import SmsAuditService
from .overtime import OvertimeSmsService, default_overtime_sms_body
from .runtime import NotificationRuntime, NotificationRuntimeDependencies
from .registration import (
    NotificationRegistrationDependencies,
    register_notification_blueprints,
)
from .admin import SmsAdministrationDependencies, create_sms_administration_blueprint
from .messaging import MessagingDependencies, create_messaging_blueprint

__all__ = (
    "NotificationDependencies", "create_notification_blueprint",
    "normalise_sms_number", "normalise_uk_mobile", "send_via_messagemedia",
    "parse_sms_number_lines",
    "email_service_configured", "send_account_email", "valid_email",
    "SmsConfigurationService",
    "SmsAuditService",
    "OvertimeSmsService", "default_overtime_sms_body",
    "NotificationRuntime", "NotificationRuntimeDependencies",
    "NotificationRegistrationDependencies", "register_notification_blueprints",
    "SmsAdministrationDependencies", "create_sms_administration_blueprint",
    "MessagingDependencies", "create_messaging_blueprint",
)
