"""Notification delivery and inbox presentation domain."""

from .blueprint import NotificationDependencies, create_notification_blueprint
from .sms import normalise_sms_number, normalise_uk_mobile, send_via_messagemedia
from .email import email_service_configured, send_account_email, valid_email

__all__ = (
    "NotificationDependencies", "create_notification_blueprint",
    "normalise_sms_number", "normalise_uk_mobile", "send_via_messagemedia",
    "email_service_configured", "send_account_email", "valid_email",
)
