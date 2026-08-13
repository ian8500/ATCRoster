"""Account security route modules."""

from .kiosk import KioskAccountDependencies, create_kiosk_account_blueprint
from .contacts import platform_support_emails, unit_admin_emails
from .lifecycle import record_successful_login
from .password import PasswordDependencies, create_password_blueprint
from .recovery import active_recovery_from_digest

__all__ = (
    "KioskAccountDependencies",
    "PasswordDependencies",
    "create_kiosk_account_blueprint",
    "platform_support_emails",
    "create_password_blueprint",
    "record_successful_login",
    "unit_admin_emails",
    "active_recovery_from_digest",
)
