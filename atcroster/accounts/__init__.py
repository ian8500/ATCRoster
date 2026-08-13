"""Account security route modules."""

from .kiosk import KioskAccountDependencies, create_kiosk_account_blueprint
from .contacts import platform_support_emails, unit_admin_emails
from .phones import normalise_phone_number
from .lifecycle import record_successful_login
from .password import PasswordDependencies, create_password_blueprint
from .recovery import active_recovery_from_digest
from .recovery_blueprint import RecoveryRequestDependencies, create_recovery_request_blueprint
from .registration import (
    AccountRegistrationDependencies,
    create_account_registration_dependencies,
    register_account_runtime_blueprints,
    register_account_blueprints,
)

__all__ = (
    "KioskAccountDependencies",
    "PasswordDependencies",
    "create_kiosk_account_blueprint",
    "platform_support_emails",
    "normalise_phone_number",
    "create_password_blueprint",
    "record_successful_login",
    "unit_admin_emails",
    "active_recovery_from_digest",
    "RecoveryRequestDependencies",
    "create_recovery_request_blueprint",
    "AccountRegistrationDependencies",
    "create_account_registration_dependencies",
    "register_account_runtime_blueprints",
    "register_account_blueprints",
)
