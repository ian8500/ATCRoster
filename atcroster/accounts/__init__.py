"""Account security route modules."""

from .kiosk import KioskAccountDependencies, create_kiosk_account_blueprint
from .password import PasswordDependencies, create_password_blueprint

__all__ = (
    "KioskAccountDependencies",
    "PasswordDependencies",
    "create_kiosk_account_blueprint",
    "create_password_blueprint",
)
