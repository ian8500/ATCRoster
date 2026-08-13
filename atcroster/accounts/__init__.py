"""Account security route modules."""

from .password import PasswordDependencies, create_password_blueprint

__all__ = ("PasswordDependencies", "create_password_blueprint")
