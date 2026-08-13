"""Notification delivery and inbox presentation domain."""

from .blueprint import NotificationDependencies, create_notification_blueprint

__all__ = ("NotificationDependencies", "create_notification_blueprint")
