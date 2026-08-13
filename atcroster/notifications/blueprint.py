"""Authenticated notification-inbox routes extracted from ``app.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, flash, redirect, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class NotificationDependencies:
    db: Any
    Notification: Any
    current_unit_id: Callable[[], int]
    utcnow: Callable[[], Any]
    validate_csrf: Callable[[], None]


def create_notification_blueprint(dependencies: NotificationDependencies) -> Blueprint:
    """Create legacy endpoint-compatible notification routes."""
    blueprint = Blueprint("notifications", __name__)

    def profile_notifications_redirect():
        return redirect(url_for("staff_profile", sid=current_user.id) + "#notifications")

    @login_required
    def mark_all_read():
        dependencies.validate_csrf()
        dependencies.Notification.query.filter_by(
            unit_id=dependencies.current_unit_id(), recipient_id=current_user.id,
            read_at=None,
        ).update({"read_at": dependencies.utcnow()}, synchronize_session=False)
        dependencies.db.session.commit()
        return profile_notifications_redirect()

    @login_required
    def mark_read(notification_id: int):
        dependencies.validate_csrf()
        item = dependencies.Notification.query.filter_by(
            id=notification_id, unit_id=dependencies.current_unit_id(),
            recipient_id=current_user.id,
        ).first_or_404()
        if not item.read_at:
            item.read_at = dependencies.utcnow()
            dependencies.db.session.commit()
            flash("Notification marked as read.", "ok")
        return profile_notifications_redirect()

    @login_required
    def delete(notification_id: int):
        dependencies.validate_csrf()
        item = dependencies.Notification.query.filter_by(
            id=notification_id, unit_id=dependencies.current_unit_id(),
            recipient_id=current_user.id,
        ).first_or_404()
        if not item.read_at:
            flash("Mark the notification as read before deleting it.", "error")
            return profile_notifications_redirect()
        dependencies.db.session.delete(item)
        dependencies.db.session.commit()
        flash("Notification deleted.", "ok")
        return profile_notifications_redirect()

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule("/notifications/read", "notifications_read", mark_all_read, methods=("POST",))
        state.app.add_url_rule("/notifications/<int:notification_id>/read", "notification_read", mark_read, methods=("POST",))
        state.app.add_url_rule("/notifications/<int:notification_id>/delete", "notification_delete", delete, methods=("POST",))

    return blueprint
