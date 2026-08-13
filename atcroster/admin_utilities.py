"""Small administration/permission routes extracted from the legacy module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, render_template, request
from flask_login import current_user, login_required


@dataclass(frozen=True)
class AdminUtilityDependencies:
    ChangeLog: Any
    is_admin_user: Callable[[Any], bool]


def create_admin_utility_blueprint(dependencies: AdminUtilityDependencies) -> Blueprint:
    blueprint = Blueprint("admin_utilities", __name__)

    @login_required
    def permission_summary():
        admin = dependencies.is_admin_user(current_user)
        watch_manager = bool(getattr(current_user, "is_wm", False))
        deputy_watch_manager = bool(getattr(current_user, "is_dwm", False))
        return {
            "is_admin_user": admin,
            "is_wm": watch_manager,
            "is_dwm": deputy_watch_manager,
            "final_can_edit": admin or watch_manager or deputy_watch_manager,
        }

    @login_required
    def change_log_page():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        ym = request.args.get("ym", "").strip() or None
        entity_type = request.args.get("entity_type", "").strip() or None
        who = request.args.get("who", "").strip() or None
        query = dependencies.ChangeLog.query.order_by(dependencies.ChangeLog.when.desc())
        if ym:
            query = query.filter(dependencies.ChangeLog.context_month == ym)
        if entity_type:
            query = query.filter(dependencies.ChangeLog.entity_type == entity_type)
        if who and who.isdigit():
            query = query.filter(dependencies.ChangeLog.who_user_id == int(who))
        return render_template("change_log.html", rows=query.limit(500).all(), ym=ym, entity_type=entity_type, who=who)

    @blueprint.record_once
    def register_legacy_endpoints(state) -> None:
        state.app.add_url_rule("/__can", "__can", permission_summary, methods=("GET",))
        state.app.add_url_rule("/admin/change-log", "change_log_page", change_log_page, methods=("GET",))

    return blueprint
