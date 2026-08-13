"""Shared administration landing route extracted from ``app.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, render_template
from flask_login import current_user, login_required


@dataclass(frozen=True)
class AdministrationDependencies:
    is_admin_user: Callable[[Any], bool]
    live_position_enabled: Callable[[int], bool]


def create_administration_blueprint(dependencies: AdministrationDependencies) -> Blueprint:
    blueprint = Blueprint("administration", __name__)

    @login_required
    def administration_home():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        return render_template(
            "administration_home.html",
            show_live_position=dependencies.live_position_enabled(current_user.unit_id),
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/administration", "administration_home", administration_home, methods=("GET",))

    return blueprint
