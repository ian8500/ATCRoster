"""Module launcher route extracted from the legacy application module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class ModuleDependencies:
    FeatureFlag: Any
    briefing_enabled: Callable[[int], bool]
    training_enabled: Callable[[int], bool]
    competency_enabled: Callable[[int], bool]
    is_admin_user: Callable[[Any], bool]


def create_module_blueprint(dependencies: ModuleDependencies) -> Blueprint:
    """Create the legacy endpoint-compatible module launcher."""
    blueprint = Blueprint("modules", __name__)

    @login_required
    def module_home():
        if current_user.role == "superadmin":
            return redirect(url_for("platform_admin"))
        return render_template(
            "module_home.html",
            show_briefing=dependencies.briefing_enabled(current_user.unit_id),
            show_training=dependencies.training_enabled(current_user.unit_id),
            show_competency=dependencies.competency_enabled(current_user.unit_id),
            show_handover=dependencies.FeatureFlag.query.filter_by(
                unit_id=current_user.unit_id, key="handover_module", enabled=True,
            ).first() is not None,
            show_administration=dependencies.is_admin_user(current_user),
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/modules", "module_home", module_home, methods=("GET",))

    return blueprint
