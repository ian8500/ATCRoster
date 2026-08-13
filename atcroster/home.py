"""Authenticated landing-route policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from flask import Blueprint, redirect, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class HomeDependencies:
    db: Any
    Unit: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]


def create_home_blueprint(dependencies: HomeDependencies) -> Blueprint:
    """Create the root redirect with the established onboarding gate."""
    blueprint = Blueprint("home", __name__)

    @login_required
    def index():
        if dependencies.is_admin_user(current_user):
            unit = dependencies.db.session.get(dependencies.Unit, dependencies.current_unit_id())
            if unit and int(unit.onboarding_step or 0) < 100:
                return redirect(url_for("unit_onboarding"))
        today = date.today()
        return redirect(url_for("roster_month", ym=f"{today.year}-{today.month:02d}"))

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule("/", "index", index, methods=("GET",))

    return blueprint
