"""Shared administration landing route extracted from ``app.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, render_template, request
from flask_login import current_user, login_required

from .actions import AdminActionDependencies, dispatch_admin_action
from .context import AdminContextDependencies, build_admin_context


@dataclass(frozen=True)
class AdministrationDependencies:
    is_admin_user: Callable[[Any], bool]
    live_position_enabled: Callable[[int], bool]


def create_administration_dependencies(
    **services: Any,
) -> AdministrationDependencies:
    """Construct the administration-home dependency contract."""
    return AdministrationDependencies(**services)


@dataclass(frozen=True)
class AdminDashboardDependencies:
    is_admin_user: Callable[[Any], bool]
    actions: AdminActionDependencies
    context: AdminContextDependencies


def create_admin_dashboard_dependencies(
    **services: Any,
) -> AdminDashboardDependencies:
    """Construct the administration-dashboard dependency contract."""
    return AdminDashboardDependencies(**services)


def create_admin_dashboard_blueprint(
    dependencies: AdminDashboardDependencies,
) -> Blueprint:
    """Create the legacy ``/admin`` roster-configuration endpoint."""
    blueprint = Blueprint("admin_dashboard", __name__)

    @login_required
    def admin_dashboard():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if request.method == "POST":
            response = dispatch_admin_action(
                request.form.get("form", ""), request.form, dependencies.actions
            )
            if response is not None:
                return response
        return render_template(
            "admin.html", **build_admin_context(dependencies.context)
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin", "admin", admin_dashboard, methods=("GET", "POST")
        )

    return blueprint


def create_administration_blueprint(
    dependencies: AdministrationDependencies,
) -> Blueprint:
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
        state.app.add_url_rule(
            "/administration",
            "administration_home",
            administration_home,
            methods=("GET",),
        )

    return blueprint
