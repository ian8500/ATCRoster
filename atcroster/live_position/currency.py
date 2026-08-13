"""Operational-currency administration route."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class OperationalCurrencyDependencies:
    db: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    live_position_enabled: Callable[[int], bool]
    currency_requirement: Callable[[int], dict[str, Any]]
    save_currency_requirement: Callable[[dict[str, Any]], None]
    currency_shortfalls: Callable[[int], dict[str, Any]]
    validate_csrf: Callable[[], None]


def create_operational_currency_blueprint(
    dependencies: OperationalCurrencyDependencies,
) -> Blueprint:
    blueprint = Blueprint("operational_currency", __name__)

    @login_required
    def admin_operational_currency():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        unit_id = dependencies.current_unit_id()
        if not dependencies.live_position_enabled(unit_id):
            abort(404)
        requirement = dependencies.currency_requirement(unit_id)
        if request.method == "POST":
            dependencies.validate_csrf()
            try:
                requirement = {
                    "enabled": request.form.get("enabled") == "on",
                    "period_type": request.form.get("period_type"),
                    "period_days": int(request.form.get("period_days") or requirement["period_days"]),
                    "period_months": int(request.form.get("period_months") or requirement["period_months"]),
                    "start_date": request.form.get("start_date") or "",
                    "hours_per_ue": float(request.form.get("hours_per_ue") or 0),
                    "ojti_credit_percent": float(request.form.get("ojti_credit_percent") or 0),
                }
                if requirement["period_type"] not in {"rolling_days", "calendar_months"}:
                    raise ValueError
                if not (1 <= requirement["period_days"] <= 731 and 1 <= requirement["period_months"] <= 24):
                    raise ValueError
                if not (0.25 <= requirement["hours_per_ue"] <= 1000 and 0 <= requirement["ojti_credit_percent"] <= 100):
                    raise ValueError
                if requirement["start_date"]:
                    date.fromisoformat(requirement["start_date"])
            except ValueError:
                flash("Enter valid currency-period and operational-time values.", "error")
            else:
                dependencies.save_currency_requirement(requirement)
                dependencies.db.session.commit()
                flash("Operational currency requirement saved.", "ok")
                return redirect(url_for("admin_operational_currency"))
        return render_template(
            "admin_operational_currency.html",
            requirement=requirement,
            currency_preview=dependencies.currency_shortfalls(unit_id),
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin/operational-currency",
            "admin_operational_currency",
            admin_operational_currency,
            methods=("GET", "POST"),
        )

    return blueprint
