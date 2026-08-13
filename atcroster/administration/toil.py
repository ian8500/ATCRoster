"""Manual TOIL adjustment route."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


def seed_toil_balances(raw_lines: str, *, db: Any, Staff: Any) -> tuple[int, int]:
    """Import legacy TOIL balances expressed as days or hours."""
    updated = errors = 0
    for line in raw_lines.strip().splitlines():
        if not line.strip():
            continue
        try:
            staff_no, raw_value = [value.strip() for value in line.split(",", 1)]
            staff = Staff.query.filter_by(staff_no=staff_no).first()
            if not staff:
                errors += 1
                continue
            value = (
                raw_value.lower()
                .replace("days", "d")
                .replace("day", "d")
                .replace("hrs", "h")
                .replace("hr", "h")
                .replace("hours", "h")
                .replace("hour", "h")
            )
            if value.endswith("d"):
                half_days = int(round(float(value[:-1]) * 2))
            elif value.endswith("h"):
                half_days = int(round((float(value[:-1]) / 8.0) * 2))
            else:
                half_days = int(round(float(value) * 2))
            staff.toil_half_days = half_days
            updated += 1
        except (TypeError, ValueError):
            errors += 1
    db.session.commit()
    return updated, errors


@dataclass(frozen=True)
class ToilAdministrationDependencies:
    db: Any
    Staff: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    record_toil_transaction: Callable[..., None]


def create_toil_administration_blueprint(
    dependencies: ToilAdministrationDependencies,
) -> Blueprint:
    blueprint = Blueprint("toil_administration", __name__)

    @login_required
    def admin_toil_new():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        atcos = (
            dependencies.Staff.query.filter_by(is_operational=True)
            .filter(dependencies.Staff.role != "position_monitor")
            .order_by(dependencies.Staff.name.asc())
            .all()
        )
        if request.method == "POST":
            dependencies.validate_csrf()
            try:
                sid = int(request.form["staff_id"])
                amount = float(request.form.get("amount", "0") or 0)
            except (KeyError, TypeError, ValueError):
                flash("Choose an ATCO and enter a valid adjustment.", "error")
                return redirect(url_for("admin_toil_new"))
            unit = request.form.get("unit", "days").lower()
            note = (request.form.get("note") or "").strip()
            staff = dependencies.Staff.query.filter_by(
                id=sid, unit_id=dependencies.current_unit_id()
            ).first_or_404()
            direction = -1 if request.form.get("direction") == "subtract" else 1
            half_days = (
                int(round(amount * 2))
                if unit.startswith("day")
                else int(round((amount / 8.0) * 2))
            )
            if amount <= 0 or half_days <= 0:
                flash("Enter an adjustment greater than zero.", "error")
                return redirect(url_for("admin_toil_new"))
            try:
                dependencies.record_toil_transaction(
                    staff.id,
                    direction * half_days,
                    note,
                    current_user.id,
                    transaction_key=request.form.get("transaction_key"),
                    source_type="manual_admin",
                )
            except ValueError as error:
                flash(str(error), "error")
                return redirect(url_for("admin_toil_new"))
            dependencies.db.session.commit()
            verb = "added to" if direction > 0 else "deducted from"
            flash(f"{amount:g} {unit} {verb} {staff.name}'s TOIL balance.", "ok")
            return redirect(url_for("admin_toil_new"))
        selected_staff_id = request.args.get("staff_id", type=int)
        if selected_staff_id and selected_staff_id not in {staff.id for staff in atcos}:
            abort(404)
        return render_template(
            "admin_toil_new.html",
            atcos=atcos,
            selected_staff_id=selected_staff_id,
            transaction_key=secrets.token_hex(24),
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin/toil/new",
            "admin_toil_new",
            admin_toil_new,
            methods=("GET", "POST"),
        )

    return blueprint
