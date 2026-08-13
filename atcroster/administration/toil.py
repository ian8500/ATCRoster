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


def annotation_accrual_half_days(
    parsed: dict[str, Any] | None,
    *,
    annotation_config: Callable[[str], dict[str, Any] | None],
) -> int:
    if not parsed:
        return 0
    config = annotation_config(parsed.get("type"))
    if not config:
        return 0
    try:
        return int(config.get("toil_half_days", 0) or 0)
    except (TypeError, ValueError):
        return 0


def apply_annotation_toil_delta(
    staff: Any,
    old_annotation: str,
    new_annotation: str,
    *,
    actor_id: int,
    parse_annotation: Callable[[str], dict[str, Any] | None],
    accrual_half_days: Callable[[dict[str, Any] | None], int],
    record_transaction: Callable[..., Any],
    transaction_key: str | None = None,
    source_id: int | None = None,
) -> None:
    old_half_days = accrual_half_days(parse_annotation(old_annotation))
    new_half_days = accrual_half_days(parse_annotation(new_annotation))
    delta = new_half_days - old_half_days
    if delta:
        record_transaction(
            staff.id,
            delta,
            "Roster annotation TOIL adjustment",
            actor_id,
            transaction_key=transaction_key,
            source_type="assignment_annotation",
            source_id=source_id,
        )


def accrued_and_used_half_days(
    staff_id: int,
    start_day: Any,
    end_day: Any,
    *,
    Assignment: Any,
    parse_annotation: Callable[[str], dict[str, Any] | None],
    accrual_half_days: Callable[[dict[str, Any] | None], int],
) -> tuple[int, int]:
    accrued = used = 0
    assignments = Assignment.query.filter(
        Assignment.staff_id == staff_id,
        Assignment.day >= start_day,
        Assignment.day <= end_day,
    ).all()
    for assignment in assignments:
        accrued += accrual_half_days(parse_annotation(assignment.annotation))
        if assignment.effective_code == "TOU8":
            used += 2
        elif assignment.effective_code == "TOUI":
            used += 1
    return accrued, used


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
