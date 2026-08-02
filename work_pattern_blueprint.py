"""Unit-administrator UI for flexible work patterns and staff rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from work_pattern_service import PATTERN_DAY_TYPES, STAFF_RULE_TYPES


@dataclass(frozen=True)
class WorkPatternBlueprintDependencies:
    db: Any
    Staff: Any
    ShiftType: Any
    WorkPattern: Any
    WorkPatternDay: Any
    WorkPatternDayAllowedShift: Any
    StaffPatternAssignment: Any
    StaffRule: Any
    is_admin_user: Callable[[Any], bool]
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    pattern_service: Any
    admin_service: Any


def create_work_pattern_blueprint(
    dependencies: WorkPatternBlueprintDependencies,
) -> Blueprint:
    blueprint = Blueprint("work_patterns", __name__)

    def require_admin() -> int:
        if not dependencies.is_admin_user(current_user):
            abort(403)
        return dependencies.current_unit_id()

    def unit_shifts(unit_id: int) -> list[Any]:
        return dependencies.ShiftType.query.filter_by(
            unit_id=unit_id, is_active=True
        ).order_by(dependencies.ShiftType.code).all()

    @blueprint.route("/administration/work-patterns", methods=["GET", "POST"])
    @login_required
    def patterns():
        unit_id = require_admin()
        if request.method == "POST":
            dependencies.validate_csrf()
            action = (request.form.get("action") or "").strip()
            try:
                if action == "seed":
                    created = dependencies.admin_service.seed_standard_patterns(unit_id)
                    dependencies.db.session.commit()
                    flash(
                        f"Added {len(created)} standard pattern(s)."
                        if created else "Standard patterns already exist.",
                        "ok",
                    )
                elif action == "create":
                    name = (request.form.get("name") or "").strip()[:120]
                    if not name:
                        raise ValueError("Pattern name is required.")
                    if dependencies.WorkPattern.query.filter_by(
                        unit_id=unit_id, name=name
                    ).first():
                        raise ValueError("A pattern with this name already exists.")
                    cycle_length = _bounded_int(
                        request.form.get("cycle_length_days"), 1, 366,
                        "Cycle length must be between 1 and 366 days.",
                    )
                    pattern = dependencies.WorkPattern(
                        unit_id=unit_id,
                        name=name,
                        description=(request.form.get("description") or "").strip()[:2000],
                        cycle_length_days=cycle_length,
                        contracted_minutes_per_cycle=_bounded_int(
                            request.form.get("contracted_minutes_per_cycle"),
                            0, 500_000,
                            "Contracted minutes must be zero or greater.",
                        ),
                        is_active=True,
                    )
                    dependencies.db.session.add(pattern)
                    dependencies.db.session.flush()
                    dependencies.admin_service.replace_pattern_days(
                        pattern,
                        [{"day_type": "OFF"} for _ in range(cycle_length)],
                    )
                    dependencies.db.session.commit()
                    flash("Pattern created. Configure each cycle day next.", "ok")
                    return redirect(url_for("work_patterns.pattern_detail", pattern_id=pattern.id))
                else:
                    abort(400, "Unknown pattern action.")
            except ValueError as exc:
                dependencies.db.session.rollback()
                flash(str(exc), "error")
            return redirect(url_for("work_patterns.patterns"))
        rows = dependencies.WorkPattern.query.filter_by(unit_id=unit_id).order_by(
            dependencies.WorkPattern.is_active.desc(), dependencies.WorkPattern.name
        ).all()
        return render_template("work_patterns/index.html", patterns=rows)

    @blueprint.route(
        "/administration/work-patterns/<int:pattern_id>", methods=["GET", "POST"]
    )
    @login_required
    def pattern_detail(pattern_id: int):
        unit_id = require_admin()
        pattern = dependencies.WorkPattern.query.filter_by(
            id=pattern_id, unit_id=unit_id
        ).first_or_404()
        shifts = unit_shifts(unit_id)
        if request.method == "POST":
            dependencies.validate_csrf()
            try:
                action = (request.form.get("action") or "save").strip()
                if action == "toggle_active":
                    pattern.is_active = not pattern.is_active
                    dependencies.db.session.commit()
                    flash(
                        "Pattern is available for new assignments."
                        if pattern.is_active
                        else "Pattern retired from new assignments; history is retained.",
                        "ok",
                    )
                    return redirect(url_for(
                        "work_patterns.pattern_detail", pattern_id=pattern.id
                    ))
                is_in_use = dependencies.StaffPatternAssignment.query.filter_by(
                    unit_id=unit_id, work_pattern_id=pattern.id
                ).first() is not None
                if is_in_use:
                    raise ValueError(
                        "Assigned patterns are locked to preserve roster history. "
                        "Create a replacement pattern for structural changes."
                    )
                cycle_length = _bounded_int(
                    request.form.get("cycle_length_days"), 1, 366,
                    "Cycle length must be between 1 and 366 days.",
                )
                name = (request.form.get("name") or "").strip()[:120]
                duplicate = dependencies.WorkPattern.query.filter(
                    dependencies.WorkPattern.unit_id == unit_id,
                    dependencies.WorkPattern.name == name,
                    dependencies.WorkPattern.id != pattern.id,
                ).first()
                if not name or duplicate:
                    raise ValueError(
                        "Pattern name is required and must be unique in this airport."
                    )
                pattern.name = name
                pattern.description = (
                    request.form.get("description") or ""
                ).strip()[:2000]
                pattern.contracted_minutes_per_cycle = _bounded_int(
                    request.form.get("contracted_minutes_per_cycle"),
                    0, 500_000,
                    "Contracted minutes must be zero or greater.",
                )
                pattern.is_active = request.form.get("is_active") == "on"
                specs = [
                    _day_spec(request, index, shifts)
                    for index in range(cycle_length)
                ]
                dependencies.admin_service.replace_pattern_days(pattern, specs)
                dependencies.db.session.commit()
                flash("Pattern saved.", "ok")
            except (TypeError, ValueError) as exc:
                dependencies.db.session.rollback()
                flash(str(exc), "error")
            return redirect(url_for("work_patterns.pattern_detail", pattern_id=pattern.id))
        days = dependencies.WorkPatternDay.query.filter_by(
            unit_id=unit_id, work_pattern_id=pattern.id
        ).order_by(dependencies.WorkPatternDay.day_index).all()
        allowed = {
            day.id: {
                row.shift_type_id
                for row in dependencies.WorkPatternDayAllowedShift.query.filter_by(
                    unit_id=unit_id, work_pattern_day_id=day.id
                ).all()
            }
            for day in days
        }
        preview_start = _optional_date(request.args.get("preview_start")) or date.today()
        preview = _pattern_preview(pattern, days, shifts, allowed, preview_start, 28)
        is_in_use = dependencies.StaffPatternAssignment.query.filter_by(
            unit_id=unit_id, work_pattern_id=pattern.id
        ).first() is not None
        return render_template(
            "work_patterns/detail.html", pattern=pattern, days=days,
            shifts=shifts, allowed=allowed, day_types=sorted(PATTERN_DAY_TYPES),
            preview=preview, preview_start=preview_start, is_in_use=is_in_use,
        )

    @blueprint.route(
        "/administration/staff/<int:staff_id>/work-rules", methods=["GET", "POST"]
    )
    @login_required
    def staff_rules(staff_id: int):
        unit_id = require_admin()
        staff = dependencies.Staff.query.filter_by(
            id=staff_id, unit_id=unit_id
        ).first_or_404()
        if request.method == "POST":
            dependencies.validate_csrf()
            action = (request.form.get("action") or "").strip()
            try:
                if action == "assign_pattern":
                    assignment = dependencies.StaffPatternAssignment(
                        unit_id=unit_id,
                        staff_id=staff.id,
                        work_pattern_id=int(request.form.get("work_pattern_id") or 0),
                        effective_from=date.fromisoformat(request.form["effective_from"]),
                        effective_to=_optional_date(request.form.get("effective_to")),
                        anchor_date=date.fromisoformat(request.form["anchor_date"]),
                        anchor_day_index=int(request.form.get("anchor_day_index") or 0),
                        contracted_minutes_override=_optional_int(
                            request.form.get("contracted_minutes_override")
                        ),
                        notes=(request.form.get("notes") or "").strip()[:500],
                    )
                    dependencies.pattern_service.validate_staff_pattern_assignment(assignment)
                    dependencies.db.session.add(assignment)
                    flash("Effective-dated pattern assignment added.", "ok")
                elif action == "end_assignment":
                    assignment = dependencies.StaffPatternAssignment.query.filter_by(
                        id=int(request.form.get("assignment_id") or 0),
                        unit_id=unit_id, staff_id=staff.id,
                    ).first_or_404()
                    end_date = date.fromisoformat(request.form["effective_to"])
                    if end_date < assignment.effective_from:
                        raise ValueError("Pattern end date cannot precede its start date.")
                    assignment.effective_to = end_date
                    dependencies.pattern_service.validate_staff_pattern_assignment(assignment)
                    flash("Pattern assignment end date saved.", "ok")
                elif action == "add_rule":
                    rule = _build_rule(dependencies, unit_id, staff.id, request)
                    dependencies.pattern_service.validate_staff_rule(rule)
                    dependencies.db.session.add(rule)
                    flash("Staff rule added.", "ok")
                elif action == "deactivate_rule":
                    rule = dependencies.StaffRule.query.filter_by(
                        id=int(request.form.get("rule_id") or 0),
                        unit_id=unit_id, staff_id=staff.id,
                    ).first_or_404()
                    rule.is_active = False
                    flash("Staff rule deactivated; its history is retained.", "ok")
                else:
                    abort(400, "Unknown staff-pattern action.")
                dependencies.db.session.commit()
            except (KeyError, TypeError, ValueError) as exc:
                dependencies.db.session.rollback()
                flash(f"Could not save: {exc}", "error")
            return redirect(url_for("work_patterns.staff_rules", staff_id=staff.id))
        patterns = dependencies.WorkPattern.query.filter_by(
            unit_id=unit_id, is_active=True
        ).order_by(dependencies.WorkPattern.name).all()
        pattern_names = {
            row.id: row.name
            for row in dependencies.WorkPattern.query.filter_by(unit_id=unit_id).all()
        }
        assignments = dependencies.StaffPatternAssignment.query.filter_by(
            unit_id=unit_id, staff_id=staff.id
        ).order_by(dependencies.StaffPatternAssignment.effective_from.desc()).all()
        rules = dependencies.StaffRule.query.filter_by(
            unit_id=unit_id, staff_id=staff.id
        ).order_by(dependencies.StaffRule.is_active.desc(), dependencies.StaffRule.effective_from.desc()).all()
        preview_start = _optional_date(request.args.get("preview_start")) or date.today()
        preview = []
        for offset in range(28):
            day = preview_start + timedelta(days=offset)
            resolution = dependencies.pattern_service.get_pattern_day_for_staff(staff.id, day)
            preview.append((day, resolution))
        return render_template(
            "work_patterns/staff_rules.html", staff=staff, patterns=patterns,
            assignments=assignments, rules=rules, shifts=unit_shifts(unit_id),
            rule_types=sorted(STAFF_RULE_TYPES), preview=preview,
            preview_start=preview_start,
            pattern_names=pattern_names,
        )

    return blueprint


def _bounded_int(raw: str | None, minimum: int, maximum: int, message: str) -> int:
    try:
        value = int(raw or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not minimum <= value <= maximum:
        raise ValueError(message)
    return value


def _optional_int(raw: str | None) -> int | None:
    return int(raw) if raw not in {None, ""} else None


def _optional_date(raw: str | None) -> date | None:
    return date.fromisoformat(raw) if raw else None


def _day_spec(req: Any, index: int, shifts: list[Any]) -> dict[str, Any]:
    day_type = (req.form.get(f"day_type_{index}") or "").strip().upper()
    if day_type not in PATTERN_DAY_TYPES:
        raise ValueError(f"Cycle day {index + 1} has an invalid day type.")
    shift_ids = {shift.id for shift in shifts}
    fixed = _optional_int(req.form.get(f"fixed_shift_type_id_{index}"))
    if fixed is not None and fixed not in shift_ids:
        raise ValueError(f"Cycle day {index + 1} references an unavailable shift.")
    allowed = tuple(
        int(value) for value in req.form.getlist(f"allowed_shift_type_ids_{index}")
        if int(value) in shift_ids
    )
    if day_type == "WORK_ALLOWED_SET" and not allowed:
        raise ValueError(f"Cycle day {index + 1} must allow at least one shift.")
    return {
        "day_type": day_type,
        "fixed_shift_type_id": fixed if day_type == "FIXED_SHIFT" else None,
        "allowed_shift_type_ids": allowed if day_type == "WORK_ALLOWED_SET" else (),
        "required_work": req.form.get(f"required_work_{index}") == "on",
        "notes": req.form.get(f"notes_{index}") or "",
    }


def _build_rule(dependencies: Any, unit_id: int, staff_id: int, req: Any) -> Any:
    shift_type_id = _optional_int(req.form.get("shift_type_id"))
    if shift_type_id and not dependencies.ShiftType.query.filter_by(
        id=shift_type_id, unit_id=unit_id
    ).first():
        raise ValueError("Selected shift is unavailable in this airport.")
    weekdays_mask = sum(
        1 << weekday for weekday in range(7)
        if req.form.get(f"weekday_{weekday}") == "on"
    ) or None
    return dependencies.StaffRule(
        unit_id=unit_id,
        staff_id=staff_id,
        rule_type=(req.form.get("rule_type") or "").strip().upper(),
        hardness=(req.form.get("hardness") or "HARD").strip().upper(),
        effective_from=date.fromisoformat(req.form["effective_from"]),
        effective_to=_optional_date(req.form.get("effective_to")),
        shift_type_id=shift_type_id,
        shift_group=(req.form.get("shift_group") or "").strip().upper() or None,
        maximum_count=_optional_int(req.form.get("maximum_count")),
        rolling_period_days=_optional_int(req.form.get("rolling_period_days")),
        weekdays_mask=weekdays_mask,
        penalty_weight=int(req.form.get("penalty_weight") or 1),
        reason=(req.form.get("reason") or "").strip()[:500],
        authorised_by_user_id=getattr(current_user, "person_id", None),
        is_active=True,
    )


def _pattern_preview(
    pattern: Any, days: list[Any], shifts: list[Any], allowed: dict[int, set[int]],
    start: date, count: int,
) -> list[tuple[date, str]]:
    by_index = {day.day_index: day for day in days}
    shift_by_id = {shift.id: shift.code for shift in shifts}
    preview = []
    for offset in range(count):
        day = by_index.get(offset % pattern.cycle_length_days)
        label = "Not configured"
        if day:
            label = day.day_type.replace("_", " ").title()
            if day.fixed_shift_type_id:
                label = shift_by_id.get(day.fixed_shift_type_id, label)
            elif day.day_type == "WORK_ALLOWED_SET":
                label = " / ".join(
                    shift_by_id[item] for item in sorted(allowed.get(day.id, set()))
                    if item in shift_by_id
                )
        preview.append((start + timedelta(days=offset), label))
    return preview
