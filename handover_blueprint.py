"""Tenant-scoped operational handovers between watch managers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


@dataclass(frozen=True)
class HandoverDependencies:
    db: Any
    Unit: Any
    Staff: Any
    ShiftType: Any
    Assignment: Any
    Requirement: Any
    SpecialRequirement: Any
    FeatureFlag: Any
    HandoverField: Any
    HandoverRecord: Any
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    is_admin_user: Callable[[Any], bool]
    is_editor_user: Callable[[Any], bool]
    requirements_for_day: Callable[..., dict[str, int]]
    shift_group_for_day: Callable[[str, int, date], str | None]
    utcnow: Callable[[], datetime]


def create_handover_blueprint(deps: HandoverDependencies) -> Blueprint:
    bp = Blueprint("handover", __name__, url_prefix="/handover")

    def enabled(unit_id: int | None = None) -> bool:
        resolved = int(unit_id or deps.current_unit_id() or 0)
        return bool(resolved and deps.FeatureFlag.query.filter_by(
            unit_id=resolved, key="handover_module", enabled=True,
        ).first())

    def require_module() -> None:
        if not enabled():
            abort(404)

    def can_write() -> bool:
        return bool(
            deps.is_admin_user(current_user)
            or deps.is_editor_user(current_user)
            or getattr(current_user, "is_wm", False)
            or getattr(current_user, "is_dwm", False)
        )

    def require_writer() -> None:
        require_module()
        if not can_write():
            abort(403)

    def require_admin() -> None:
        require_module()
        if not deps.is_admin_user(current_user):
            abort(403)

    def local_now(unit) -> datetime:
        try:
            zone = ZoneInfo(unit.timezone or "Europe/London")
        except ZoneInfoNotFoundError:
            zone = ZoneInfo("Europe/London")
        return deps.utcnow().astimezone(zone)

    def field_options(field) -> list[str]:
        try:
            options = json.loads(field.options_json or "[]")
        except (TypeError, ValueError):
            options = []
        return [str(value) for value in options if str(value).strip()]

    def active_fields(unit_id: int) -> list[Any]:
        fields = deps.HandoverField.query.filter_by(
            unit_id=unit_id, active=True,
        ).order_by(
            deps.HandoverField.display_order,
            deps.HandoverField.id,
        ).all()
        for field in fields:
            field.select_options = field_options(field)
        return fields

    def next_shift(unit_id: int, unit) -> dict[str, Any]:
        now = local_now(unit)
        end_day = now.date() + timedelta(days=7)
        shifts = {
            shift.code: shift for shift in deps.ShiftType.query.filter(
                deps.ShiftType.unit_id == unit_id,
                deps.ShiftType.is_working.is_(True),
                deps.ShiftType.is_active.is_(True),
                deps.ShiftType.start_time.is_not(None),
            ).all()
        }
        assignments = deps.Assignment.query.join(
            deps.Staff, deps.Staff.id == deps.Assignment.staff_id,
        ).filter(
            deps.Assignment.unit_id == unit_id,
            deps.Assignment.day.between(now.date(), end_day),
            deps.Assignment.effective_code.in_(tuple(shifts) or ("__none__",)),
            deps.Staff.membership_status == "active",
            deps.Staff.role != "position_monitor",
        ).order_by(deps.Assignment.day, deps.Assignment.code, deps.Staff.name).all()
        occurrences: dict[tuple[date, str], list[Any]] = {}
        for assignment in assignments:
            effective_code = assignment.effective_code
            shift = shifts.get(effective_code)
            if not shift:
                continue
            starts_at = datetime.combine(assignment.day, shift.start_time, tzinfo=now.tzinfo)
            if starts_at <= now:
                continue
            occurrences.setdefault((assignment.day, effective_code), []).append(assignment)
        if not occurrences:
            return {"available": False, "people": [], "staffing": 0, "required": None}
        day, code = min(
            occurrences,
            key=lambda item: datetime.combine(item[0], shifts[item[1]].start_time, tzinfo=now.tzinfo),
        )
        shift = shifts[code]
        rows = occurrences[(day, code)]
        people = [{"id": row.staff.id, "name": row.staff.name, "staff_no": row.staff.staff_no} for row in rows]
        requirement = deps.Requirement.query.filter_by(
            unit_id=unit_id, year=day.year, month=day.month,
        ).first()
        special = deps.SpecialRequirement.query.filter_by(unit_id=unit_id, day=day).first()
        required = None
        group = deps.shift_group_for_day(code, day, unit_id)
        if requirement and group:
            required = deps.requirements_for_day(requirement, day, special).get(group)
        starts_at = datetime.combine(day, shift.start_time, tzinfo=now.tzinfo)
        return {
            "available": True,
            "day": day.isoformat(),
            "day_label": day.strftime("%A %d %B %Y"),
            "code": code,
            "name": shift.name or code,
            "start": starts_at.isoformat(),
            "start_label": starts_at.strftime("%H:%M"),
            "people": people,
            "staffing": len(people),
            "required": required,
        }

    def decode_record(record) -> None:
        try:
            record.next_shift = json.loads(record.next_shift_json or "{}")
        except (TypeError, ValueError):
            record.next_shift = {}
        try:
            record.responses = json.loads(record.responses_json or "[]")
        except (TypeError, ValueError):
            record.responses = []

    @bp.before_request
    @login_required
    def protect_module():
        require_module()

    @bp.get("/")
    def home():
        unit_id = deps.current_unit_id()
        records = deps.HandoverRecord.query.filter_by(
            unit_id=unit_id, status="published",
        ).order_by(deps.HandoverRecord.created_at.desc()).limit(20).all()
        for record in records:
            decode_record(record)
        latest = records[0] if records else None
        return render_template(
            "handover/home.html", latest=latest, history=records[1:],
            next_shift=next_shift(unit_id, deps.db.session.get(deps.Unit, unit_id)),
            can_write_handover=can_write(),
        )

    @bp.route("/new", methods=["GET", "POST"])
    def create():
        require_writer()
        unit_id = deps.current_unit_id()
        unit = deps.db.session.get(deps.Unit, unit_id)
        fields = active_fields(unit_id)
        shift = next_shift(unit_id, unit)
        if request.method == "POST":
            deps.validate_csrf()
            responses = []
            errors = []
            for field in fields:
                value = (request.form.get(f"field_{field.id}") or "").strip()
                options = field_options(field)
                if field.required and not value:
                    errors.append(f"{field.label} is required.")
                if field.field_type == "select" and value and value not in options:
                    abort(400, "Invalid handover field selection.")
                responses.append({
                    "field_id": field.id,
                    "section": field.section_name,
                    "label": field.label,
                    "field_type": field.field_type,
                    "value": value,
                })
            if errors:
                for message in errors:
                    flash(message, "error")
            else:
                start = datetime.fromisoformat(shift["start"]) if shift.get("start") else None
                record = deps.HandoverRecord(
                    unit_id=unit_id,
                    status="published",
                    created_by_id=current_user.id,
                    created_by_name=current_user.name,
                    target_shift_day=date.fromisoformat(shift["day"]) if shift.get("day") else None,
                    target_shift_code=shift.get("code", ""),
                    target_shift_name=shift.get("name", ""),
                    target_shift_start=start.replace(tzinfo=None) if start else None,
                    next_shift_json=json.dumps(shift, separators=(",", ":")),
                    responses_json=json.dumps(responses, separators=(",", ":")),
                )
                deps.db.session.add(record)
                deps.db.session.commit()
                flash("Watch handover published.", "ok")
                return redirect(url_for("handover.view", record_id=record.id))
        return render_template("handover/create.html", fields=fields, next_shift=shift)

    @bp.get("/<int:record_id>")
    def view(record_id: int):
        record = deps.HandoverRecord.query.filter_by(
            id=record_id, unit_id=deps.current_unit_id(), status="published",
        ).first_or_404()
        decode_record(record)
        return render_template("handover/view.html", handover=record)

    @bp.route("/settings", methods=["GET", "POST"])
    def settings():
        require_admin()
        unit_id = deps.current_unit_id()
        if request.method == "POST":
            deps.validate_csrf()
            action = (request.form.get("action") or "add").strip()
            if action == "add":
                label = (request.form.get("label") or "").strip()[:120]
                field_type = (request.form.get("field_type") or "text").strip()
                if not label or field_type not in {"text", "select"}:
                    abort(400, "Enter a label and valid field type.")
                options = [line.strip() for line in (request.form.get("options") or "").splitlines() if line.strip()]
                if field_type == "select" and not options:
                    abort(400, "Add at least one dropdown option.")
                highest = deps.db.session.query(deps.db.func.max(deps.HandoverField.display_order)).filter_by(unit_id=unit_id).scalar() or 0
                deps.db.session.add(deps.HandoverField(
                    unit_id=unit_id,
                    section_name=(request.form.get("section_name") or "Operational overview").strip()[:80],
                    label=label,
                    field_type=field_type,
                    options_json=json.dumps(options),
                    help_text=(request.form.get("help_text") or "").strip()[:240],
                    placeholder=(request.form.get("placeholder") or "").strip()[:160],
                    required=bool(request.form.get("required")),
                    display_order=highest + 10,
                ))
                flash("Handover field added.", "ok")
            else:
                try:
                    field_id = int(request.form.get("field_id") or 0)
                except ValueError:
                    abort(400)
                field = deps.HandoverField.query.filter_by(id=field_id, unit_id=unit_id).first_or_404()
                if action == "toggle":
                    field.active = not field.active
                elif action in {"up", "down"}:
                    field.display_order += -15 if action == "up" else 15
                else:
                    abort(400)
                flash("Handover layout updated.", "ok")
            deps.db.session.commit()
            return redirect(url_for("handover.settings"))
        fields = deps.HandoverField.query.filter_by(unit_id=unit_id).order_by(
            deps.HandoverField.display_order, deps.HandoverField.id,
        ).all()
        for field in fields:
            field.select_options = field_options(field)
        return render_template("handover/settings.html", fields=fields)

    bp.handover_enabled = enabled
    return bp
