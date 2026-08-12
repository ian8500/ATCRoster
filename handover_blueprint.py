"""Tenant-scoped operational handovers between watch managers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import re
import time
from typing import Any, Callable
from urllib import parse as urllib_parse, request as urllib_request
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
    HandoverOperationalState: Any
    HandoverEquipment: Any
    OperationalPosition: Any
    PositionSession: Any
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    is_admin_user: Callable[[Any], bool]
    is_editor_user: Callable[[Any], bool]
    requirements_for_day: Callable[..., dict[str, int]]
    shift_group_for_day: Callable[[str, int, date], str | None]
    utcnow: Callable[[], datetime]
    live_position_enabled: Callable[[int], bool]


def create_handover_blueprint(deps: HandoverDependencies) -> Blueprint:
    bp = Blueprint("handover", __name__, url_prefix="/handover")
    metar_cache: dict[str, tuple[float, dict[str, str]]] = {}

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

    def operational_state(unit_id: int):
        state = deps.HandoverOperationalState.query.filter_by(unit_id=unit_id).first()
        if state is None:
            state = deps.HandoverOperationalState(unit_id=unit_id)
            deps.db.session.add(state)
            deps.db.session.flush()
        return state

    def runway_options(state: Any) -> list[str]:
        try:
            options = json.loads(state.runway_options_json or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            options = []
        if not isinstance(options, list):
            return []
        return [str(option).strip()[:40] for option in options if str(option).strip()]

    def equipment_rows(unit_id: int, active_only: bool = True) -> list[Any]:
        query = deps.HandoverEquipment.query.filter_by(unit_id=unit_id)
        if active_only:
            query = query.filter(deps.HandoverEquipment.active.is_(True))
        return query.order_by(
            deps.HandoverEquipment.display_order, deps.HandoverEquipment.id
        ).all()

    def current_metar(icao: str) -> dict[str, str]:
        code = (icao or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{4}", code):
            return {"available": "false", "raw": "", "observed": ""}
        cached = metar_cache.get(code)
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        result = {"available": "false", "raw": "", "observed": ""}
        try:
            url = "https://aviationweather.gov/api/data/metar?" + urllib_parse.urlencode({"ids": code, "format": "json"})
            req = urllib_request.Request(url, headers={"User-Agent": "ATCRoster/1.0 operational-handover"})
            # The endpoint is a fixed HTTPS government-weather origin and the
            # ICAO value is constrained above; this is not user-controlled URL IO.
            with urllib_request.urlopen(req, timeout=4) as response:  # nosec B310
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, list) and payload:
                item = payload[0]
                result = {
                    "available": "true",
                    "raw": str(item.get("rawOb") or item.get("raw_text") or ""),
                    "observed": str(item.get("reportTime") or item.get("obsTime") or ""),
                }
        except Exception:
            pass
        metar_cache[code] = (time.monotonic(), result)
        return result

    def live_positions(unit_id: int) -> list[dict[str, str]]:
        if not deps.live_position_enabled(unit_id):
            return []
        rows = deps.db.session.query(
            deps.OperationalPosition.code,
            deps.OperationalPosition.label,
            deps.Staff.name,
        ).join(
            deps.PositionSession,
            deps.PositionSession.position_id == deps.OperationalPosition.id,
        ).join(
            deps.Staff, deps.Staff.id == deps.PositionSession.primary_person_id,
        ).filter(
            deps.PositionSession.unit_id == unit_id,
            deps.PositionSession.ended_at.is_(None),
            deps.PositionSession.is_void.is_(False),
        ).order_by(deps.OperationalPosition.code).all()
        return [{"code": code, "label": label, "name": name} for code, label, name in rows]

    def persistent_values(unit_id: int) -> dict[int, str]:
        latest = deps.HandoverRecord.query.filter_by(
            unit_id=unit_id, status="published"
        ).order_by(deps.HandoverRecord.created_at.desc()).first()
        if not latest:
            return {}
        decode_record(latest)
        return {int(item.get("field_id") or 0): str(item.get("value") or "") for item in latest.responses}

    def page_context(unit_id: int) -> dict[str, Any]:
        state = operational_state(unit_id)
        return {
            "operational_state": state,
            "runway_options": runway_options(state),
            "metar": current_metar(state.metar_icao),
            "equipment": equipment_rows(unit_id),
            "live_positions": live_positions(unit_id),
            "live_position_active": deps.live_position_enabled(unit_id),
        }

    @bp.before_request
    @login_required
    def protect_module():
        require_module()

    @bp.get("/")
    def home():
        if can_write():
            return redirect(url_for("handover.edit"))
        unit_id = deps.current_unit_id()
        records = deps.HandoverRecord.query.filter_by(
            unit_id=unit_id, status="published",
        ).order_by(deps.HandoverRecord.created_at.desc()).limit(20).all()
        for record in records:
            decode_record(record)
        latest = records[0] if records else None
        context = page_context(unit_id)
        deps.db.session.commit()
        return render_template(
            "handover/home.html", latest=latest, history=records[1:],
            next_shift=next_shift(unit_id, deps.db.session.get(deps.Unit, unit_id)),
            can_write_handover=can_write(),
            **context,
        )

    @bp.route("/edit", methods=["GET", "POST"])
    @bp.route("/new", methods=["GET", "POST"])
    def edit():
        """Edit the unit's single current handover record.

        ``/new`` remains as a compatibility URL for existing bookmarks, but
        does not create a succession of published handovers.
        """
        require_writer()
        unit_id = deps.current_unit_id()
        unit = deps.db.session.get(deps.Unit, unit_id)
        fields = active_fields(unit_id)
        shift = next_shift(unit_id, unit)
        retained_values = persistent_values(unit_id)
        context = page_context(unit_id)
        if request.method == "POST":
            deps.validate_csrf()
            state = context["operational_state"]
            state.runway_in_use = (request.form.get("runway_in_use") or "").strip()[:40]
            configured_runways = context["runway_options"]
            if state.runway_in_use and configured_runways and state.runway_in_use not in configured_runways:
                abort(400, "Select a configured runway option.")
            state.updated_by_id = current_user.id
            state.updated_by_name = current_user.name
            for equipment in context["equipment"]:
                status = (request.form.get(f"equipment_status_{equipment.id}") or equipment.status).strip().lower()
                if status not in {"green", "amber", "red"}:
                    abort(400, "Invalid equipment status.")
                equipment.status = status
                equipment.note = (request.form.get(f"equipment_note_{equipment.id}") or "").strip()[:240]
                equipment.updated_by_id = current_user.id
                equipment.updated_by_name = current_user.name
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
                record = deps.HandoverRecord.query.filter_by(
                    unit_id=unit_id, status="published"
                ).order_by(deps.HandoverRecord.created_at.desc()).first()
                if record is None:
                    record = deps.HandoverRecord(
                        unit_id=unit_id,
                        status="published",
                        created_by_id=current_user.id,
                        created_by_name=current_user.name,
                    )
                    deps.db.session.add(record)
                record.target_shift_day = (
                    date.fromisoformat(shift["day"]) if shift.get("day") else None
                )
                record.target_shift_code = shift.get("code", "")
                record.target_shift_name = shift.get("name", "")
                record.target_shift_start = start.replace(tzinfo=None) if start else None
                record.next_shift_json = json.dumps(shift, separators=(",", ":"))
                record.responses_json = json.dumps(responses, separators=(",", ":"))
                deps.db.session.commit()
                flash("Current handover updated.", "ok")
                return redirect(url_for("handover.edit"))
        return render_template(
            "handover/create.html", fields=fields, next_shift=shift,
            retained_values=retained_values, **context,
        )

    @bp.get("/<int:record_id>")
    def view(record_id: int):
        unit_id = deps.current_unit_id()
        record = deps.HandoverRecord.query.filter_by(
            id=record_id, unit_id=unit_id, status="published",
        ).first_or_404()
        decode_record(record)
        context = page_context(unit_id)
        deps.db.session.commit()
        return render_template("handover/view.html", handover=record, **context)

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
            elif action == "save_operational":
                state = operational_state(unit_id)
                icao = (request.form.get("metar_icao") or "").strip().upper()
                if icao and not re.fullmatch(r"[A-Z]{4}", icao):
                    abort(400, "Enter a four-letter ICAO code.")
                state.metar_icao = icao
                runways = []
                for line in (request.form.get("runway_options") or "").splitlines():
                    runway = line.strip()[:40]
                    if runway and runway not in runways:
                        runways.append(runway)
                state.runway_options_json = json.dumps(runways)
                state.updated_by_id = current_user.id
                state.updated_by_name = current_user.name
                flash("Operational header settings saved.", "ok")
            elif action == "add_equipment":
                name = (request.form.get("equipment_name") or "").strip()[:120]
                if not name:
                    abort(400, "Enter an equipment name.")
                highest = deps.db.session.query(deps.db.func.max(deps.HandoverEquipment.display_order)).filter_by(unit_id=unit_id).scalar() or 0
                deps.db.session.add(deps.HandoverEquipment(
                    unit_id=unit_id, name=name, display_order=highest + 10,
                    updated_by_id=current_user.id, updated_by_name=current_user.name,
                ))
                flash("Equipment added.", "ok")
            else:
                try:
                    field_id = int(request.form.get("field_id") or 0)
                except ValueError:
                    abort(400)
                if action == "toggle_equipment":
                    equipment = deps.HandoverEquipment.query.filter_by(id=field_id, unit_id=unit_id).first_or_404()
                    equipment.active = not equipment.active
                    flash("Equipment list updated.", "ok")
                    deps.db.session.commit()
                    return redirect(url_for("handover.settings"))
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
        state = operational_state(unit_id)
        equipment = equipment_rows(unit_id, active_only=False)
        deps.db.session.commit()
        return render_template(
            "handover/settings.html",
            fields=fields,
            operational_state=state,
            runway_options=runway_options(state),
            equipment=equipment,
        )

    bp.handover_enabled = enabled
    return bp
