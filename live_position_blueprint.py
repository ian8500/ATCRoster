"""HTTP boundary for the Live Position Monitoring module."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone, tzinfo
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import (
    Blueprint,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from flask_login import current_user, login_required

from live_position_service import (
    LivePositionConflict,
    LivePositionModels,
    LivePositionService,
    LivePositionValidationError,
)


@dataclass(frozen=True)
class LivePositionDependencies:
    db: Any
    Unit: Any
    OperationalPosition: Any
    OperationalPositionTimeAllowance: Any
    OperationalPositionGroup: Any
    PositionCurrencyCategory: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    PositionParticipantRole: Any
    PositionSessionAudit: Any
    PositionEndorsement: Any
    Staff: Any
    Watch: Any
    utcnow: Callable[[], Any]
    is_admin_user: Callable[[Any], bool]
    live_position_enabled: Callable[[int], bool]
    competency_enabled: Callable[[int], bool]
    authenticated_database_route_optional: Callable[[], Any]
    authenticated_unit_context: Callable[[int, str | None], Any]


def _minutes_between(start: datetime, end: datetime) -> int:
    return max(0, round((end - start).total_seconds() / 60))


def _overlap_minutes(
    start: datetime, end: datetime, intervals: list[tuple[datetime, datetime]]
) -> int:
    clipped = sorted(
        (max(start, item_start), min(end, item_end))
        for item_start, item_end in intervals
        if item_start < end and item_end > start
    )
    if not clipped:
        return 0
    merged: list[list[datetime]] = []
    for item_start, item_end in clipped:
        if not merged or item_start > merged[-1][1]:
            merged.append([item_start, item_end])
        else:
            merged[-1][1] = max(merged[-1][1], item_end)
    return sum(
        _minutes_between(item_start, item_end) for item_start, item_end in merged
    )


def create_live_position_blueprint(
    dependencies: LivePositionDependencies,
) -> Blueprint:
    blueprint = Blueprint("live_position", __name__, url_prefix="/live-positions")

    def _unit_id() -> int:
        return int(getattr(current_user, "unit_id", 0) or 0)

    def _require_module() -> None:
        if not dependencies.live_position_enabled(_unit_id()):
            abort(404)

    def _require_kiosk() -> None:
        _require_module()
        if getattr(current_user, "role", "") != "position_monitor":
            abort(403)

    def _service() -> LivePositionService:
        return LivePositionService(
            dependencies.db,
            LivePositionModels(
                dependencies.Staff,
                dependencies.OperationalPosition,
                dependencies.PositionStatusEvent,
                dependencies.PositionSession,
                dependencies.PositionSessionParticipant,
                dependencies.PositionParticipantRole,
                dependencies.PositionSessionAudit,
            ),
            dependencies.utcnow,
        )

    def _payload() -> dict[str, Any]:
        return request.get_json(silent=True) or request.form.to_dict()

    def _request_key(data: dict[str, Any]) -> str:
        return str(
            data.get("request_key") or request.headers.get("Idempotency-Key", "")
        )

    def _int_field(data: dict[str, Any], name: str, default: int = 0) -> int:
        try:
            return int(data.get(name) or default)
        except (TypeError, ValueError):
            return default

    def _iso_timestamp(value: Any) -> str:
        rendered = value.isoformat()
        if rendered.endswith("+00:00"):
            return rendered[:-6] + "Z"
        return rendered if getattr(value, "tzinfo", None) else rendered + "Z"

    def _position_time_matrix(position: Any) -> dict[str, int]:
        rows = dependencies.OperationalPositionTimeAllowance.query.filter_by(
            unit_id=_unit_id(), position_id=position.id
        ).all()
        return {
            f"{row.weekday}:{row.start_hour}": row.maximum_duration_minutes
            for row in rows
        }

    def _matrix_from_form(data: dict[str, Any]) -> dict[str, int]:
        matrix: dict[str, int] = {}
        for weekday in range(7):
            for hour in range(24):
                raw = str(data.get(f"allowance_{weekday}_{hour}") or "").strip()
                if not raw:
                    continue
                try:
                    minutes = int(raw)
                except ValueError as error:
                    raise LivePositionValidationError(
                        "Matrix allowances must be whole minutes."
                    ) from error
                if not 1 <= minutes <= 1440:
                    raise LivePositionValidationError(
                        "Matrix allowances must be between 1 and 1,440 minutes."
                    )
                matrix[f"{weekday}:{hour}"] = minutes
        return matrix

    def _allowance_minutes(position: Any, at: datetime | None = None) -> int:
        unit = dependencies.db.session.get(dependencies.Unit, _unit_id())
        moment = at or dependencies.utcnow()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        try:
            local_timezone: tzinfo = ZoneInfo(
                getattr(unit, "timezone", "Europe/London") or "Europe/London"
            )
        except ZoneInfoNotFoundError:
            local_timezone = timezone.utc
        local_moment = moment.astimezone(local_timezone)
        return _position_time_matrix(position).get(
            f"{local_moment.weekday()}:{local_moment.hour}",
            position.maximum_session_duration_minutes,
        )

    def _controller(person_id: int) -> Any:
        person = dependencies.Staff.query.filter_by(
            id=person_id,
            unit_id=_unit_id(),
            membership_status="active",
            is_operational=True,
        ).first()
        if not person:
            raise LivePositionValidationError("Select an active controller.")
        return person

    def _configured_position(position_id: int) -> Any:
        position = dependencies.OperationalPosition.query.filter_by(
            id=position_id, unit_id=_unit_id(), is_active=True
        ).first()
        if not position:
            raise LivePositionValidationError("Unknown or inactive position.")
        return position

    def _eligibility_reasons(
        person: Any, *, position: Any | None = None, role_code: str = ""
    ) -> list[str]:
        today = dependencies.utcnow().date()
        reasons: list[str] = []
        if not person.is_operational:
            reasons.append("This person is not currently operational.")
        if not person.medical_expiry or person.medical_expiry < today:
            reasons.append("A current medical is required to log on.")
        legacy_ue = any(
            expiry and expiry >= today
            for expiry in (
                person.tower_ue_expiry,
                person.radar_ue_expiry,
                person.met_ue_expiry,
            )
        )
        endorsement = dependencies.PositionEndorsement.query.filter(
            dependencies.PositionEndorsement.unit_id == _unit_id(),
            dependencies.PositionEndorsement.person_id == person.id,
            dependencies.PositionEndorsement.status == "valid",
            dependencies.PositionEndorsement.valid_from <= today,
            dependencies.db.or_(
                dependencies.PositionEndorsement.valid_until.is_(None),
                dependencies.PositionEndorsement.valid_until >= today,
            ),
        ).first()
        if not legacy_ue and not endorsement:
            reasons.append("A current unit endorsement is required to log on.")
        if position:
            position_uses_endorsements = dependencies.PositionEndorsement.query.filter_by(
                unit_id=_unit_id(), position_id=position.id
            ).first()
            if position_uses_endorsements:
                position_endorsement = dependencies.PositionEndorsement.query.filter(
                    dependencies.PositionEndorsement.unit_id == _unit_id(),
                    dependencies.PositionEndorsement.person_id == person.id,
                    dependencies.PositionEndorsement.position_id == position.id,
                    dependencies.PositionEndorsement.status == "valid",
                    dependencies.PositionEndorsement.valid_from <= today,
                    dependencies.db.or_(
                        dependencies.PositionEndorsement.valid_until.is_(None),
                        dependencies.PositionEndorsement.valid_until >= today,
                    ),
                ).first()
                if not position_endorsement:
                    reasons.append(
                        "A current endorsement for this position is required to log on."
                    )
        if role_code == "ojti" and not person.has_ojti:
            reasons.append("A current OJTI is required.")
        if role_code in {"assessor", "examiner"} and not person.has_assessor:
            reasons.append("A current assessor authority is required.")
        return reasons

    def _validate_primary(person: Any, position: Any) -> None:
        if reasons := _eligibility_reasons(person, position=position):
            raise LivePositionValidationError(reasons[0])

    def _validate_support(person: Any, role: Any, position: Any) -> None:
        if reasons := _eligibility_reasons(
            person, position=position, role_code=role.code
        ):
            raise LivePositionValidationError(reasons[0])

    def _validate_primary_arrangement(person: Any, session_type: str) -> None:
        if bool(getattr(person, "is_trainee", False)) and session_type != "training":
            raise LivePositionValidationError(
                "A trainee requires a current OJTI to log on."
            )

    def _secondary_selection(
        data: dict[str, Any], *, position: Any, required: bool = False
    ) -> tuple[list[dict[str, int]], str]:
        role_code = str(data.get("support_role") or "").strip().lower()
        person_id = _int_field(data, "support_person_id")
        if not role_code and not person_id and not required:
            return [], "operational"
        if role_code not in {"ojti", "assessor"} or not person_id:
            raise LivePositionValidationError(
                "Select both an OJTI or Assessor role and a secondary controller."
            )
        _ensure_participant_roles()
        role = dependencies.PositionParticipantRole.query.filter_by(
            unit_id=_unit_id(), code=role_code, is_active=True, is_primary=False
        ).first()
        if not role:
            raise LivePositionValidationError("That secondary role is unavailable.")
        person = _controller(person_id)
        _validate_support(person, role, position)
        return (
            [{"person_id": person.id, "role_id": role.id}],
            "training" if role.code == "ojti" else "assessment",
        )

    def _mutation_error(error: Exception):
        status = 409 if isinstance(error, LivePositionConflict) else 422
        return jsonify({"ok": False, "error": str(error)}), status

    def _ensure_participant_roles() -> None:
        defaults = (
            ("primary", "Primary controller", True, True),
            ("ojti", "OJTI", False, False),
            ("assessor", "Assessor", False, False),
        )
        existing = {
            row.code
            for row in dependencies.PositionParticipantRole.query.filter_by(
                unit_id=_unit_id()
            ).all()
        }
        missing = [
            dependencies.PositionParticipantRole(
                unit_id=_unit_id(),
                code=code,
                label=label,
                is_primary=is_primary,
                counts_for_currency=counts,
                is_active=True,
            )
            for code, label, is_primary, counts in defaults
            if code not in existing
        ]
        if missing:
            dependencies.db.session.add_all(missing)
            dependencies.db.session.commit()

    @blueprint.get("/")
    @login_required
    def admin_home():
        _require_module()
        if not dependencies.is_admin_user(current_user):
            abort(403)
        return redirect(url_for("administration_home"))

    @blueprint.get("/reports/operational-activity")
    @login_required
    def operational_activity():
        _require_module()
        if getattr(current_user, "role", "") == "position_monitor":
            abort(403)
        unit_id = _unit_id()
        today = dependencies.utcnow().date()
        try:
            start_day = date.fromisoformat(
                request.args.get("start") or (today - timedelta(days=29)).isoformat()
            )
            end_day = date.fromisoformat(request.args.get("end") or today.isoformat())
        except ValueError:
            abort(400, "Enter valid report dates.")
        if end_day < start_day or (end_day - start_day).days > 731:
            abort(400, "Choose a date range of no more than two years.")

        can_view_all = bool(
            dependencies.is_admin_user(current_user)
            or getattr(current_user, "role", "") == "editor"
            or getattr(current_user, "is_wm", False)
            or getattr(current_user, "is_dwm", False)
            or getattr(current_user, "has_ojti", False)
            or getattr(current_user, "has_assessor", False)
        )
        people = (
            dependencies.Staff.query.filter_by(
                unit_id=unit_id, membership_status="active", is_operational=True
            )
            .filter(dependencies.Staff.role != "position_monitor")
            .order_by(dependencies.Staff.name)
            .all()
        )
        watches = (
            dependencies.Watch.query.filter_by(unit_id=unit_id)
            .order_by(dependencies.Watch.order_index, dependencies.Watch.name)
            .all()
        )
        positions = (
            dependencies.OperationalPosition.query.filter_by(unit_id=unit_id)
            .order_by(
                dependencies.OperationalPosition.group_name,
                dependencies.OperationalPosition.display_order,
                dependencies.OperationalPosition.code,
            )
            .all()
        )
        people_by_id = {person.id: person for person in people}
        positions_by_id = {position.id: position for position in positions}
        roles = {
            role.id: role
            for role in dependencies.PositionParticipantRole.query.filter_by(
                unit_id=unit_id
            ).all()
        }
        report_type = request.args.get("scope", "individual")
        if report_type not in {"individual", "watch", "unit"}:
            abort(400, "Choose an individual, watch or unit report.")
        selected_person_id = request.args.get("person_id", type=int)
        selected_watch_id = request.args.get("watch_id", type=int)
        selected_position_id = request.args.get("position_id", type=int)
        if not can_view_all:
            report_type = "individual"
            selected_person_id = current_user.id
            selected_watch_id = None
        elif report_type == "individual" and not selected_person_id:
            selected_watch_id = None
        elif report_type == "watch":
            selected_person_id = None
        elif report_type == "unit":
            selected_person_id = None
            selected_watch_id = None
        if selected_person_id and selected_person_id not in people_by_id:
            abort(404)
        if selected_watch_id and selected_watch_id not in {row.id for row in watches}:
            abort(404)
        if selected_position_id and selected_position_id not in positions_by_id:
            abort(404)

        range_start = datetime.combine(start_day, datetime_time.min)
        range_end = datetime.combine(end_day + timedelta(days=1), datetime_time.min)
        now = dependencies.utcnow()
        sessions = dependencies.PositionSession.query.filter(
            dependencies.PositionSession.unit_id == unit_id,
            dependencies.PositionSession.is_void.is_(False),
            dependencies.PositionSession.started_at < range_end,
            dependencies.db.or_(
                dependencies.PositionSession.ended_at.is_(None),
                dependencies.PositionSession.ended_at > range_start,
            ),
        ).all()
        session_ids = [row.id for row in sessions]
        participant_rows = (
            dependencies.PositionSessionParticipant.query.filter(
                dependencies.PositionSessionParticipant.unit_id == unit_id,
                dependencies.PositionSessionParticipant.session_id.in_(session_ids),
            ).all()
            if session_ids
            else []
        )
        participants_by_session: dict[int, list[Any]] = {}
        for participant in participant_rows:
            participants_by_session.setdefault(participant.session_id, []).append(
                participant
            )

        activity: list[dict[str, Any]] = []
        role_labels = {
            "solo": "Solo",
            "under_training": "Under instruction",
            "under_assessment": "Under assessment",
            "ojti": "OJTI",
            "assessor": "Assessor",
        }

        def add_activity(
            person_id: int,
            position: Any,
            day: date,
            role_code: str,
            minutes: int,
            session_id: int,
        ) -> None:
            person = people_by_id.get(person_id)
            if not person or minutes <= 0:
                return
            if report_type == "individual" and selected_person_id and person.id != selected_person_id:
                return
            if report_type == "watch" and selected_watch_id and person.watch_id != selected_watch_id:
                return
            if selected_position_id and position.id != selected_position_id:
                return
            activity.append(
                {
                    "day": day,
                    "person": person,
                    "watch": person.watch,
                    "position": position,
                    "role": role_code,
                    "role_label": role_labels.get(role_code, role_code.title()),
                    "minutes": minutes,
                    "session_id": session_id,
                }
            )

        for session_row in sessions:
            position = positions_by_id.get(session_row.position_id)
            if not position:
                continue
            session_start = max(session_row.started_at, range_start)
            session_end = min(session_row.ended_at or now, range_end)
            if session_end <= session_start:
                continue
            participants = participants_by_session.get(session_row.id, [])
            cursor_day = session_start.date()
            while cursor_day <= session_end.date():
                day_start = max(
                    session_start, datetime.combine(cursor_day, datetime_time.min)
                )
                day_end = min(
                    session_end,
                    datetime.combine(cursor_day + timedelta(days=1), datetime_time.min),
                )
                if day_end > day_start:
                    all_intervals: list[tuple[datetime, datetime]] = []
                    role_intervals: dict[str, list[tuple[datetime, datetime]]] = {
                        "ojti": [],
                        "assessor": [],
                    }
                    for participant in participants:
                        participant_end = participant.ended_at or now
                        if (
                            participant.started_at < day_end
                            and participant_end > day_start
                        ):
                            interval = (participant.started_at, participant_end)
                            all_intervals.append(interval)
                            role = roles.get(participant.role_id)
                            role_code = (
                                "assessor"
                                if role and role.code in {"assessor", "examiner"}
                                else "ojti"
                            )
                            role_intervals[role_code].append(interval)
                            add_activity(
                                participant.person_id,
                                position,
                                cursor_day,
                                role_code,
                                _overlap_minutes(day_start, day_end, [interval]),
                                session_row.id,
                            )
                    total_minutes = _minutes_between(day_start, day_end)
                    supported_minutes = _overlap_minutes(
                        day_start, day_end, all_intervals
                    )
                    add_activity(
                        session_row.primary_person_id,
                        position,
                        cursor_day,
                        "solo",
                        max(0, total_minutes - supported_minutes),
                        session_row.id,
                    )
                    for support_role, primary_role in (
                        ("ojti", "under_training"),
                        ("assessor", "under_assessment"),
                    ):
                        add_activity(
                            session_row.primary_person_id,
                            position,
                            cursor_day,
                            primary_role,
                            _overlap_minutes(
                                day_start, day_end, role_intervals[support_role]
                            ),
                            session_row.id,
                        )
                cursor_day += timedelta(days=1)

        activity.sort(
            key=lambda row: (
                row["day"],
                row["person"].name,
                row["position"].code,
                row["role"],
            )
        )
        person_summary: dict[int, dict[str, Any]] = {}
        position_summary: dict[int, dict[str, Any]] = {}
        instruction_summary: dict[int, dict[str, Any]] = {}
        report_totals: dict[str, Any] = {
            "roles": {},
            "controller_activity": 0,
            "occupied": 0,
            "instruction": 0,
            "people": set(),
            "sessions": set(),
        }
        for row in activity:
            report_totals["roles"][row["role"]] = (
                report_totals["roles"].get(row["role"], 0) + row["minutes"]
            )
            report_totals["controller_activity"] += row["minutes"]
            report_totals["people"].add(row["person"].id)
            report_totals["sessions"].add(row["session_id"])
            if row["role"] in {"solo", "under_training", "under_assessment"}:
                report_totals["occupied"] += row["minutes"]
            if row["role"] in {"ojti", "assessor"}:
                report_totals["instruction"] += row["minutes"]
            person_total = person_summary.setdefault(
                row["person"].id,
                {"person": row["person"], "roles": {}, "total": 0},
            )
            person_total["roles"][row["role"]] = (
                person_total["roles"].get(row["role"], 0) + row["minutes"]
            )
            person_total["total"] += row["minutes"]
            position_total = position_summary.setdefault(
                row["position"].id,
                {
                    "position": row["position"],
                    "roles": {},
                    "total": 0,
                    "people": set(),
                    "sessions": set(),
                },
            )
            position_total["roles"][row["role"]] = (
                position_total["roles"].get(row["role"], 0) + row["minutes"]
            )
            position_total["sessions"].add(row["session_id"])
            if row["role"] in {"solo", "under_training", "under_assessment"}:
                position_total["total"] += row["minutes"]
                position_total["people"].add(row["person"].id)
            if row["role"] in {"ojti", "assessor"}:
                instruction_total = instruction_summary.setdefault(
                    row["person"].id,
                    {
                        "person": row["person"],
                        "ojti": 0,
                        "assessor": 0,
                        "sessions": set(),
                    },
                )
                instruction_total[row["role"]] += row["minutes"]
                instruction_total["sessions"].add(row["session_id"])

        return render_template(
            "live_position/operational_activity_report.html",
            start_day=start_day,
            end_day=end_day,
            report_type=report_type,
            activity=activity,
            person_summary=sorted(
                person_summary.values(), key=lambda row: row["person"].name
            ),
            position_summary=sorted(
                position_summary.values(),
                key=lambda row: (
                    row["position"].group_name,
                    row["position"].display_order,
                    row["position"].code,
                ),
            ),
            instruction_summary=sorted(
                instruction_summary.values(), key=lambda row: row["person"].name
            ),
            report_totals=report_totals,
            people=people,
            watches=watches,
            positions=positions,
            selected_person_id=selected_person_id,
            selected_person=people_by_id.get(selected_person_id),
            selected_watch_id=selected_watch_id,
            selected_position_id=selected_position_id,
            can_view_all=can_view_all,
            competency_location=dependencies.competency_enabled(unit_id),
        )

    @blueprint.route("/admin/positions", methods=["GET", "POST"])
    @login_required
    def position_configuration():
        _require_module()
        if not dependencies.is_admin_user(current_user):
            abort(403)
        unit_id = _unit_id()
        if request.method == "POST":
            data = _payload()
            action = str(data.get("action") or "")
            if action == "create_group":
                name = str(data.get("group_name") or "").strip()[:80]
                duplicate = dependencies.OperationalPositionGroup.query.filter(
                    dependencies.OperationalPositionGroup.unit_id == unit_id,
                    dependencies.db.func.lower(
                        dependencies.OperationalPositionGroup.name
                    )
                    == name.lower(),
                ).first()
                if not name:
                    flash("Enter a group name.", "error")
                elif duplicate:
                    flash("That position group already exists.", "error")
                else:
                    dependencies.db.session.add(
                        dependencies.OperationalPositionGroup(
                            unit_id=unit_id,
                            name=name,
                            display_order=max(
                                0, _int_field(data, "group_display_order", 100)
                            ),
                            is_active=True,
                        )
                    )
                    dependencies.db.session.commit()
                    flash("Position group added.", "ok")
                    return redirect(url_for("live_position.position_configuration"))
            elif action == "create_category":
                code = str(data.get("category_code") or "").strip().upper()
                label = str(data.get("category_label") or "").strip()
                if not code or not label:
                    flash("Enter a category code and display name.", "error")
                elif dependencies.PositionCurrencyCategory.query.filter_by(
                    unit_id=unit_id, code=code
                ).first():
                    flash("That currency-category code is already in use.", "error")
                else:
                    dependencies.db.session.add(
                        dependencies.PositionCurrencyCategory(
                            unit_id=unit_id,
                            code=code[:30],
                            label=label[:120],
                            description=str(
                                data.get("category_description") or ""
                            ).strip(),
                            is_active=True,
                        )
                    )
                    dependencies.db.session.commit()
                    flash("Currency category added.", "ok")
                    return redirect(url_for("live_position.position_configuration"))
            elif action in {"create_position", "update_position"}:
                position_id = _int_field(data, "position_id")
                position = (
                    dependencies.OperationalPosition.query.filter_by(
                        id=position_id, unit_id=unit_id
                    ).first_or_404()
                    if action == "update_position"
                    else dependencies.OperationalPosition(unit_id=unit_id)
                )
                code = str(data.get("code") or "").strip().upper()
                label = str(data.get("label") or "").strip()
                duplicate = dependencies.OperationalPosition.query.filter(
                    dependencies.OperationalPosition.unit_id == unit_id,
                    dependencies.OperationalPosition.code == code,
                    dependencies.OperationalPosition.id != (position.id or 0),
                ).first()
                category_id = _int_field(data, "currency_category_id")
                category = (
                    dependencies.PositionCurrencyCategory.query.filter_by(
                        id=category_id, unit_id=unit_id, is_active=True
                    ).first()
                    if category_id
                    else None
                )
                group_id = _int_field(data, "position_group_id")
                group = (
                    dependencies.OperationalPositionGroup.query.filter_by(
                        id=group_id, unit_id=unit_id, is_active=True
                    ).first()
                    if group_id
                    else None
                )
                requested_active = str(data.get("is_active") or "") == "on"
                try:
                    allowance_matrix = _matrix_from_form(data)
                except LivePositionValidationError as error:
                    allowance_matrix = None
                    flash(str(error), "error")
                occupied = bool(
                    position.id
                    and dependencies.PositionSession.query.filter_by(
                        unit_id=unit_id,
                        position_id=position.id,
                        ended_at=None,
                        is_void=False,
                    ).first()
                )
                if not code or not label:
                    flash("Enter a position code and display name.", "error")
                elif duplicate:
                    flash("That position code is already in use.", "error")
                elif category_id and not category:
                    flash("Select a valid currency category.", "error")
                elif group_id and not group:
                    flash("Select a valid position group.", "error")
                elif allowance_matrix is None:
                    pass
                elif occupied and not requested_active:
                    flash(
                        "Log off and close this position before making it inactive.",
                        "error",
                    )
                else:
                    if action == "create_position":
                        dependencies.db.session.add(position)
                    position.code = code[:30]
                    position.label = label[:120]
                    position.description = str(data.get("description") or "").strip()
                    position.display_order = max(
                        0, _int_field(data, "display_order", 100)
                    )
                    position.maximum_session_duration_minutes = max(
                        1,
                        min(
                            1440,
                            _int_field(data, "maximum_session_duration_minutes", 120),
                        ),
                    )
                    if position.id is None:
                        dependencies.db.session.flush()
                    dependencies.OperationalPositionTimeAllowance.query.filter_by(
                        unit_id=_unit_id(), position_id=position.id
                    ).delete(synchronize_session=False)
                    for key, minutes in allowance_matrix.items():
                        weekday, start_hour = (int(part) for part in key.split(":"))
                        dependencies.db.session.add(
                            dependencies.OperationalPositionTimeAllowance(
                                unit_id=_unit_id(),
                                position_id=position.id,
                                weekday=weekday,
                                start_hour=start_hour,
                                maximum_duration_minutes=minutes,
                            )
                        )
                    position.group_name = group.name if group else ""
                    position.currency_category_id = category.id if category else None
                    position.supporting_participants_allowed = (
                        str(data.get("supporting_participants_allowed") or "") == "on"
                    )
                    position.multiple_supporting_participants_allowed = (
                        str(data.get("multiple_supporting_participants_allowed") or "")
                        == "on"
                    )
                    position.training_supported = (
                        str(data.get("training_supported") or "") == "on"
                    )
                    position.assessment_supported = (
                        str(data.get("assessment_supported") or "") == "on"
                    )
                    position.is_safety_critical = (
                        str(data.get("is_safety_critical") or "") == "on"
                    )
                    position.is_active = requested_active
                    dependencies.db.session.flush()
                    dependencies.db.session.add(
                        dependencies.PositionSessionAudit(
                            unit_id=unit_id,
                            position_id=position.id,
                            actor_id=current_user.id,
                            action="position_configured",
                            occurred_at=dependencies.utcnow(),
                            new_value_json=json.dumps(
                                {
                                    "code": position.code,
                                    "label": position.label,
                                    "is_active": position.is_active,
                                    "maximum_session_duration_minutes": (
                                        position.maximum_session_duration_minutes
                                    ),
                                    "maximum_session_duration_matrix": allowance_matrix,
                                },
                                sort_keys=True,
                            ),
                            transaction_key=LivePositionService.transaction_key(),
                        )
                    )
                    dependencies.db.session.commit()
                    flash(
                        "Position updated."
                        if action == "update_position"
                        else "Position added.",
                        "ok",
                    )
                    return redirect(url_for("live_position.position_configuration"))
        groups = (
            dependencies.OperationalPositionGroup.query.filter_by(unit_id=unit_id)
            .order_by(
                dependencies.OperationalPositionGroup.display_order,
                dependencies.OperationalPositionGroup.name,
            )
            .all()
        )
        categories = (
            dependencies.PositionCurrencyCategory.query.filter_by(unit_id=unit_id)
            .order_by(dependencies.PositionCurrencyCategory.label)
            .all()
        )
        positions = (
            dependencies.OperationalPosition.query.filter_by(unit_id=unit_id)
            .order_by(
                dependencies.OperationalPosition.display_order,
                dependencies.OperationalPosition.code,
            )
            .all()
        )
        return render_template(
            "live_position/position_configuration.html",
            positions=positions,
            groups=groups,
            categories=categories,
            position_matrices={
                position.id: _position_time_matrix(position) for position in positions
            },
            weekdays=(
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ),
            matrix_hours=range(24),
        )

    @blueprint.get("/kiosk")
    @login_required
    def kiosk_hmi():
        _require_kiosk()
        unit = dependencies.db.session.get(dependencies.Unit, _unit_id())
        return render_template("live_position/kiosk.html", unit=unit)

    @blueprint.get("/api/controllers")
    @login_required
    def controllers():
        _require_kiosk()
        _ensure_participant_roles()
        people = (
            dependencies.Staff.query.filter_by(
                unit_id=_unit_id(),
                membership_status="active",
                is_operational=True,
            )
            .order_by(dependencies.Staff.name)
            .all()
        )
        return jsonify(
            {
                "controllers": [
                    {
                        "id": person.id,
                        "name": person.name,
                        "is_ojti": bool(person.has_ojti),
                        "is_assessor": bool(person.has_assessor),
                    }
                    for person in people
                ],
            }
        )

    @blueprint.post("/api/positions/<int:position_id>/open")
    @login_required
    def open_position(position_id: int):
        _require_kiosk()
        data = _payload()
        try:
            _service().set_position_open(
                unit_id=_unit_id(),
                position_id=position_id,
                actor_id=current_user.id,
                open_position=True,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/close")
    @login_required
    def close_position(position_id: int):
        _require_kiosk()
        data = _payload()
        try:
            _service().set_position_open(
                unit_id=_unit_id(),
                position_id=position_id,
                actor_id=current_user.id,
                open_position=False,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/logon")
    @login_required
    def logon(position_id: int):
        _require_kiosk()
        data = _payload()
        try:
            position = _configured_position(position_id)
            person = _controller(_int_field(data, "person_id"))
            _validate_primary(person, position)
            participants, session_type = _secondary_selection(data, position=position)
            _validate_primary_arrangement(person, session_type)
            _service().start_session(
                unit_id=_unit_id(),
                position_id=position_id,
                person_id=person.id,
                actor_id=current_user.id,
                session_type=session_type,
                maximum_duration_seconds=_allowance_minutes(position) * 60,
                request_key=_request_key(data),
                participants=participants,
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/logoff")
    @login_required
    def logoff(position_id: int):
        _require_kiosk()
        data = _payload()
        active = dependencies.PositionSession.query.filter_by(
            unit_id=_unit_id(),
            position_id=position_id,
            ended_at=None,
            is_void=False,
        ).first()
        if not active:
            return _mutation_error(
                LivePositionConflict("The position is not occupied.")
            )
        try:
            if str(data.get("close_position") or "").lower() in {"1", "true", "yes"}:
                _service().set_position_open(
                    unit_id=_unit_id(),
                    position_id=position_id,
                    actor_id=current_user.id,
                    open_position=False,
                    request_key=_request_key(data),
                )
            else:
                _service().end_session(
                    unit_id=_unit_id(),
                    position_id=position_id,
                    actor_id=current_user.id,
                    request_key=_request_key(data),
                )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/handover")
    @login_required
    def handover(position_id: int):
        _require_kiosk()
        data = _payload()
        try:
            position = _configured_position(position_id)
            incoming = _controller(_int_field(data, "person_id"))
            _validate_primary(incoming, position)
            participants, session_type = _secondary_selection(data, position=position)
            _validate_primary_arrangement(incoming, session_type)
            _service().handover(
                unit_id=_unit_id(),
                position_id=position_id,
                incoming_person_id=incoming.id,
                actor_id=current_user.id,
                session_type=session_type,
                maximum_duration_seconds=_allowance_minutes(position) * 60,
                request_key=_request_key(data),
                participants=participants,
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post("/api/positions/<int:position_id>/participants")
    @login_required
    def add_participant(position_id: int):
        _require_kiosk()
        data = _payload()
        try:
            position = _configured_position(position_id)
            participants, _session_type = _secondary_selection(
                data, position=position, required=True
            )
            participant = participants[0]
            row = _service().add_participant(
                unit_id=_unit_id(),
                position_id=position_id,
                person_id=participant["person_id"],
                role_id=participant["role_id"],
                actor_id=current_user.id,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True, "participant_id": row.id})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    @blueprint.post(
        "/api/positions/<int:position_id>/participants/<int:participant_id>/logoff"
    )
    @login_required
    def remove_participant(position_id: int, participant_id: int):
        _require_kiosk()
        data = _payload()
        participant = dependencies.PositionSessionParticipant.query.filter_by(
            id=participant_id, unit_id=_unit_id(), ended_at=None
        ).first()
        if not participant:
            return _mutation_error(
                LivePositionConflict("The secondary controller is no longer active.")
            )
        try:
            _service().end_participant(
                unit_id=_unit_id(),
                position_id=position_id,
                participant_id=participant_id,
                actor_id=current_user.id,
                request_key=_request_key(data),
            )
            return jsonify({"ok": True})
        except (LivePositionConflict, LivePositionValidationError) as error:
            return _mutation_error(error)

    def _state_payload() -> dict[str, Any]:
        unit_id = _unit_id()
        now = dependencies.utcnow()
        groups = (
            dependencies.OperationalPositionGroup.query.filter_by(
                unit_id=unit_id, is_active=True
            )
            .order_by(
                dependencies.OperationalPositionGroup.display_order,
                dependencies.OperationalPositionGroup.name,
            )
            .all()
        )
        group_order = {group.name: group.display_order for group in groups}
        positions = (
            dependencies.OperationalPosition.query.filter_by(
                unit_id=unit_id, is_active=True
            )
            .order_by(
                dependencies.OperationalPosition.display_order,
                dependencies.OperationalPosition.code,
            )
            .all()
        )
        state = []
        for position in positions:
            status_event = (
                dependencies.PositionStatusEvent.query.filter_by(
                    unit_id=unit_id, position_id=position.id
                )
                .order_by(
                    dependencies.PositionStatusEvent.occurred_at.desc(),
                    dependencies.PositionStatusEvent.id.desc(),
                )
                .first()
            )
            session = dependencies.PositionSession.query.filter_by(
                unit_id=unit_id,
                position_id=position.id,
                ended_at=None,
                is_void=False,
            ).first()
            primary = (
                dependencies.db.session.get(
                    dependencies.Staff, session.primary_person_id
                )
                if session
                else None
            )
            participants = []
            maximum_duration_seconds = (
                session.maximum_duration_seconds
                if session and session.maximum_duration_seconds
                else _allowance_minutes(
                    position, session.started_at if session else now
                )
                * 60
            )
            if session:
                for row in dependencies.PositionSessionParticipant.query.filter_by(
                    unit_id=unit_id, session_id=session.id, ended_at=None
                ).all():
                    person = dependencies.db.session.get(
                        dependencies.Staff, row.person_id
                    )
                    role = dependencies.db.session.get(
                        dependencies.PositionParticipantRole, row.role_id
                    )
                    participants.append(
                        {
                            "id": row.id,
                            "person_name": person.name if person else "Unknown",
                            "role_id": row.role_id,
                            "role_label": role.label if role else "Supporting",
                            "role_code": role.code if role else "",
                            "started_at": _iso_timestamp(row.started_at),
                            "maximum_duration_seconds": maximum_duration_seconds,
                            "eligibility_warnings": (
                                _eligibility_reasons(
                                    person, position=position, role_code=role.code
                                )
                                if person and role
                                else ["Supporting controller details are unavailable."]
                            ),
                        }
                    )
            eligibility_warnings = (
                _eligibility_reasons(primary, position=position)
                if primary and session
                else []
            )
            if primary and primary.is_trainee and session:
                has_ojti = any(
                    participant["role_code"] == "ojti" for participant in participants
                )
                if not has_ojti:
                    eligibility_warnings.append(
                        "Trainee is active without a current OJTI."
                    )
            physical_status = status_event.status if status_event else "closed"
            display_status = (
                "closed"
                if physical_status == "closed"
                else session.session_type
                if session
                else "vacant"
            )
            state.append(
                {
                    "id": position.id,
                    "code": position.code,
                    "label": position.label,
                    "group_name": position.group_name,
                    "group_order": group_order.get(position.group_name, 999999),
                    "physical_status": physical_status,
                    "display_status": display_status,
                    "primary": (
                        {
                            "id": primary.id,
                            "name": primary.name,
                            "started_at": _iso_timestamp(session.started_at),
                            "maximum_duration_seconds": maximum_duration_seconds,
                            "due_off_at": (
                                _iso_timestamp(session.due_off_at)
                                if session.due_off_at
                                else None
                            ),
                        }
                        if primary and session
                        else None
                    ),
                    "participants": participants,
                    "eligibility_warnings": eligibility_warnings,
                }
            )
        return {"server_time": _iso_timestamp(now), "positions": state}

    @blueprint.get("/api/state")
    @login_required
    def live_state():
        _require_kiosk()
        return jsonify(_state_payload())

    @blueprint.get("/api/events")
    @login_required
    def live_events():
        _require_kiosk()
        unit_id = _unit_id()
        database_route = dependencies.authenticated_database_route_optional()

        @stream_with_context
        def events():
            # Database state is authoritative. A short heartbeat makes this
            # work consistently across multiple web processes; the browser's
            # normal polling remains the fallback when SSE is unavailable.
            # Flask tears down the original request before iterating a streamed
            # response, so restore only the already-authenticated tenant route
            # for the lifetime of the stream.
            with dependencies.authenticated_unit_context(
                unit_id,
                database_route.secret_name if database_route is not None else None,
            ):
                for _index in range(120):
                    yield f"event: state\ndata: {json.dumps(_state_payload())}\n\n"
                    time.sleep(5)

        return Response(
            events(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return blueprint
