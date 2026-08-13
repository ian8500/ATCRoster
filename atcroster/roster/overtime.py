"""Overtime candidate and notification route."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, render_template, request
from flask_login import current_user, login_required


def count_tagged_assignments(
    staff_id: int,
    upto: Any,
    tags: tuple[str, ...],
    *,
    Assignment: Any,
    parse_annotation: Callable[[Any], Any],
    annotation_tags_for: Callable[[str], set[str]],
) -> dict[str, int]:
    """Count tagged annotations in the current April-to-March year."""
    start = upto.replace(
        year=upto.year if upto.month >= 4 else upto.year - 1,
        month=4,
        day=1,
    )
    counts = dict.fromkeys(tags, 0)
    assignments = Assignment.query.filter(
        Assignment.staff_id == staff_id,
        Assignment.day >= start,
        Assignment.day <= upto,
    ).all()
    for assignment in assignments:
        parsed = parse_annotation(assignment.annotation)
        if not parsed:
            continue
        assignment_tags = annotation_tags_for(parsed["type"])
        for tag in tags:
            if tag in assignment_tags:
                counts[tag] += 1
    return counts


def worked_like_consecutive_days(
    staff: Any,
    upto_day: Any,
    *,
    Assignment: Any,
    working_codes: Callable[[], set[str]],
    lookback_days: int = 10,
) -> int:
    count = 0
    current_day = upto_day
    codes = working_codes()
    for _ in range(lookback_days):
        assignment = Assignment.query.filter_by(
            staff_id=staff.id, day=current_day
        ).first()
        code = assignment.code if assignment else None
        if not code or code not in codes:
            break
        count += 1
        current_day -= timedelta(days=1)
    return count


def had_sickness_within_48_hours(
    staff: Any,
    reference_day: Any,
    reference_shift: Any,
    *,
    Assignment: Any,
    span: Callable[..., tuple[Any, Any]],
    get_shift: Callable[..., Any],
) -> bool:
    reference_start, _ = (
        span(reference_day, reference_shift)
        if reference_shift
        else (datetime.combine(reference_day, time(0, 0)), None)
    )
    start_window = reference_start - timedelta(hours=48)
    assignments = Assignment.query.filter(
        Assignment.staff_id == staff.id,
        Assignment.day >= start_window.date() - timedelta(days=1),
        Assignment.day <= reference_start.date(),
    ).all()
    for assignment in assignments:
        if assignment.code not in {"SC", "SSC"}:
            continue
        shift = get_shift(assignment.code)
        start, end = span(assignment.day, shift) if shift else (None, None)
        if start and end and end > start_window and start < reference_start:
            return True
    return False


def has_in_date_endorsement(staff: Any, reference_day: Any) -> bool:
    """Return whether tower or radar endorsement is valid and unrestricted."""

    def valid(expiry: Any, under_training: bool) -> bool:
        return not under_training and expiry is not None and expiry >= reference_day

    return valid(staff.tower_ue_expiry, staff.tower_ut) or valid(
        staff.radar_ue_expiry, staff.radar_ut
    )


@dataclass(frozen=True)
class OvertimeDependencies:
    ShiftType: Any
    Staff: Any
    current_unit_id: Callable[[], int]
    consume_rate_limit: Callable[..., bool]
    is_editor_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    parse_date: Callable[[str | None], Any]
    compute_candidates: Callable[[Any, str], tuple[list[Any], list[Any], str | None]]
    can_send_messages: Callable[[Any], bool]
    send_sms: Callable[[list[Any], str], tuple[int, list[tuple[Any, str]]]]
    default_sms_body: Callable[[Any, str], str]
    sms_configured: Callable[[], bool]


@dataclass(frozen=True)
class OvertimeCandidateDependencies:
    Assignment: Any
    Staff: Any
    Watch: Any
    current_unit_id: Callable[[], int]
    get_shift: Callable[..., Any]
    ensure_assignments_for_range: Callable[[Any, Any], None]
    annotation_codes_for_tag: Callable[..., list[str]]
    get_annotation_config: Callable[[str], Any]
    staff_has_shift_qualification: Callable[[Any, Any, Any], bool]
    has_in_date_ue: Callable[[Any, Any], bool]
    worked_like_consecutive_days: Callable[..., int]
    would_create_new_fatigue_issues: Callable[..., dict[Any, list[str]]]
    count_aava_soal: Callable[[int, Any], tuple[int, int]]
    had_sc_within_48h: Callable[[Any, Any, Any], bool]


class OvertimeCandidateService:
    """Find and rank operational staff eligible for an overtime duty."""

    def __init__(self, dependencies: OvertimeCandidateDependencies):
        self.dependencies = dependencies

    def compute(
        self, chosen_date: Any, chosen_shift_code: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
        dependencies = self.dependencies
        shift_code = (chosen_shift_code or "").upper().strip()
        unit_id = dependencies.current_unit_id()
        shift = dependencies.get_shift(shift_code, unit_id)
        if not (chosen_date and shift and shift.is_working):
            return [], [], "Please select a valid date and working shift."

        lookahead_days = 14
        dependencies.ensure_assignments_for_range(
            chosen_date - timedelta(days=30),
            chosen_date + timedelta(days=lookahead_days),
        )
        staff_members = (
            dependencies.Staff.query.outerjoin(
                dependencies.Watch,
                dependencies.Staff.watch_id == dependencies.Watch.id,
            )
            .filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.is_operational.is_(True),
            )
            .order_by(dependencies.Watch.order_index, dependencies.Staff.name)
            .all()
        )

        soal_codes = dependencies.annotation_codes_for_tag("soal", active_only=False)
        soal_display = "SOAL"
        if soal_codes:
            first = soal_codes[0]
            info = dependencies.get_annotation_config(first)
            soal_display = (info.get("label") if info else first) or first

        results: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for staff in staff_members:
            reasons = self._exclusion_reasons(
                staff, shift, chosen_date, shift_code, unit_id, lookahead_days
            )
            assignment = dependencies.Assignment.query.filter_by(
                unit_id=unit_id, staff_id=staff.id, day=chosen_date
            ).first()
            rostered_code = assignment.code if assignment else "OFF"
            blocking, warnings = self._fatigue_outcome(
                staff, chosen_date, shift_code, lookahead_days
            )
            reasons.extend(blocking)
            if reasons:
                excluded.append(
                    {
                        "staff": staff,
                        "watch": self._watch_name(staff),
                        "rostered_code": rostered_code,
                        "reasons": reasons,
                    }
                )
                continue

            aava, soal = dependencies.count_aava_soal(
                staff.id, chosen_date - timedelta(days=1)
            )
            flags = []
            if rostered_code == "AL":
                flags.append(f"On AL that day — {soal_display} required")
            if dependencies.had_sc_within_48h(staff, chosen_date, shift):
                flags.append("SC/SSC within 48h — managerial approval required")
            flags.extend(warnings)
            results.append(
                {
                    "staff": staff,
                    "watch": self._watch_name(staff),
                    "aava_to_date": aava,
                    "soal_to_date": soal,
                    "total_to_date": aava + soal,
                    "score": aava + soal,
                    "flags": flags,
                }
            )

        results.sort(
            key=lambda row: (
                row["aava_to_date"],
                row["soal_to_date"],
                row["staff"].name.lower(),
            )
        )
        excluded.sort(key=lambda row: row["staff"].name.lower())
        return results, excluded, None

    def _exclusion_reasons(
        self, staff, shift, chosen_date, shift_code, unit_id, lookahead_days
    ) -> list[str]:
        dependencies = self.dependencies
        reasons = ["Opted out of overtime"] if staff.exclude_from_ot else []
        if not reasons and not dependencies.staff_has_shift_qualification(
            staff, shift, chosen_date
        ):
            qualification = (shift.required_qualification or "").strip().upper()
            reasons.append(
                f"Missing or expired {qualification} qualification"
                if qualification
                else "Missing required shift qualification"
            )
        assignment = dependencies.Assignment.query.filter_by(
            unit_id=unit_id, staff_id=staff.id, day=chosen_date
        ).first()
        rostered_code = assignment.code if assignment else "OFF"
        rostered_shift = dependencies.get_shift(rostered_code, unit_id)
        if rostered_shift and rostered_shift.is_working:
            reasons.append(f"Already rostered for {rostered_code}")
        if rostered_code in {"SC", "SSC"}:
            reasons.append(f"Rostered {rostered_code}")
        if not dependencies.has_in_date_ue(staff, chosen_date):
            reasons.append("No in-date tower or radar endorsement")
        consecutive = dependencies.worked_like_consecutive_days(
            staff, chosen_date - timedelta(days=1), lookback_days=6
        )
        if consecutive >= 6:
            reasons.append("Already worked six consecutive duties")
        return reasons

    def _fatigue_outcome(
        self, staff, chosen_date, shift_code, lookahead_days
    ) -> tuple[list[str], list[str]]:
        issues = self.dependencies.would_create_new_fatigue_issues(
            staff,
            chosen_date,
            shift_code,
            lookback_days=30,
            lookahead_days=lookahead_days,
        )
        warnings = []
        blocking = []
        for finding_day, findings in issues.items():
            for finding in findings:
                if finding.startswith(("D24:", "D24 rest deficit")):
                    warnings.append(f"{finding_day.isoformat()}: {finding}")
                else:
                    blocking.append(finding)
        return (
            ["Blocking fatigue rule: " + "; ".join(sorted(set(blocking)))]
            if blocking
            else [],
            warnings,
        )

    @staticmethod
    def _watch_name(staff: Any) -> str:
        return staff.watch.name.replace("Watch ", "") if staff.watch else "-"


def create_overtime_blueprint(dependencies: OvertimeDependencies) -> Blueprint:
    blueprint = Blueprint("overtime", __name__)
    ShiftType = dependencies.ShiftType
    Staff = dependencies.Staff
    _current_unit_id = dependencies.current_unit_id
    _consume_rate_limit = dependencies.consume_rate_limit
    is_editor_user = dependencies.is_editor_user
    _validate_csrf = dependencies.validate_csrf
    _parse_date = dependencies.parse_date
    _compute_overtime_candidates = dependencies.compute_candidates
    can_send_unit_messages = dependencies.can_send_messages
    _send_overtime_sms_notifications = dependencies.send_sms
    _default_overtime_sms_body = dependencies.default_sms_body
    _sms_service_configured = dependencies.sms_configured

    @login_required
    def overtime():
        if request.method == "POST" and not _consume_rate_limit(
            "overtime-search",
            current_user.id,
            limit=60,
            window=timedelta(hours=1),
        ):
            abort(429)
        if not (
            is_editor_user(current_user)
            or getattr(current_user, "is_wm", False)
            or getattr(current_user, "is_dwm", False)
        ):
            abort(403)

        unit_id = _current_unit_id()
        shifts = (
            ShiftType.query.filter_by(unit_id=unit_id, is_working=True)
            .order_by(ShiftType.code)
            .all()
        )
        overtime_staff = (
            Staff.query.filter_by(unit_id=unit_id, is_operational=True)
            .order_by(Staff.name)
            .all()
        )
        results = []
        excluded = []
        what_if_result = None
        chosen_date = None
        chosen_shift = None
        what_if_staff_id = None
        selected_staff_ids: set[str] = set()
        sms_body = ""
        searched = request.method == "POST"

        if request.method == "POST":
            _validate_csrf()
            action = request.form.get("action", "find")
            chosen_date = _parse_date(request.form.get("date"))
            chosen_shift = (request.form.get("shift_code") or "").upper().strip()
            raw_what_if_staff_id = request.form.get("what_if_staff_id", "")
            what_if_staff_id = (
                int(raw_what_if_staff_id) if raw_what_if_staff_id.isdigit() else None
            )
            selected_staff_ids = {sid for sid in request.form.getlist("staff_ids")}
            sms_body = (request.form.get("message") or "").strip()

            results, excluded, error_msg = _compute_overtime_candidates(
                chosen_date, chosen_shift
            )

            if action == "what_if":
                searched = False
                selected_staff = next(
                    (
                        person
                        for person in overtime_staff
                        if person.id == what_if_staff_id
                    ),
                    None,
                )
                if error_msg:
                    flash(error_msg, "error")
                elif selected_staff is None:
                    flash("Select an ATCO to check.", "error")
                else:
                    eligible_result = next(
                        (
                            row
                            for row in results
                            if row["staff"].id == selected_staff.id
                        ),
                        None,
                    )
                    excluded_result = next(
                        (
                            row
                            for row in excluded
                            if row["staff"].id == selected_staff.id
                        ),
                        None,
                    )
                    if eligible_result:
                        what_if_result = {
                            "eligible": True,
                            "staff": selected_staff,
                            "flags": eligible_result["flags"],
                        }
                    else:
                        what_if_result = {
                            "eligible": False,
                            "staff": selected_staff,
                            "reasons": (
                                excluded_result["reasons"]
                                if excluded_result
                                else ["Eligibility could not be determined"]
                            ),
                        }
            elif action == "send_sms":
                if not can_send_unit_messages(current_user):
                    abort(403)
                if error_msg:
                    flash(error_msg, "error")
                    results = []
                else:
                    if not sms_body:
                        flash("Enter a message to send.", "error")
                    elif len(sms_body) > 480:
                        flash("Message is too long (limit 480 characters).", "error")
                    else:
                        eligible_map = {r["staff"].id: r["staff"] for r in results}
                        selected_staff = [
                            eligible_map[int(sid)]
                            for sid in selected_staff_ids
                            if sid.isdigit() and int(sid) in eligible_map
                        ]
                        missing_ids = [
                            sid
                            for sid in selected_staff_ids
                            if sid.isdigit() and int(sid) not in eligible_map
                        ]
                        if not selected_staff:
                            flash("Select at least one eligible staff member.", "error")
                        else:
                            if missing_ids:
                                flash(
                                    "Some selected staff are no longer eligible; please refresh the list.",
                                    "error",
                                )
                            sent, failures = _send_overtime_sms_notifications(
                                selected_staff, sms_body
                            )
                            if sent:
                                plural = "s" if sent != 1 else ""
                                flash(f"SMS sent to {sent} staff member{plural}.", "ok")
                            if failures:
                                parts = []
                                for staff, msg in failures:
                                    name = staff.name if staff else "System"
                                    parts.append(f"{name}: {msg}")
                                flash("SMS failed for " + "; ".join(parts), "error")

            else:  # action == find or unknown
                if error_msg:
                    flash(error_msg, "error")
                    results = []

            if not sms_body:
                sms_body = _default_overtime_sms_body(chosen_date, chosen_shift)

        sms_ready = _sms_service_configured()

        return render_template(
            "overtime.html",
            shifts=shifts,
            results=results,
            overtime_staff=overtime_staff,
            what_if_result=what_if_result,
            what_if_staff_id=what_if_staff_id,
            chosen_date=chosen_date,
            chosen_shift=chosen_shift,
            sms_body=sms_body,
            sms_ready=sms_ready,
            selected_staff_ids=selected_staff_ids,
            searched=searched,
            excluded=excluded,
        )

    # ===== Leave Report (HTML + CSV) =====
    # (unchanged core; monthly AL-only kept to endpoints)

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/overtime", "overtime", overtime, methods=("GET", "POST")
        )

    return blueprint
