"""Overtime candidate and notification route."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import Blueprint, abort, flash, render_template, request
from flask_login import current_user, login_required


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
