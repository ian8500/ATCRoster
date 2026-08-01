"""Route ownership for absence and shift-request workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import re
from typing import Any, Callable

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required


@dataclass(frozen=True)
class AbsenceRequestDependencies:
    db: Any
    Staff: Any
    Leave: Any
    Assignment: Any
    is_admin_user: Callable
    parse_year_month: Callable
    month_range: Callable
    clamp_prev_next: Callable
    validate_csrf: Callable
    get_absence_types: Callable
    save_absence_types: Callable
    tenant_get: Callable
    current_unit_id: Callable
    refresh_day_from_pattern_and_leave: Callable
    group_sickness_instances: Callable
    ShiftType: Any
    ShiftRequest: Any
    unit_request_rules: Callable
    request_date_bounds: Callable
    is_month_locked: Callable
    request_audit: Callable
    utcnow: Callable
    safe_request_admin_month: Callable
    request_statuses: frozenset
    request_transitions: dict
    would_create_new_fatigue_issues: Callable
    staff_has_shift_qualification: Callable
    can_override_roster_conflicts: Callable
    notify_requester: Callable
    lock_roster_month: Callable[[int, int, int], Any]
    record_toil_transaction: Callable[..., Any]


def create_absence_requests_blueprint(
    dependencies: AbsenceRequestDependencies,
) -> Blueprint:
    blueprint = Blueprint("absence_requests", __name__)

    @login_required
    def leave():
        # Page visibility: editors & admins only
        if not (
            dependencies.is_admin_user(current_user)
            or getattr(current_user, "role", "") in ("editor", "admin")
        ):
            abort(403)

        staff = dependencies.Staff.query.order_by(dependencies.Staff.name).all()

        # ---------- month selection ----------
        today = date.today()
        ym_param = request.args.get("ym") or f"{today.year:04d}-{today.month:02d}"
        year, month = dependencies.parse_year_month(ym_param)
        start_of_month, days = dependencies.month_range(year, month)
        end_of_month = days[-1]
        month_title = datetime(year, month, 1).strftime("%B %Y")
        prev_ym, next_ym = dependencies.clamp_prev_next(year, month)

        if request.method == "POST":
            dependencies.validate_csrf()
            # (still restrict POST actions too)
            if not (
                dependencies.is_admin_user(current_user)
                or getattr(current_user, "role", "") in ("editor", "admin")
            ):
                flash("Editors or Admins only.", "error")
                return redirect(url_for("leave", ym=ym_param))

            form = request.form.get("form", "")

            if form in {"absence_type_add", "absence_type_delete"}:
                if not dependencies.is_admin_user(current_user):
                    abort(403)
                types = dependencies.get_absence_types(active_only=False)
                if form == "absence_type_add":
                    code = (request.form.get("code") or "").strip().upper()
                    label = (request.form.get("label") or "").strip()
                    category = (request.form.get("category") or "").strip().lower()
                    if (
                        not re.fullmatch(r"[A-Z0-9]{1,10}", code)
                        or category not in {"leave", "sickness"}
                        or not label
                    ):
                        flash(
                            "Enter a name, category and a 1–10 character code.", "error"
                        )
                        return redirect(url_for("leave", ym=ym_param))
                    existing = next(
                        (item for item in types if item["code"] == code), None
                    )
                    if existing:
                        existing.update(
                            label=label[:80], category=category, active=True
                        )
                    else:
                        types.append(
                            {
                                "code": code,
                                "label": label[:80],
                                "category": category,
                                "active": True,
                            }
                        )
                    dependencies.save_absence_types(types)
                    flash(f"{label} is now available for this airport.", "ok")
                else:
                    code = (request.form.get("code") or "").strip().upper()
                    item = next((item for item in types if item["code"] == code), None)
                    if not item:
                        abort(404)
                    item["active"] = False
                    dependencies.save_absence_types(types)
                    flash(
                        f"{item['label']} was removed from new records and reports. "
                        "Historical records were retained.",
                        "ok",
                    )
                return redirect(url_for("leave", ym=ym_param))

            if form == "leave_add":
                staff_id = int(request.form["staff_id"])
                lv_type = request.form["leave_type"].upper().strip()
                start_d = date.fromisoformat(request.form["start"])
                end_d = date.fromisoformat(request.form["end"])

                # NEW: allow TOU8 / TOUI in this form (write to roster, deduct TOIL)
                if lv_type in {"TOU8", "TOUI"}:
                    s = dependencies.tenant_get(dependencies.Staff, staff_id)
                    if not s:
                        abort(404)
                    used_per_day_half = 2 if lv_type == "TOU8" else 1
                    cur = start_d
                    while cur <= end_d:
                        a = dependencies.Assignment.query.filter_by(
                            unit_id=dependencies.current_unit_id(),
                            staff_id=staff_id,
                            day=cur,
                        ).first()
                        if not a:
                            a = dependencies.Assignment(staff=s, day=cur)
                        a.code, a.source, a.note, a.annotation = (
                            lv_type,
                            "manual",
                            "toil use (via leave form)",
                            "",
                        )
                        dependencies.db.session.add(a)
                        dependencies.db.session.flush()
                        dependencies.record_toil_transaction(
                            s.id,
                            -used_per_day_half,
                            f"TOIL use {lv_type} on {cur.isoformat()}",
                            current_user.id,
                            transaction_key=hashlib.sha256(
                                f"leave-toil:{dependencies.current_unit_id()}:"
                                f"{s.id}:{cur.isoformat()}:{lv_type}".encode()
                            ).hexdigest(),
                            source_type="leave_form",
                            source_id=a.id,
                        )
                        cur += timedelta(days=1)
                    dependencies.db.session.commit()
                    flash(
                        f"TOIL use recorded: {lv_type} from {start_d.isoformat()} to {end_d.isoformat()}.",
                        "ok",
                    )
                    return redirect(url_for("leave", ym=ym_param))

                # Original behaviour: AL/PL/SPL create Leave rows
                active_leave_codes = {
                    item["code"] for item in dependencies.get_absence_types("leave")
                }
                if lv_type not in active_leave_codes:
                    flash("Select an active leave type for this airport.", "error")
                    return redirect(url_for("leave", ym=ym_param))

                lv = dependencies.Leave(
                    staff_id=staff_id, leave_type=lv_type, start=start_d, end=end_d
                )
                dependencies.db.session.add(lv)
                dependencies.db.session.commit()
                s = dependencies.tenant_get(dependencies.Staff, staff_id)
                if not s:
                    abort(404)
                cur = start_d
                while cur <= end_d:
                    dependencies.refresh_day_from_pattern_and_leave(s, cur)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash("Leave recorded", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "leave_edit":
                lid = int(request.form["leave_id"])
                lv = dependencies.Leave.query.filter_by(
                    id=lid, unit_id=dependencies.current_unit_id()
                ).first_or_404()
                old_range = (lv.start, lv.end)
                lv.staff_id = int(request.form["staff_id"])
                lv.leave_type = request.form["leave_type"].upper()
                if lv.leave_type not in {
                    item["code"] for item in dependencies.get_absence_types("leave")
                }:
                    flash("Select an active leave type for this airport.", "error")
                    return redirect(url_for("leave", ym=ym_param))
                lv.start = date.fromisoformat(request.form["start"])
                lv.end = date.fromisoformat(request.form["end"])
                dependencies.db.session.commit()
                s = dependencies.tenant_get(dependencies.Staff, lv.staff_id)
                for rng in [old_range, (lv.start, lv.end)]:
                    cur = rng[0]
                    while cur <= rng[1]:
                        dependencies.refresh_day_from_pattern_and_leave(s, cur)
                        cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash("Leave updated", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "leave_delete":
                lid = int(request.form["leave_id"])
                lv = dependencies.Leave.query.filter_by(
                    id=lid, unit_id=dependencies.current_unit_id()
                ).first_or_404()
                s = dependencies.tenant_get(dependencies.Staff, lv.staff_id)
                start_d, end_d = lv.start, lv.end
                dependencies.db.session.delete(lv)
                dependencies.db.session.commit()
                cur = start_d
                while cur <= end_d:
                    dependencies.refresh_day_from_pattern_and_leave(s, cur)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash("Leave deleted.", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "sick_add":
                staff_id = int(request.form["staff_id"])
                code = request.form["sick_code"].upper()
                sickness_codes = {
                    item["code"] for item in dependencies.get_absence_types("sickness")
                }
                if code not in sickness_codes:
                    flash("Invalid sickness code.", "error")
                    return redirect(url_for("leave", ym=ym_param))
                start_d = date.fromisoformat(request.form["start"])
                end_d = date.fromisoformat(request.form["end"])
                s = dependencies.tenant_get(dependencies.Staff, staff_id)
                if not s:
                    abort(404)
                cur = start_d
                while cur <= end_d:
                    a = dependencies.Assignment.query.filter_by(
                        unit_id=dependencies.current_unit_id(),
                        staff_id=staff_id,
                        day=cur,
                    ).first()
                    if not a:
                        a = dependencies.Assignment(staff=s, day=cur)
                    a.code, a.source, a.note, a.annotation = (
                        code,
                        "manual",
                        "sickness",
                        "",
                    )
                    dependencies.db.session.add(a)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash(f"Sickness {code} recorded.", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "sick_edit":
                staff_id = int(request.form["staff_id"])
                start_d = date.fromisoformat(request.form["start"])
                end_d = date.fromisoformat(request.form["end"])
                new_code = request.form["sick_code"].upper()
                sickness_codes = {
                    item["code"] for item in dependencies.get_absence_types("sickness")
                }
                if new_code not in sickness_codes:
                    flash("Invalid sickness code.", "error")
                    return redirect(url_for("leave", ym=ym_param))
                cur = start_d
                while cur <= end_d:
                    a = dependencies.Assignment.query.filter_by(
                        unit_id=dependencies.current_unit_id(),
                        staff_id=staff_id,
                        day=cur,
                    ).first()
                    if a and a.code in {
                        item["code"]
                        for item in dependencies.get_absence_types(
                            "sickness", active_only=False
                        )
                    }:
                        a.code = new_code
                        a.annotation = ""
                        a.source = "manual"
                        a.note = "sickness"
                        dependencies.db.session.add(a)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash("Sickness updated.", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "sick_delete":
                staff_id = int(request.form["staff_id"])
                start_d = date.fromisoformat(request.form["start"])
                end_d = date.fromisoformat(request.form["end"])
                s = dependencies.tenant_get(dependencies.Staff, staff_id)
                if not s:
                    abort(404)
                cur = start_d
                while cur <= end_d:
                    a = dependencies.Assignment.query.filter_by(
                        unit_id=dependencies.current_unit_id(),
                        staff_id=staff_id,
                        day=cur,
                    ).first()
                    if a and a.code in {
                        item["code"]
                        for item in dependencies.get_absence_types(
                            "sickness", active_only=False
                        )
                    }:
                        dependencies.db.session.delete(a)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                cur = start_d
                while cur <= end_d:
                    dependencies.refresh_day_from_pattern_and_leave(s, cur)
                    cur += timedelta(days=1)
                dependencies.db.session.commit()
                flash("Sickness deleted.", "ok")
                return redirect(url_for("leave", ym=ym_param))

            if form == "toil_use":
                staff_id = int(request.form["staff_id"])
                code = request.form["toil_code"].upper()
                if code not in {"TOU8", "TOUI"}:
                    flash("Invalid TOIL code.", "error")
                    return redirect(url_for("leave", ym=ym_param))
                day = date.fromisoformat(request.form["day"])
                s = dependencies.tenant_get(dependencies.Staff, staff_id)
                if not s:
                    abort(404)
                a = dependencies.Assignment.query.filter_by(
                    unit_id=dependencies.current_unit_id(),
                    staff_id=staff_id,
                    day=day,
                ).first()
                if not a:
                    a = dependencies.Assignment(staff=s, day=day)
                a.code, a.source, a.note, a.annotation = code, "manual", "toil use", ""
                dependencies.db.session.add(a)
                dependencies.db.session.flush()
                used_half = 2 if code == "TOU8" else 1
                dependencies.record_toil_transaction(
                    s.id,
                    -used_half,
                    f"TOIL use {code} on {day.isoformat()}",
                    current_user.id,
                    transaction_key=hashlib.sha256(
                        f"toil-use:{dependencies.current_unit_id()}:"
                        f"{s.id}:{day.isoformat()}:{code}".encode()
                    ).hexdigest(),
                    source_type="toil_use",
                    source_id=a.id,
                )
                dependencies.db.session.commit()
                flash(f"TOIL used: {code} on {day.isoformat()}.", "ok")
                return redirect(url_for("leave", ym=ym_param))

        # ---------- GET: month-filtered data ----------
        leaves = (
            dependencies.Leave.query.filter(
                dependencies.Leave.end >= start_of_month,
                dependencies.Leave.start <= end_of_month,
            )
            .order_by(dependencies.Leave.start.asc())
            .all()
        )
        all_sickness_codes = [
            item["code"]
            for item in dependencies.get_absence_types("sickness", active_only=False)
        ]
        sickness = (
            dependencies.Assignment.query.filter(
                dependencies.Assignment.code.in_(all_sickness_codes)
            )
            .order_by(
                dependencies.Assignment.staff_id.asc(),
                dependencies.Assignment.day.asc(),
            )
            .all()
        )
        sickness_instances = dependencies.group_sickness_instances(
            sickness, start_of_month, end_of_month
        )

        return render_template(
            "leave.html",
            staff=staff,
            leaves=leaves,
            sickness_instances=sickness_instances,
            leave_types=dependencies.get_absence_types("leave"),
            sickness_types=dependencies.get_absence_types("sickness"),
            absence_types=dependencies.get_absence_types(active_only=False),
            ym=f"{year:04d}-{month:02d}",
            month_title=month_title,
            prev_ym=prev_ym,
            next_ym=next_ym,
        )

    @login_required
    def requests_page():
        today = date.today()
        unit_id = dependencies.current_unit_id()
        if not unit_id:
            abort(403)
        months_ahead, _ = dependencies.unit_request_rules(unit_id)
        first_allowed, last_allowed = dependencies.request_date_bounds(today, unit_id)

        # ---- user/editor: show configured future months they can request into ----
        months = []
        base_y, base_m = today.year, today.month
        for k in range(1, months_ahead + 1):
            t_m = base_m + k
            t_y = base_y + (t_m - 1) // 12
            t_m = ((t_m - 1) % 12) + 1
            months.append((t_y, t_m))

        # ---- POST (create/delete own requests) ----
        if request.method == "POST":
            dependencies.validate_csrf()
            form = request.form.get("form", "")
            if form == "add":
                try:
                    day = date.fromisoformat(request.form.get("day", ""))
                except (TypeError, ValueError):
                    flash("Enter a valid request date.", "error")
                    return redirect(url_for("requests_page"))
                code = (request.form.get("code") or "").upper().strip()
                comment = (request.form.get("comment") or "").strip()
                if len(comment) > 500:
                    flash("Requester comments are limited to 500 characters.", "error")
                    return redirect(url_for("requests_page"))
                shift = dependencies.ShiftType.query.filter_by(
                    unit_id=unit_id,
                    code=code,
                    is_active=True,
                    is_requestable=True,
                    is_working=True,
                ).first()
                if not shift:
                    flash("That shift is inactive or cannot be requested.", "error")
                    return redirect(url_for("requests_page"))
                if day < first_allowed or day > last_allowed:
                    flash(
                        f"Requests must be between {first_allowed} and {last_allowed}.",
                        "error",
                    )
                    return redirect(url_for("requests_page"))
                if dependencies.is_month_locked(day.year, day.month, today, unit_id):
                    flash("Requests for that month are locked.", "error")
                    return redirect(url_for("requests_page"))
                ex = dependencies.ShiftRequest.query.filter_by(
                    unit_id=unit_id, staff_id=current_user.id, day=day
                ).first()
                if not ex:
                    ex = dependencies.ShiftRequest(
                        unit_id=unit_id,
                        staff_id=current_user.id,
                        day=day,
                        code=code,
                        requester_comment=comment,
                    )
                    dependencies.db.session.add(ex)
                    dependencies.db.session.flush()
                    dependencies.request_audit(
                        ex,
                        current_user.id,
                        "created",
                        {},
                        {
                            "code": code,
                            "comment": comment,
                            "status": "pending",
                        },
                    )
                else:
                    if ex.status != "pending":
                        flash("Only pending requests can be edited.", "error")
                        return redirect(url_for("requests_page"))
                    old = {
                        "code": ex.code,
                        "comment": ex.requester_comment,
                        "status": ex.status,
                    }
                    ex.code = code
                    ex.requester_comment = comment
                    ex.updated_at = dependencies.utcnow()
                    ex.submitted_at = dependencies.utcnow()
                    ex.status = "pending"
                    ex.admin_response = ""
                    ex.responded_by_id = None
                    ex.responded_at = None
                    dependencies.request_audit(
                        ex,
                        current_user.id,
                        "updated",
                        old,
                        {
                            "code": code,
                            "comment": comment,
                            "status": "pending",
                        },
                    )
                dependencies.db.session.commit()
                flash("Request saved.", "ok")
                return redirect(url_for("requests_page"))

            if form == "del":
                try:
                    rid = int(request.form.get("rid", ""))
                except (TypeError, ValueError):
                    abort(400)
                req = dependencies.ShiftRequest.query.filter_by(
                    id=rid, unit_id=unit_id
                ).first_or_404()
                if req.staff_id != current_user.id:
                    abort(403)
                if req.status != "pending":
                    abort(409, "Only pending requests can be cancelled.")
                if dependencies.is_month_locked(
                    req.day.year, req.day.month, today, unit_id
                ):
                    flash("Requests for that month are locked.", "error")
                    return redirect(url_for("requests_page"))
                old = req.status
                req.status = "cancelled"
                req.cancelled_at = dependencies.utcnow()
                req.updated_at = dependencies.utcnow()
                dependencies.request_audit(
                    req,
                    current_user.id,
                    "cancelled",
                    old,
                    req.status,
                    "Cancelled by requester",
                )
                dependencies.db.session.commit()
                flash("Request cancelled; its history has been preserved.", "ok")
                return redirect(url_for("requests_page"))
            if form == "dismiss":
                try:
                    rid = int(request.form.get("rid", ""))
                except (TypeError, ValueError):
                    abort(400)
                req = dependencies.ShiftRequest.query.filter_by(
                    id=rid, unit_id=unit_id, staff_id=current_user.id
                ).first_or_404()
                if req.status not in {"fulfilled", "rejected"}:
                    abort(409, "Only fulfilled or rejected requests can be removed.")
                if req.dismissed_by_requester_at is None:
                    req.dismissed_by_requester_at = dependencies.utcnow()
                    req.updated_at = dependencies.utcnow()
                    dependencies.request_audit(
                        req,
                        current_user.id,
                        "dismissed_by_requester",
                        {"visible_to_requester": True},
                        {"visible_to_requester": False},
                        "Removed from the requester's personal list",
                    )
                    dependencies.db.session.commit()
                flash("Completed request removed from your list.", "ok")
                return redirect(url_for("requests_page"))
            abort(400)

        # ---- My requests (everyone) ----
        my_reqs = dependencies.ShiftRequest.query.filter_by(
            unit_id=unit_id,
            staff_id=current_user.id,
            dismissed_by_requester_at=None,
        ).all()
        req_map = defaultdict(dict)
        for r in my_reqs:
            req_map[(r.day.year, r.day.month)][r.day] = r

        all_shifts = (
            dependencies.ShiftType.query.filter_by(
                unit_id=unit_id, is_active=True, is_requestable=True
            )
            .order_by(dependencies.ShiftType.code)
            .all()
        )
        codes = [s.code for s in all_shifts]

        # ---- Admin: month-selectable “All requests” panel ----
        admin_view = dependencies.is_admin_user(current_user)
        admin_grouped = {}
        admin_ym = None
        admin_month_title = None
        admin_prev_ym = None
        admin_next_ym = None
        admin_total = 0

        if admin_view:
            # default to current month unless ?ym=YYYY-MM provided
            admin_ym = dependencies.safe_request_admin_month(
                request.args.get("ym"), today
            )
            ay, am = dependencies.parse_year_month(admin_ym)
            start_of_month, month_days = dependencies.month_range(ay, am)
            end_of_month = month_days[-1]

            admin_month_title = datetime(ay, am, 1).strftime("%B %Y")
            admin_prev_ym, admin_next_ym = dependencies.clamp_prev_next(ay, am)

            # fetch only the chosen month; order by day then staff name
            admin_requests = (
                dependencies.ShiftRequest.query.join(
                    dependencies.Staff,
                    dependencies.ShiftRequest.staff_id == dependencies.Staff.id,
                )
                .filter(
                    dependencies.ShiftRequest.unit_id == unit_id,
                    dependencies.ShiftRequest.day >= start_of_month,
                    dependencies.ShiftRequest.day <= end_of_month,
                )
                .order_by(
                    dependencies.ShiftRequest.day.asc(), dependencies.Staff.name.asc()
                )
                .all()
            )

            # group by day for a tidy display
            grouped = defaultdict(list)
            for r in admin_requests:
                grouped[r.day].append(r)
            admin_grouped = dict(grouped)
            admin_total = len(admin_requests)

        return render_template(
            "requests.html",
            months=months,
            is_locked=dependencies.is_month_locked,
            req_map=req_map,
            codes=codes,
            # admin block
            admin_view=admin_view,
            admin_grouped=admin_grouped,
            admin_total=admin_total,
            admin_ym=admin_ym,
            admin_month_title=admin_month_title,
            admin_prev_ym=admin_prev_ym,
            admin_next_ym=admin_next_ym,
            request_lock_day=dependencies.unit_request_rules(unit_id)[1],
            first_allowed=first_allowed,
            last_allowed=last_allowed,
        )

    @login_required
    def admin_request_respond(rid):
        if not dependencies.is_admin_user(current_user):
            abort(403)
        dependencies.validate_csrf()
        unit_id = dependencies.current_unit_id()
        r = (
            dependencies.ShiftRequest.query.filter_by(id=rid, unit_id=unit_id)
            .with_for_update()
            .first_or_404()
        )
        dependencies.lock_roster_month(unit_id, r.day.year, r.day.month)
        action = (request.form.get("action") or "status").strip()
        if action not in {
            "approve",
            "refuse",
            "status",
            "approve_only",
            "approve_apply",
        }:
            abort(400, "Invalid request action.")
        return_month = dependencies.safe_request_admin_month(
            request.form.get("ym"), date.today()
        )
        response = (request.form.get("admin_response") or "").strip()
        if len(response) > 500:
            abort(400, "Response is limited to 500 characters.")
        requested_status = (request.form.get("status") or "").strip().lower()
        if action == "refuse":
            requested_status = "rejected"
        elif action == "approve_only":
            requested_status = "approved"
        elif action in {"approve", "approve_apply"}:
            requested_status = "fulfilled"
        if requested_status not in dependencies.request_statuses:
            abort(400, "Invalid request status.")
        if (r.status or "pending") not in dependencies.request_statuses:
            abort(409, "The request has an invalid current status.")
        if not r.staff or r.staff.unit_id != unit_id:
            abort(409, "The requester does not belong to this airport.")
        approval_actions = {"approve", "approve_only", "approve_apply"}
        is_approval = action in approval_actions or (
            action == "status" and requested_status in {"approved", "fulfilled"}
        )
        if is_approval and r.staff_id == current_user.id:
            active_admin_count = dependencies.Staff.query.filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.membership_status == "active",
                dependencies.db.or_(
                    dependencies.Staff.role == "admin",
                    dependencies.Staff.is_admin.is_(True),
                ),
            ).count()
            if active_admin_count > 1:
                abort(
                    403,
                    "Administrators cannot approve their own shift requests "
                    "while another active administrator is available.",
                )

        old = {"status": r.status, "response": r.admin_response}
        if action in {"approve", "approve_apply"}:
            if r.status not in {"pending", "approved"}:
                abort(409, "Only pending or approved requests can be applied.")
            shift = dependencies.ShiftType.query.filter_by(
                unit_id=unit_id,
                code=r.code,
                is_active=True,
                is_requestable=True,
                is_working=True,
            ).first()
            if not shift:
                abort(409, "The requested shift is no longer valid.")
            if dependencies.is_month_locked(r.day.year, r.day.month, unit_id=unit_id):
                abort(409, "The roster month is locked.")
            # The primary manager workflow is intentionally one step: approve and
            # place on the roster. Existing roster fatigue warnings remain visible
            # after the change. The legacy endpoint retains its explicit conflict
            # confirmation for backwards-compatible API clients.
            if action == "approve_apply":
                conflicts = list(
                    dependencies.would_create_new_fatigue_issues(
                        r.staff, r.day, r.code
                    ).values()
                )
                if not dependencies.staff_has_shift_qualification(
                    r.staff, shift, r.day
                ):
                    conflicts.append(["Required qualification is missing or expired."])
                override = request.form.get("confirm_override") == "yes"
                if override and not dependencies.can_override_roster_conflicts(
                    current_user
                ):
                    abort(403, "You do not have permission to override conflicts.")
                if conflicts and override and len(response) < 10:
                    abort(400, "A reason of at least 10 characters is required.")
                if conflicts and not override:
                    warning_text = "; ".join(
                        str(item)
                        for group in conflicts
                        for item in (
                            group if isinstance(group, (list, tuple, set)) else [group]
                        )
                    )
                    flash(
                        "Applying this request has conflicts: "
                        f"{warning_text[:700]}. Review and confirm the permitted override.",
                        "error",
                    )
                    return redirect(url_for("requests_page", ym=return_month))
            assignment = dependencies.Assignment.query.filter_by(
                unit_id=unit_id, staff_id=r.staff_id, day=r.day
            ).first()
            if not assignment:
                assignment = dependencies.Assignment(
                    unit_id=unit_id, staff_id=r.staff_id, day=r.day
                )
                dependencies.db.session.add(assignment)
            assignment.code = r.code
            assignment.source = "request"
            assignment.note = f"Applied from shift request #{r.id}"
            dependencies.db.session.flush()
            r.resulting_assignment_id = assignment.id
            r.fulfilled_at = dependencies.utcnow()
            requested_status = "fulfilled"
        else:
            if requested_status == "fulfilled":
                abort(400, "Fulfilment is only available through Approve and apply.")
            allowed = dependencies.request_transitions.get(
                r.status or "pending", frozenset()
            )
            if requested_status not in allowed:
                abort(
                    409,
                    f"Transition from {r.status or 'pending'} to "
                    f"{requested_status} is not permitted.",
                )
            if (
                action != "refuse"
                and r.status == "approved"
                and requested_status in {"rejected", "cancelled"}
                and len(response) < 10
            ):
                abort(400, "Changing an approved request requires an audited reason.")

        r.admin_response = response
        r.status = requested_status
        r.responded_by_id = getattr(current_user, "id", None)
        r.responded_at = dependencies.utcnow()
        r.updated_at = dependencies.utcnow()
        if requested_status == "cancelled" and r.cancelled_at is None:
            r.cancelled_at = dependencies.utcnow()
        dependencies.request_audit(
            r,
            current_user.id,
            action,
            old,
            {
                "status": r.status,
                "response": response,
                "assignment_id": r.resulting_assignment_id,
            },
            response,
        )
        if old["status"] != r.status:
            dependencies.notify_requester(r)
        dependencies.db.session.commit()
        if requested_status == "fulfilled":
            flash("Request approved and added to the roster.", "ok")
        elif requested_status == "rejected":
            flash("Request refused. The ATCO has been notified.", "ok")
        else:
            flash("Response saved.", "ok")
        return redirect(url_for("requests_page", ym=return_month))

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/leave", "leave", leave, ["GET", "POST"]),
            ("/requests", "requests_page", requests_page, ["GET", "POST"]),
            (
                "/admin/requests/<int:rid>/respond",
                "admin_request_respond",
                admin_request_respond,
                ["POST"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
