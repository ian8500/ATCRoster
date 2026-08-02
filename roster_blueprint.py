"""Roster routes extracted incrementally from the legacy application module."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError
from reporting import csv_safe_cell


@dataclass(frozen=True)
class RosterDependencies:
    db: Any
    RosterPublication: Any
    Staff: Any
    Notification: Any
    Assignment: Any
    Leave: Any
    Watch: Any
    Requirement: Any
    SpecialRequirement: Any
    ShiftRequest: Any
    AnnotationType: Any
    AnnotationAudit: Any
    can_publish_roster: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    parse_year_month: Callable[[str], tuple[int, int]]
    current_unit_id: Callable[[], int]
    month_has_data: Callable[[int, int], bool]
    ensure_month_requirement: Callable[[int, int], Any]
    generate_month: Callable[[int, int], None]
    active_publication: Callable[[int, int], Any]
    publication_matches_live: Callable[[Any, int, int], bool]
    roster_snapshot: Callable[[int, int], dict]
    utcnow: Callable[[], Any]
    send_publication_emails: Callable[[int, int, int, Any], tuple[int, int, int]]
    log_change: Callable[..., None]
    consume_rate_limit: Callable[..., bool]
    month_range: Callable[[int, int], tuple[date, list[date]]]
    requirements_for_day: Callable[..., dict[str, int]]
    staff_is_countable_on: Callable[[Any, date], bool]
    exclude_from_counters: Callable[[], set[str]]
    get_shift: Callable[[str], Any]
    shift_counter_group_for_day: Callable[[str, date, int], str | None]
    night_active_on: Callable[[int, date], bool]
    can_edit_roster: Callable[[Any], bool]
    banned_roster_codes: Callable[[], set[str]]
    can_apply_annotations: Callable[[Any], bool]
    parse_annotation: Callable[[str], dict | None]
    is_admin_user: Callable[[Any], bool]
    apply_toil_annotation_delta: Callable[..., None]
    load_month_roster: Callable[[int, int, int], tuple]
    add_months: Callable[[int, int, int], tuple[int, int]]
    shift_groups: Callable[[int], tuple]
    watch_ids_for_staff_on: Callable[[list[Any], date], dict[int, int | None]]
    roster_fatigue_flags: Callable[..., dict]
    roster_validation: Any
    get_annotation_groups: Callable[[], list]
    lock_roster_month: Callable[[int, int, int], Any]


def create_roster_blueprint(dependencies: RosterDependencies) -> Blueprint:
    blueprint = Blueprint("roster", __name__)

    def annotation_is_soal(value: str | None) -> bool:
        parsed = dependencies.parse_annotation((value or "").strip().upper())
        return bool(parsed and parsed.get("type") == "SOAL")

    @login_required
    def roster_month_publish(ym):
        if not dependencies.can_publish_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        if not dependencies.month_has_data(year, month):
            dependencies.ensure_month_requirement(year, month)
            dependencies.generate_month(year, month)
        # Serialise roster writes before evaluating publication eligibility so
        # the validated assignments are the same assignments we snapshot.
        dependencies.lock_roster_month(unit_id, year, month)
        month_start, month_days = dependencies.month_range(year, month)
        validation = dependencies.roster_validation.validate_range(
            unit_id, month_start, month_days[-1]
        )
        if not validation.can_publish:
            flash(
                f"Roster publication blocked: {validation.blocking_count} "
                "pattern or hard-rule breach"
                f"{'es' if validation.blocking_count != 1 else ''} must be resolved.",
                "error",
            )
            return redirect(url_for("roster_month", ym=ym))
        active = dependencies.active_publication(year, month)
        if active and dependencies.publication_matches_live(active, year, month):
            flash(
                f"The {date(year, month, 1).strftime('%B %Y')} roster is already published.",
                "info",
            )
            return redirect(url_for("roster_month", ym=ym))

        published_at = dependencies.utcnow()
        latest_version = (
            dependencies.db.session.query(
                dependencies.db.func.max(dependencies.RosterPublication.version)
            )
            .filter(
                dependencies.RosterPublication.unit_id == unit_id,
                dependencies.RosterPublication.year == year,
                dependencies.RosterPublication.month == month,
            )
            .scalar()
            or 0
        )
        if active:
            active.state = "superseded"
            active.superseded_at = published_at
        snapshot = dependencies.roster_snapshot(year, month)
        snapshot["published_by"] = {
            "id": current_user.id,
            "name": current_user.name,
            "published_at": published_at.isoformat(),
        }
        publication = dependencies.RosterPublication(
            unit_id=unit_id,
            year=year,
            month=month,
            version=latest_version + 1,
            state="published",
            snapshot_json=json.dumps(snapshot),
            published_at=published_at,
        )
        dependencies.db.session.add(publication)
        for person in dependencies.Staff.query.filter_by(
            unit_id=unit_id,
            is_operational=True,
            membership_status="active",
        ).all():
            if person.id != current_user.id:
                dependencies.db.session.add(
                    dependencies.Notification(
                        unit_id=unit_id,
                        recipient_id=person.id,
                        kind="roster_published",
                        message=(
                            f"The {date(year, month, 1).strftime('%B %Y')} roster "
                            f"was published on {published_at.strftime('%d %B %Y')}."
                        ),
                    )
                )
        dependencies.db.session.commit()
        email_sent, email_failed, email_recipients = (
            dependencies.send_publication_emails(unit_id, year, month, published_at)
        )
        if email_failed:
            current_app.logger.warning(
                "roster_publication_email_delivery_incomplete "
                "unit_id=%s year=%s month=%s sent=%s failed=%s",
                unit_id,
                year,
                month,
                email_sent,
                email_failed,
            )
        dependencies.log_change(
            "RosterPublication",
            publication.id,
            "state",
            "draft",
            "published",
            note=(
                f"Published directly from the monthly roster by {current_user.name}."
            ),
            context_day=date(year, month, 1),
        )
        message = f"{date(year, month, 1).strftime('%B %Y')} roster published."
        if email_recipients:
            message += (
                f" Email sent to {email_sent} registered "
                f"user{'s' if email_sent != 1 else ''}."
            )
            if email_failed:
                message += (
                    f" {email_failed} email"
                    f"{'s' if email_failed != 1 else ''} could not be delivered."
                )
        else:
            message += " No registered users have an email address."
        flash(message, "ok" if not email_failed else "warning")
        return redirect(url_for("roster_month", ym=ym))

    @login_required
    def roster_month_unpublish(ym):
        if not dependencies.can_publish_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        dependencies.lock_roster_month(unit_id, year, month)
        publication = dependencies.active_publication(year, month)
        if not publication:
            flash("This roster is already in Draft.", "info")
            return redirect(url_for("roster_month", ym=ym))
        publication.state = "withdrawn"
        publication.superseded_at = dependencies.utcnow()
        for person in dependencies.Staff.query.filter_by(
            unit_id=unit_id,
            is_operational=True,
            membership_status="active",
        ).all():
            if person.id != current_user.id:
                dependencies.db.session.add(
                    dependencies.Notification(
                        unit_id=unit_id,
                        recipient_id=person.id,
                        kind="roster_unpublished",
                        message=(
                            f"The {date(year, month, 1).strftime('%B %Y')} roster "
                            "has returned to Draft and may be subject to change."
                        ),
                    )
                )
        dependencies.db.session.commit()
        dependencies.log_change(
            "RosterPublication",
            publication.id,
            "state",
            "published",
            "withdrawn",
            note=(
                f"Publication undone by {current_user.name}; roster returned to Draft."
            ),
            context_day=date(year, month, 1),
        )
        flash(
            f"{date(year, month, 1).strftime('%B %Y')} roster returned to Draft.",
            "ok",
        )
        return redirect(url_for("roster_month", ym=ym))

    @login_required
    def roster_export_csv(ym):
        if not dependencies.consume_rate_limit(
            "roster-export",
            current_user.id,
            limit=30,
            window=timedelta(hours=1),
        ):
            abort(429)
        year, month = dependencies.parse_year_month(ym)
        start, days = dependencies.month_range(year, month)
        staff = (
            dependencies.Staff.query.outerjoin(
                dependencies.Watch,
                dependencies.Staff.watch_id == dependencies.Watch.id,
            )
            .filter(dependencies.Staff.role != "position_monitor")
            .order_by(dependencies.Watch.order_index, dependencies.Staff.name)
            .all()
        )
        assignment_map = defaultdict(dict)
        month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)
        for assignment in dependencies.Assignment.query.filter(
            dependencies.Assignment.day >= start,
            dependencies.Assignment.day < month_end,
        ):
            assignment_map[assignment.staff_id][assignment.day] = assignment.code
        requirement = dependencies.Requirement.query.filter_by(
            year=year, month=month
        ).first()
        special_by_day = {
            row.day: row
            for row in dependencies.SpecialRequirement.query.filter(
                dependencies.SpecialRequirement.day >= start,
                dependencies.SpecialRequirement.day < month_end,
            ).all()
        }
        requirements = {
            day: dependencies.requirements_for_day(
                requirement, day, special_by_day.get(day)
            )
            for day in days
        }
        counters = {day: Counter() for day in days}
        excluded = dependencies.exclude_from_counters()
        unit_id = dependencies.current_unit_id()
        for person in staff:
            if not person.is_operational:
                continue
            for day in days:
                if not dependencies.staff_is_countable_on(person, day):
                    continue
                code = assignment_map[person.id].get(day)
                if not code or code in excluded:
                    continue
                shift = dependencies.get_shift(code)
                if not shift or shift.is_training:
                    continue
                group = dependencies.shift_counter_group_for_day(code, day, unit_id)
                if group:
                    counters[day][group] += 1
        rag = {}
        for day in days:
            rag[day] = {}
            for code in ("M", "D", "A", "N"):
                available = counters[day][code]
                needed = (
                    0
                    if code == "N" and not dependencies.night_active_on(unit_id, day)
                    else requirements[day][code]
                )
                rag[day][code] = (
                    "green"
                    if available >= needed
                    else ("amber" if available >= max(0, needed - 1) else "red")
                )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["Name", "Staff #", "Watch"] + [day.isoformat() for day in days]
        )
        for person in staff:
            writer.writerow(
                csv_safe_cell(value)
                for value in [
                    person.name,
                    person.staff_no,
                    person.watch.name.replace("Watch ", "") if person.watch else "-",
                    *[assignment_map[person.id].get(day, "") for day in days],
                ]
            )
        writer.writerow([])
        writer.writerow(
            ["Totals (M/D/A/N)", "", ""]
            + [
                f"M:{counters[day]['M']}/{requirements[day]['M']}-{rag[day]['M']} | "
                f"D:{counters[day]['D']}/{requirements[day]['D']}-{rag[day]['D']} | "
                f"A:{counters[day]['A']}/{requirements[day]['A']}-{rag[day]['A']} | "
                f"N:{counters[day]['N']}/{requirements[day]['N']}-{rag[day]['N']}"
                for day in days
            ]
        )
        return Response(
            output.getvalue().encode("utf-8"),
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=roster_{year:04d}-{month:02d}.csv"
                )
            },
        )

    @login_required
    def roster_print_view(ym):
        return redirect(url_for("roster_month", ym=ym))

    @login_required
    def roster_month(ym):
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        if not dependencies.month_has_data(year, month):
            dependencies.ensure_month_requirement(year, month)
            dependencies.generate_month(year, month)

        days, staff, assignment_tuples, requirement = dependencies.load_month_roster(
            unit_id, year, month
        )
        assignment_map: dict[int, dict[date, str]] = {}
        assignment_version_map: dict[tuple[int, date], int] = {
            (assignment.staff_id, assignment.day): assignment.version
            for assignment in dependencies.Assignment.query.filter(
                dependencies.Assignment.unit_id == unit_id,
                dependencies.Assignment.day >= date(year, month, 1),
                dependencies.Assignment.day
                < date(*dependencies.add_months(year, month, 1), 1),
            ).all()
        }
        annotation_map: dict[int, dict[date, str]] = {}
        annotation_note_map: dict[int, dict[date, str]] = {}
        for staff_id, day_map in assignment_tuples.items():
            assignment_map[staff_id] = {}
            annotation_map[staff_id] = {}
            annotation_note_map[staff_id] = {}
            for duty_day, (code, _source, annotation, note) in day_map.items():
                assignment_map[staff_id][duty_day] = code
                annotation_map[staff_id][duty_day] = annotation or ""
                annotation_note_map[staff_id][duty_day] = note or ""

        previous_year, previous_month = dependencies.add_months(year, month, -1)
        next_year, next_month = dependencies.add_months(year, month, 1)
        previous_ym = f"{previous_year:04d}-{previous_month:02d}"
        next_ym = f"{next_year:04d}-{next_month:02d}"
        start = date(year, month, 1)
        month_end = date(next_year, next_month, 1)
        for leave in dependencies.Leave.query.filter(
            dependencies.Leave.unit_id == unit_id,
            dependencies.Leave.leave_type == "AL",
            dependencies.Leave.start < month_end,
            dependencies.Leave.end >= start,
        ).all():
            leave_day = max(start, leave.start)
            final_day = min(month_end - timedelta(days=1), leave.end)
            while leave_day <= final_day:
                if not annotation_is_soal(
                    annotation_map.get(leave.staff_id, {}).get(leave_day)
                ):
                    assignment_map.setdefault(leave.staff_id, {})[leave_day] = "AL"
                leave_day += timedelta(days=1)
        working_shifts, training_shifts, nonworking_shifts = dependencies.shift_groups(
            unit_id
        )
        training_codes = {shift.code for shift in training_shifts}
        display_watch_by_staff = dependencies.watch_ids_for_staff_on(staff, start)
        try:
            watch_order = {
                watch.id: watch.order_index for watch in dependencies.Watch.query.all()
            }
        except Exception:
            watch_order = {}

        def rank_within_watch(person):
            if getattr(person, "is_wm", False):
                return 0
            if getattr(person, "is_dwm", False):
                return 1
            return 2

        staff.sort(
            key=lambda person: (
                watch_order.get(display_watch_by_staff.get(person.id), 9999),
                rank_within_watch(person),
                person.name,
            )
        )
        counters = {duty_day: Counter() for duty_day in days}
        excluded = dependencies.exclude_from_counters()
        for person in staff:
            if not getattr(person, "is_operational", True):
                continue
            row = assignment_map.get(person.id, {})
            for duty_day in days:
                if not dependencies.staff_is_countable_on(person, duty_day):
                    continue
                code = (row.get(duty_day) or "").upper()
                if (
                    not code
                    or code in excluded
                    or code in training_codes
                    or code in ("AL", "NOPS")
                ):
                    continue
                group = dependencies.shift_counter_group_for_day(
                    code, duty_day, unit_id
                )
                if group:
                    counters[duty_day][group] += 1

        night_active = {
            duty_day: dependencies.night_active_on(unit_id, duty_day)
            for duty_day in days
        }
        special_requirements = (
            dependencies.SpecialRequirement.query.filter(
                dependencies.SpecialRequirement.day >= start,
                dependencies.SpecialRequirement.day < month_end,
            )
            .order_by(dependencies.SpecialRequirement.day)
            .all()
        )
        special_by_day = {row.day: row for row in special_requirements}
        requirements = {
            duty_day: dependencies.requirements_for_day(
                requirement, duty_day, special_by_day.get(duty_day)
            )
            for duty_day in days
        }
        rag = {}
        for duty_day in days:
            rag[duty_day] = {}
            for code in ("M", "D", "A", "N"):
                available = counters[duty_day][code]
                needed = (
                    0
                    if code == "N" and not night_active[duty_day]
                    else requirements[duty_day][code]
                )
                rag[duty_day][code] = (
                    "green"
                    if available >= needed
                    else ("amber" if available >= max(0, needed - 1) else "red")
                )
        fatigue = {
            person.id: dependencies.roster_fatigue_flags(
                person,
                days,
                assignment_map.get(person.id, {}),
                unit_id,
            )
            for person in staff
        }
        roster_validation = dependencies.roster_validation.validate_range(
            unit_id, days[0], days[-1]
        )
        requests = dependencies.ShiftRequest.query.filter(
            dependencies.ShiftRequest.unit_id == unit_id,
            dependencies.ShiftRequest.day >= start,
            dependencies.ShiftRequest.day < month_end,
        ).all()
        pending_requests = {
            (item.staff_id, item.day): {
                "code": item.code,
                "status": (item.status or "pending").lower(),
            }
            for item in requests
            if (item.status or "pending").lower() in {"pending", "approved"}
        }
        applied_requests = {
            (staff_id, request_day): {"code": request_code}
            for staff_id, request_day, request_code in (
                dependencies.db.session.query(
                    dependencies.ShiftRequest.staff_id,
                    dependencies.ShiftRequest.day,
                    dependencies.ShiftRequest.code,
                )
                .join(
                    dependencies.Assignment,
                    dependencies.ShiftRequest.resulting_assignment_id
                    == dependencies.Assignment.id,
                )
                .filter(
                    dependencies.ShiftRequest.unit_id == unit_id,
                    dependencies.ShiftRequest.status == "fulfilled",
                    dependencies.ShiftRequest.day >= start,
                    dependencies.ShiftRequest.day < month_end,
                    dependencies.Assignment.code == dependencies.ShiftRequest.code,
                )
                .all()
            )
        }
        today = date.today()

        def expiry_class(expiry: date | None, under_training=False):
            if under_training:
                return "exp-amber"
            if not expiry:
                return ""
            remaining = (expiry - today).days
            if remaining < 0:
                return "exp-red"
            if remaining <= 90:
                return "exp-amber"
            return "exp-green"

        expiry_classes = {
            person.id: {
                "medical": expiry_class(person.medical_expiry),
                "tower": expiry_class(person.tower_ue_expiry, person.tower_ut),
                "radar": expiry_class(person.radar_ue_expiry, person.radar_ut),
                "met": expiry_class(person.met_ue_expiry, person.met_ut),
            }
            for person in staff
        }
        watch_break_after_ids = []
        previous_watch = None
        previous_id = None
        for person in staff:
            current_watch = display_watch_by_staff.get(person.id)
            if (
                previous_watch is not None
                and current_watch != previous_watch
                and previous_id is not None
            ):
                watch_break_after_ids.append(previous_id)
            previous_watch = current_watch
            previous_id = person.id
        active_publication = dependencies.active_publication(year, month)
        roster_publication = (
            active_publication
            if dependencies.publication_matches_live(active_publication, year, month)
            else None
        )
        return render_template(
            "roster_month.html",
            ym=ym,
            year=year,
            month=month,
            days=days,
            staff=staff,
            a_map=assignment_map,
            assignment_version_map=assignment_version_map,
            ann_map=annotation_map,
            ann_note_map=annotation_note_map,
            req_by_day=requirements,
            special_requirements=special_requirements,
            counters=counters,
            req=requirement,
            requirement=requirement,
            rag=rag,
            expiry_classes=expiry_classes,
            fatigue=fatigue,
            roster_validation=roster_validation,
            roster_validation_by_cell=roster_validation.by_cell(),
            watch_break_after_ids=watch_break_after_ids,
            prev_ym=previous_ym,
            next_ym=next_ym,
            shifts_working=working_shifts,
            shifts_training=training_shifts,
            shifts_non=nonworking_shifts,
            can_edit=dependencies.can_edit_roster(current_user),
            readonly=False,
            month_title=datetime(year, month, 1).strftime("%B %Y"),
            today=today,
            req_pending_map=pending_requests,
            applied_request_map=applied_requests,
            show_ot_finder=True,
            display_watch_by_staff=display_watch_by_staff,
            annotation_groups=dependencies.get_annotation_groups(),
            night_active=night_active,
            roster_publication=roster_publication,
            can_publish_roster=dependencies.can_publish_roster(current_user),
        )

    @login_required
    def assign_cell(staff_id, ym, day):
        if not dependencies.can_edit_roster(current_user):
            return "Forbidden", 403
        dependencies.validate_csrf()
        try:
            duty_day = date.fromisoformat(day)
            year, month = dependencies.parse_year_month(ym)
            if duty_day.year != year or duty_day.month != month:
                raise ValueError
        except (TypeError, ValueError):
            abort(400, "Invalid roster date.")
        unit_id = dependencies.current_unit_id()

        def edit_error(message: str, status: int = 422):
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(ok=False, error=message), status
            flash(message, "error")
            return redirect(url_for("roster_month", ym=ym))

        def updated_day_summary() -> dict[str, Any]:
            people = dependencies.Staff.query.filter_by(unit_id=unit_id).all()
            assignments = dependencies.Assignment.query.filter_by(
                unit_id=unit_id, day=duty_day
            ).all()
            codes = {item.staff_id: (item.code or "").upper() for item in assignments}
            excluded = dependencies.exclude_from_counters()
            _, training, _ = dependencies.shift_groups(unit_id)
            training_codes = {shift.code for shift in training}
            counts = Counter({group: 0 for group in ("M", "D", "A", "N")})
            for member in people:
                if not getattr(member, "is_operational", True):
                    continue
                if not dependencies.staff_is_countable_on(member, duty_day):
                    continue
                member_code = codes.get(member.id, "")
                if (
                    not member_code
                    or member_code in excluded
                    or member_code in training_codes
                    or member_code in {"AL", "NOPS"}
                ):
                    continue
                group = dependencies.shift_counter_group_for_day(
                    member_code, duty_day, unit_id
                )
                if group:
                    counts[group] += 1
            requirement = dependencies.ensure_month_requirement(year, month)
            special = dependencies.SpecialRequirement.query.filter_by(
                unit_id=unit_id, day=duty_day
            ).first()
            required = dependencies.requirements_for_day(requirement, duty_day, special)
            night_active = dependencies.night_active_on(unit_id, duty_day)
            rag = {}
            for group in ("M", "D", "A", "N"):
                needed = 0 if group == "N" and not night_active else required[group]
                available = counts[group]
                rag[group] = (
                    "green"
                    if available >= needed
                    else ("amber" if available >= max(0, needed - 1) else "red")
                )
            return {
                "counts": dict(counts),
                "required": required,
                "rag": rag,
                "night_active": night_active,
                "total": sum(counts[group] for group in ("M", "D", "A"))
                + (counts["N"] if night_active else 0),
            }

        dependencies.lock_roster_month(unit_id, year, month)
        person = dependencies.Staff.query.filter_by(
            id=staff_id, unit_id=unit_id
        ).first_or_404()
        annual_leave = dependencies.Leave.query.filter(
            dependencies.Leave.unit_id == unit_id,
            dependencies.Leave.staff_id == staff_id,
            dependencies.Leave.leave_type == "AL",
            dependencies.Leave.start <= duty_day,
            dependencies.Leave.end >= duty_day,
        ).first()
        assignment = (
            dependencies.Assignment.query.filter_by(
                unit_id=unit_id, staff_id=staff_id, day=duty_day
            )
            .with_for_update()
            .first()
        )
        current_version = assignment.version if assignment else 0
        raw_version = request.form.get("assignment_version")
        try:
            submitted_version = (
                current_version if raw_version is None else int(raw_version)
            )
        except (TypeError, ValueError):
            abort(400, "Invalid roster cell version.")
        if submitted_version != current_version:
            abort(409, "This roster cell changed after the page was loaded.")
        if assignment is None:
            assignment = dependencies.Assignment(
                unit_id=unit_id,
                staff=person,
                day=duty_day,
                code="OFF",
            )
            dependencies.db.session.add(assignment)

        code = (request.form.get("code") or "").strip().upper()
        annotation = request.form.get("annotation")
        if code:
            if annual_leave and not annotation_is_soal(assignment.annotation):
                return edit_error(
                    "This annual-leave cell is locked. Apply the SOAL annotation "
                    "before entering a shift, or amend the leave in Leave Administration."
                )
            if code in dependencies.banned_roster_codes():
                return edit_error(
                    "Leave, sickness and TOIL use must be logged via the form, "
                    "not the roster grid."
                )
            if not dependencies.get_shift(code):
                return edit_error(f"Unknown shift code '{code}'")
            assignment.code = code
            assignment.source = "manual"

        if annotation is not None:
            if not dependencies.can_apply_annotations(current_user):
                abort(403)
            old_value = assignment.annotation or ""
            old_note = assignment.annotation_note or ""
            new_value = (annotation or "").strip().upper()
            if new_value == "__REMOVE__":
                new_value = ""
            note_was_posted = "annotation_detail_update" in request.form
            annotation_note = (request.form.get("annotation_note") or "").strip()[:140]
            parsed = dependencies.parse_annotation(new_value) if new_value else None
            definition = None
            if parsed:
                definition = dependencies.AnnotationType.query.filter_by(
                    unit_id=unit_id, code=parsed["type"]
                ).first()
            if new_value and (not parsed or not definition):
                return edit_error(f"Unknown annotation '{new_value}'.")
            if definition and not definition.is_active and old_value != new_value:
                return edit_error(
                    f"{definition.code} is inactive and cannot be newly applied."
                )
            if (
                definition
                and definition.admin_only
                and not dependencies.is_admin_user(current_user)
            ):
                abort(403)
            if definition and definition.note_required and not annotation_note:
                return edit_error(f"{definition.code} requires a note.")
            if old_value != new_value:
                transaction_key = (request.form.get("transaction_key") or "").strip()[
                    :64
                ]
                if (
                    transaction_key
                    and dependencies.AnnotationAudit.query.filter_by(
                        unit_id=unit_id, transaction_key=transaction_key
                    ).first()
                ):
                    return redirect(url_for("roster_month", ym=ym))
                dependencies.db.session.flush()
                dependencies.apply_toil_annotation_delta(
                    staff=person,
                    old_annot=old_value,
                    new_annot=new_value,
                    actor_id=current_user.id,
                    transaction_key=transaction_key or None,
                    source_id=assignment.id,
                )
                assignment.annotation = new_value
                assignment.annotation_note = annotation_note if new_value else ""
                if definition:
                    definition.has_been_used = True
                dependencies.db.session.flush()
                dependencies.db.session.add(
                    dependencies.AnnotationAudit(
                        unit_id=unit_id,
                        annotation_type_id=definition.id if definition else None,
                        assignment_id=assignment.id,
                        actor_id=current_user.id,
                        action="applied" if new_value else "removed",
                        old_value=old_value,
                        new_value=new_value,
                        transaction_key=transaction_key or None,
                    )
                )
            elif note_was_posted:
                assignment.annotation_note = annotation_note
                if old_note != annotation_note:
                    dependencies.db.session.flush()
                    dependencies.db.session.add(
                        dependencies.AnnotationAudit(
                            unit_id=unit_id,
                            annotation_type_id=definition.id if definition else None,
                            assignment_id=assignment.id,
                            actor_id=current_user.id,
                            action="detail_updated",
                            old_value=old_note,
                            new_value=annotation_note,
                        )
                    )
            if annual_leave and not annotation_is_soal(new_value):
                assignment.code = "AL"
                assignment.source = "leave"
                assignment.note = "annual leave"
        assignment.version = current_version + 1
        try:
            dependencies.db.session.commit()
        except IntegrityError:
            dependencies.db.session.rollback()
            abort(409, "This roster cell changed concurrently.")
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            saved_shift = dependencies.get_shift(assignment.code)
            return jsonify(
                ok=True,
                staff_id=staff_id,
                day=duty_day.isoformat(),
                code=assignment.code,
                annotation=assignment.annotation or "",
                annotation_note=assignment.annotation_note or "",
                version=assignment.version,
                is_training=bool(saved_shift and saved_shift.is_training),
                day_summary=updated_day_summary(),
            )
        return redirect(url_for("roster_month", ym=ym))

    @blueprint.record_once
    def register_routes(state):
        routes = (
            (
                "/roster/<ym>/publish",
                "roster_month_publish",
                roster_month_publish,
                ["POST"],
            ),
            (
                "/roster/<ym>/unpublish",
                "roster_month_unpublish",
                roster_month_unpublish,
                ["POST"],
            ),
            ("/roster/<ym>", "roster_month", roster_month, ["GET"]),
            (
                "/assign/<int:staff_id>/<ym>/<day>",
                "assign_cell",
                assign_cell,
                ["POST"],
            ),
            ("/roster/<ym>/export", "roster_export_csv", roster_export_csv, ["GET"]),
            ("/roster/<ym>/print", "roster_print_view", roster_print_view, ["GET"]),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
