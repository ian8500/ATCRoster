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
from atcroster.roster.month_view import MonthDisplayDependencies, RosterMonthViewService
from reporting import csv_safe_cell


@dataclass(frozen=True)
class RosterDependencies:
    db: Any
    RosterPublication: Any
    RosterAcknowledgement: Any
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
    publication_service: Any
    validate_csrf: Callable[[], None]
    parse_year_month: Callable[[str], tuple[int, int]]
    current_unit_id: Callable[[], int]
    roster_month_service: Any
    assignment_runtime: Any
    utcnow: Callable[[], Any]
    log_change: Callable[..., None]
    stage_change: Callable[..., None]
    consume_rate_limit: Callable[..., bool]
    requirements_for_day: Callable[..., dict[str, int]]
    staff_is_countable_on: Callable[[Any, date], bool]
    operational_capability_matrix: Callable[[list[Any], list[date]], dict]
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
    roster_fatigue_matrix: Callable[..., dict]
    roster_validation: Any
    roster_month_cache: Any
    metrics: Any
    get_annotation_groups: Callable[[], list]
    RosterProposal: Any = None
    RosterProposalAssignment: Any = None
    roster_proposal_service: Any = None


def create_roster_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> RosterDependencies:
    """Bind roster route records at the roster composition boundary."""
    return RosterDependencies(
        db=db,
        RosterPublication=saas_models.RosterPublication,
        RosterAcknowledgement=saas_models.RosterAcknowledgement,
        Staff=operational_models.Staff,
        Notification=operational_models.Notification,
        Assignment=operational_models.Assignment,
        Leave=operational_models.Leave,
        Watch=operational_models.Watch,
        Requirement=operational_models.Requirement,
        SpecialRequirement=operational_models.SpecialRequirement,
        ShiftRequest=operational_models.ShiftRequest,
        AnnotationType=operational_models.AnnotationType,
        AnnotationAudit=operational_models.AnnotationAudit,
        RosterProposal=saas_models.RosterProposal,
        RosterProposalAssignment=saas_models.RosterProposalAssignment,
        **services,
    )


def create_roster_blueprint(dependencies: RosterDependencies) -> Blueprint:
    blueprint = Blueprint("roster", __name__)

    def annotation_is_soal(value: str | None) -> bool:
        parsed = dependencies.parse_annotation((value or "").strip().upper())
        return bool(parsed and parsed.get("type") == "SOAL")

    def _proposal_or_404(proposal_id: int):
        return dependencies.RosterProposal.query.filter_by(
            id=proposal_id, unit_id=dependencies.current_unit_id()
        ).first_or_404()

    @login_required
    def roster_proposals():
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        unit_id = dependencies.current_unit_id()
        if request.method == "POST":
            dependencies.validate_csrf()
            try:
                start_date = date.fromisoformat(request.form.get("start_date") or "")
                end_date = date.fromisoformat(request.form.get("end_date") or "")
                lookback = max(1, min(730, int(
                    request.form.get("fairness_lookback_days") or 180
                )))
                proposal = dependencies.roster_proposal_service.generate(
                    unit_id,
                    start_date,
                    end_date,
                    current_user.id,
                    allow_overtime=request.form.get("allow_overtime") == "1",
                    fairness_lookback_days=lookback,
                )
            except (TypeError, ValueError) as exc:
                flash(str(exc) or "Choose valid proposal dates.", "error")
            else:
                flash("Automatic allocation proposal generated for review.", "ok")
                return redirect(url_for("roster_proposal_detail", proposal_id=proposal.id))
        rows = dependencies.RosterProposal.query.filter_by(
            unit_id=unit_id
        ).order_by(dependencies.RosterProposal.created_at.desc()).limit(50).all()
        return render_template(
            "roster_proposals.html", proposals=rows, today=date.today()
        )

    @login_required
    def roster_proposal_detail(proposal_id: int):
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        proposal = _proposal_or_404(proposal_id)
        items = dependencies.RosterProposalAssignment.query.filter_by(
            unit_id=proposal.unit_id, proposal_id=proposal.id
        ).order_by(
            dependencies.RosterProposalAssignment.day,
            dependencies.RosterProposalAssignment.shift_code,
        ).all()
        staff = {
            row.id: row for row in dependencies.Staff.query.filter(
                dependencies.Staff.unit_id == proposal.unit_id,
                dependencies.Staff.id.in_([item.staff_id for item in items] or [0]),
            ).all()
        }
        uncovered = json.loads(proposal.uncovered_json or "[]")
        return render_template(
            "roster_proposal_detail.html",
            proposal=proposal,
            items=items,
            staff_by_id=staff,
            uncovered=uncovered,
            json_loads=json.loads,
        )

    @login_required
    def roster_proposal_review(proposal_id: int, item_id: int):
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        proposal = _proposal_or_404(proposal_id)
        if proposal.workflow_state != "draft":
            abort(409, "This proposal is no longer open for review.")
        item = dependencies.RosterProposalAssignment.query.filter_by(
            id=item_id, proposal_id=proposal.id, unit_id=proposal.unit_id
        ).first_or_404()
        state = request.form.get("review_state") or ""
        if state not in {"accepted", "rejected", "pending"}:
            abort(400)
        item.review_state = state
        item.reviewed_by_user_id = current_user.id
        item.reviewed_at = dependencies.utcnow()
        dependencies.db.session.commit()
        return redirect(url_for("roster_proposal_detail", proposal_id=proposal.id))

    @login_required
    def roster_proposal_apply(proposal_id: int):
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        proposal = _proposal_or_404(proposal_id)
        try:
            applied = dependencies.roster_proposal_service.apply(
                proposal, current_user.id
            )
        except ValueError as exc:
            dependencies.db.session.rollback()
            flash(str(exc), "error")
        else:
            flash(f"Applied {applied} accepted proposed duties.", "ok")
        return redirect(url_for("roster_proposal_detail", proposal_id=proposal.id))

    @login_required
    def roster_proposal_discard(proposal_id: int):
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        proposal = _proposal_or_404(proposal_id)
        if proposal.workflow_state != "draft":
            abort(409)
        proposal.workflow_state = "discarded"
        proposal.discarded_by_user_id = current_user.id
        proposal.discarded_at = dependencies.utcnow()
        dependencies.db.session.commit()
        flash("Proposal discarded; the live roster was not changed.", "ok")
        return redirect(url_for("roster_proposal_detail", proposal_id=proposal.id))

    @login_required
    def roster_month_publish(ym):
        if not dependencies.publication_service.can_publish(current_user):
            abort(403)
        dependencies.validate_csrf()
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        if not dependencies.roster_month_service.has_data(year, month):
            dependencies.assignment_runtime.ensure_month_requirement(year, month)
            dependencies.assignment_runtime.generate_month(year, month)
        # Serialise roster writes before evaluating publication eligibility so
        # the validated assignments are the same assignments we snapshot.
        dependencies.roster_month_service.lock(unit_id, year, month)
        month_start, month_days = dependencies.roster_month_service.range(year, month)
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
        active = dependencies.publication_service.active_publication(year, month)
        if active and dependencies.publication_service.matches_live(active, year, month):
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
        snapshot = dependencies.publication_service.snapshot(year, month)
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
            dependencies.publication_service.send_emails(
                unit_id, year, month, published_at
            )
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
        if not dependencies.publication_service.can_publish(current_user):
            abort(403)
        dependencies.validate_csrf()
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        dependencies.roster_month_service.lock(unit_id, year, month)
        publication = dependencies.publication_service.active_publication(year, month)
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
        unit_id = dependencies.current_unit_id()
        start, days = dependencies.roster_month_service.range(year, month)
        staff = (
            dependencies.Staff.query.outerjoin(
                dependencies.Watch,
                dependencies.Staff.watch_id == dependencies.Watch.id,
            )
            .filter(
                dependencies.Staff.unit_id == unit_id,
                dependencies.Staff.role != "position_monitor",
            )
            .order_by(dependencies.Watch.order_index, dependencies.Staff.name)
            .all()
        )
        assignment_map = defaultdict(dict)
        month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)
        for assignment in dependencies.Assignment.query.filter(
            dependencies.Assignment.unit_id == unit_id,
            dependencies.Assignment.day >= start,
            dependencies.Assignment.day < month_end,
        ):
            assignment_map[assignment.staff_id][assignment.day] = (
                assignment.effective_code
            )
        requirement = dependencies.Requirement.query.filter_by(
            unit_id=unit_id, year=year, month=month
        ).first()
        special_by_day = {
            row.day: row
            for row in dependencies.SpecialRequirement.query.filter(
                dependencies.SpecialRequirement.unit_id == unit_id,
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
    def roster_telemetry():
        dependencies.validate_csrf()
        payload = request.get_json(silent=True) or {}

        def bounded_number(name: str, maximum: float) -> float:
            try:
                value = float(payload.get(name, 0) or 0)
            except (TypeError, ValueError):
                return 0.0
            return min(max(value, 0.0), maximum)

        render_ms = bounded_number("render_ms", 60_000)
        dom_ms = bounded_number("dom_ms", 60_000)
        transfer_bytes = bounded_number("transfer_bytes", 50_000_000)
        decoded_bytes = bounded_number("decoded_bytes", 50_000_000)
        dependencies.metrics.add("roster_browser_samples_total")
        for metric, value in (
            ("roster_browser_render_milliseconds_sum", render_ms),
            ("roster_browser_dom_milliseconds_sum", dom_ms),
            ("roster_browser_transfer_bytes_sum", transfer_bytes),
            ("roster_browser_decoded_bytes_sum", decoded_bytes),
        ):
            dependencies.metrics.add(metric, value)
        if render_ms >= 2_000:
            dependencies.metrics.add("roster_browser_slow_renders_total")
        return "", 204

    @login_required
    def roster_month(ym):
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        if not dependencies.roster_month_service.has_data(year, month):
            dependencies.assignment_runtime.ensure_month_requirement(year, month)
            dependencies.assignment_runtime.generate_month(year, month)

        days, staff, assignment_tuples, requirement = dependencies.load_month_roster(
            unit_id, year, month
        )
        assignment_map: dict[int, dict[date, str]] = {}
        month_assignments = dependencies.Assignment.query.filter(
            dependencies.Assignment.unit_id == unit_id,
            dependencies.Assignment.day >= date(year, month, 1),
            dependencies.Assignment.day
            < date(*dependencies.add_months(year, month, 1), 1),
        ).all()
        assignment_version_map: dict[tuple[int, date], int] = {
            (assignment.staff_id, assignment.day): assignment.version
            for assignment in month_assignments
        }
        assignment_override_map = {
            (assignment.staff_id, assignment.day): True
            for assignment in month_assignments
            if assignment.override_code is not None
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

        cached_analysis = dependencies.roster_month_cache.get(
            unit_id, year, month
        )
        capability_matrix = (
            cached_analysis["capability_matrix"] if cached_analysis else
            dependencies.operational_capability_matrix(staff, days)
        )
        excluded = dependencies.exclude_from_counters()
        night_active = {
            duty_day: dependencies.night_active_on(unit_id, duty_day)
            for duty_day in days
        }
        special_requirements = (
            dependencies.SpecialRequirement.query.filter(
                dependencies.SpecialRequirement.unit_id == unit_id,
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
        display_state = RosterMonthViewService(MonthDisplayDependencies(
            staff_is_countable_on=dependencies.staff_is_countable_on,
            shift_counter_group_for_day=dependencies.shift_counter_group_for_day,
        )).build(
            staff=staff, days=days, assignment_map=assignment_map,
            capability_matrix=capability_matrix, excluded=excluded,
            training_codes=training_codes, requirements=requirements,
            night_active=night_active, unit_id=unit_id,
            display_watch_by_staff=display_watch_by_staff, watch_order=watch_order,
            today=date.today(),
        )
        counters = display_state["counters"]
        rag = display_state["rag"]
        if cached_analysis:
            fatigue = cached_analysis["fatigue"]
            roster_validation = cached_analysis["roster_validation"]
        else:
            fatigue = dependencies.roster_fatigue_matrix(
                staff, days, assignment_map, unit_id
            )
            roster_validation = dependencies.roster_validation.validate_range(
                unit_id, days[0], days[-1]
            )
            dependencies.roster_month_cache.set(unit_id, year, month, {
                "capability_matrix": capability_matrix,
                "fatigue": fatigue,
                "roster_validation": roster_validation,
            })
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
                    dependencies.Assignment.effective_code
                    == dependencies.ShiftRequest.code,
                )
                .all()
            )
        }
        today = date.today()
        expiry_classes = display_state["expiry_classes"]
        watch_break_after_ids = display_state["watch_break_after_ids"]
        active_publication = dependencies.publication_service.active_publication(year, month)
        roster_publication = (
            active_publication
            if dependencies.publication_service.matches_live(
                active_publication, year, month
            )
            else None
        )
        roster_acknowledgement = None
        if roster_publication:
            roster_acknowledgement = dependencies.RosterAcknowledgement.query.filter_by(
                unit_id=unit_id, publication_id=roster_publication.id,
                person_id=current_user.id,
            ).first()
        return render_template(
            "roster_month.html",
            ym=ym,
            year=year,
            month=month,
            days=days,
            staff=staff,
            a_map=assignment_map,
            assignment_version_map=assignment_version_map,
            assignment_override_map=assignment_override_map,
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
            roster_acknowledgement=roster_acknowledgement,
            can_publish_roster=dependencies.publication_service.can_publish(current_user),
            capability_matrix=capability_matrix,
        )

    @login_required
    def roster_month_acknowledge(ym):
        dependencies.validate_csrf()
        year, month = dependencies.parse_year_month(ym)
        unit_id = dependencies.current_unit_id()
        publication = dependencies.publication_service.active_publication(year, month)
        if (
            not publication
            or publication.unit_id != unit_id
            or not dependencies.publication_service.matches_live(publication, year, month)
        ):
            abort(404, "There is no published roster to acknowledge.")
        acknowledgement = dependencies.RosterAcknowledgement.query.filter_by(
            unit_id=unit_id,
            publication_id=publication.id,
            person_id=current_user.id,
        ).first()
        if acknowledgement is None:
            acknowledgement = dependencies.RosterAcknowledgement(
                unit_id=unit_id,
                publication_id=publication.id,
                person_id=current_user.id,
                acknowledged_at=dependencies.utcnow(),
            )
            dependencies.db.session.add(acknowledgement)
            try:
                dependencies.db.session.commit()
            except IntegrityError:
                dependencies.db.session.rollback()
                flash("You have already acknowledged this roster version.", "info")
                return redirect(url_for("roster_month", ym=ym))
            dependencies.log_change(
                "RosterAcknowledgement", acknowledgement.id, "acknowledged_at",
                None, acknowledgement.acknowledged_at.isoformat(),
                note=f"Acknowledged roster publication version {publication.version}.",
                context_day=date(year, month, 1),
            )
            flash("Roster acknowledgement recorded.", "ok")
        else:
            flash("You have already acknowledged this roster version.", "info")
        return redirect(url_for("roster_month", ym=ym))

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

        def edit_conflict(message: str):
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(
                    ok=False,
                    error=message,
                    reload_required=True,
                ), 409
            abort(409, message)

        def updated_day_summary() -> dict[str, Any]:
            people = dependencies.Staff.query.filter_by(unit_id=unit_id).all()
            assignments = dependencies.Assignment.query.filter_by(
                unit_id=unit_id, day=duty_day
            ).all()
            codes = {
                item.staff_id: (item.effective_code or "").upper()
                for item in assignments
            }
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
            requirement = dependencies.assignment_runtime.ensure_month_requirement(year, month)
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

        dependencies.roster_month_service.lock(unit_id, year, month)
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
            return edit_conflict("This roster cell changed after the page was loaded.")
        if assignment is None:
            assignment = dependencies.Assignment(
                unit_id=unit_id,
                staff=person,
                day=duty_day,
                code="OFF",
                generated_code="OFF",
                generation_version="cell-created-v1",
            )
            dependencies.db.session.add(assignment)

        raw_code = (request.form.get("code") or "").strip().upper()
        clear_override = raw_code == "__BASELINE__"
        code = "" if clear_override else raw_code
        annotation = request.form.get("annotation")
        old_code = assignment.effective_code
        code_changed = False
        if clear_override:
            assignment.clear_editor_override()
            assignment.source = "auto"
            assignment.note = "generated baseline"
            code_changed = old_code != assignment.effective_code
        elif code:
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
            assignment.set_editor_override(
                code,
                actor_id=current_user.id,
                reason="Manual roster edit",
            )
            assignment.source = "manual"
            code_changed = old_code != assignment.effective_code

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
                assignment.set_editor_override(
                    "AL",
                    actor_id=current_user.id,
                    reason="Annual leave",
                    override_type="SYSTEM_ABSENCE",
                )
                assignment.source = "leave"
                assignment.note = "annual leave"
        assignment.version = current_version + 1
        try:
            # The audit row must live in the same transaction as the roster
            # edit.  A successful response therefore always has evidence.
            dependencies.db.session.flush()
            if code_changed:
                dependencies.stage_change(
                    "Assignment", assignment.id, "code", old_code,
                    assignment.code, note="Manual roster edit",
                    context_day=duty_day,
                )
            dependencies.db.session.commit()
        except IntegrityError:
            dependencies.db.session.rollback()
            return edit_conflict("This roster cell changed concurrently.")
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            saved_shift = dependencies.get_shift(assignment.effective_code)
            return jsonify(
                ok=True,
                staff_id=staff_id,
                day=duty_day.isoformat(),
                code=assignment.effective_code,
                annotation=assignment.annotation or "",
                annotation_note=assignment.annotation_note or "",
                version=assignment.version,
                is_training=bool(saved_shift and saved_shift.is_training),
                day_summary=updated_day_summary(),
            )
        return redirect(url_for("roster_month", ym=ym))

    @login_required
    def assignment_lock(assignment_id):
        if not dependencies.can_edit_roster(current_user):
            abort(403)
        dependencies.validate_csrf()
        unit_id = dependencies.current_unit_id()
        assignment = dependencies.Assignment.query.filter_by(
            id=assignment_id, unit_id=unit_id
        ).with_for_update().first_or_404()
        status = (request.form.get("lock_status") or "").upper()
        if status not in {"UNLOCKED", "SOFT_LOCKED", "HARD_LOCKED"}:
            abort(400, "Invalid assignment lock status.")
        old_status = assignment.lock_status or "UNLOCKED"
        assignment.lock_status = status
        if status == "UNLOCKED":
            assignment.locked_by_user_id = None
            assignment.locked_at = None
            assignment.lock_reason = ""
        else:
            assignment.locked_by_user_id = current_user.id
            assignment.locked_at = dependencies.utcnow()
            assignment.lock_reason = (
                request.form.get("lock_reason") or ""
            ).strip()[:250]
        assignment.version = int(assignment.version or 0) + 1
        dependencies.stage_change(
            "Assignment", assignment.id, "lock_status", old_status, status,
            note=assignment.lock_reason, context_day=assignment.day,
        )
        dependencies.db.session.commit()
        return redirect(url_for(
            "roster_month", ym=f"{assignment.day.year:04d}-{assignment.day.month:02d}"
        ))

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
            ("/roster/<ym>/acknowledge", "roster_month_acknowledge", roster_month_acknowledge, ["POST"]),
            ("/roster/<ym>", "roster_month", roster_month, ["GET"]),
            ("/roster/telemetry", "roster_telemetry", roster_telemetry, ["POST"]),
            (
                "/assign/<int:staff_id>/<ym>/<day>",
                "assign_cell",
                assign_cell,
                ["POST"],
            ),
            (
                "/assignment/<int:assignment_id>/lock",
                "assignment_lock",
                assignment_lock,
                ["POST"],
            ),
            ("/roster/<ym>/export", "roster_export_csv", roster_export_csv, ["GET"]),
            ("/roster/<ym>/print", "roster_print_view", roster_print_view, ["GET"]),
            ("/roster/proposals", "roster_proposals", roster_proposals, ["GET", "POST"]),
            ("/roster/proposals/<int:proposal_id>", "roster_proposal_detail", roster_proposal_detail, ["GET"]),
            ("/roster/proposals/<int:proposal_id>/items/<int:item_id>", "roster_proposal_review", roster_proposal_review, ["POST"]),
            ("/roster/proposals/<int:proposal_id>/apply", "roster_proposal_apply", roster_proposal_apply, ["POST"]),
            ("/roster/proposals/<int:proposal_id>/discard", "roster_proposal_discard", roster_proposal_discard, ["POST"]),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
