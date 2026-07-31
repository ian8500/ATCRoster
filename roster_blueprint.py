"""Roster routes extracted incrementally from the legacy application module."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    url_for,
)
from flask_login import current_user, login_required


@dataclass(frozen=True)
class RosterDependencies:
    db: Any
    RosterPublication: Any
    Staff: Any
    Notification: Any
    Assignment: Any
    Watch: Any
    Requirement: Any
    SpecialRequirement: Any
    roster_month: Callable
    assign_cell: Callable
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


def create_roster_blueprint(dependencies: RosterDependencies) -> Blueprint:
    blueprint = Blueprint("roster", __name__)

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
                [
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
            ("/roster/<ym>", "roster_month", dependencies.roster_month, ["GET"]),
            (
                "/assign/<int:staff_id>/<ym>/<day>",
                "assign_cell",
                dependencies.assign_cell,
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
