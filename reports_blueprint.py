"""Report routes extracted from the legacy application module.

The blueprint deliberately registers the historical global endpoint names so
existing templates, redirects, bookmarks and authorization tests remain valid.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

from flask import (
    Blueprint,
    Response,
    abort,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required
from reporting import csv_safe_cell


@dataclass(frozen=True)
class ReportsDependencies:
    Assignment: Any
    Staff: Any
    Watch: Any
    is_admin_user: Callable[[Any], bool]
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., bool]
    compute_metrics_range: Callable[[date, date, int | None], tuple]
    financial_year_start: Callable[[date], date]
    parse_year_month: Callable[[str], tuple[int, int]]
    ensure_month_requirement: Callable[[int, int], Any]
    generate_month: Callable[[int, int], None]
    leave_summary_for_month: Callable[..., tuple]
    current_leave_year_window: Callable[[Any, date | None], tuple[date, date]]
    toil_accrued_used: Callable[[int, date, date], tuple[int, int]]
    group_consecutive_days: Callable[[set[date]], int]
    get_absence_types: Callable[..., list]


def create_reports_blueprint(dependencies: ReportsDependencies) -> Blueprint:
    blueprint = Blueprint("reports", __name__)

    def acknowledgement_key() -> str:
        return f"{current_user.id}:{dependencies.current_unit_id()}"

    def sensitive_data_acknowledged() -> bool:
        return session.get("reports_sensitive_data_ack") == acknowledgement_key()

    def require_acknowledgement():
        if sensitive_data_acknowledged():
            return None
        return redirect(url_for("reports_index"))

    def watch_selection():
        unit_id = dependencies.current_unit_id()
        watches = (
            dependencies.Watch.query.filter(dependencies.Watch.unit_id == unit_id)
            .order_by(dependencies.Watch.order_index, dependencies.Watch.name)
            .all()
        )
        raw_watch_id = (request.args.get("watch_id") or "").strip()
        if not raw_watch_id:
            return watches, None
        try:
            watch_id = int(raw_watch_id)
        except ValueError:
            abort(400, "Invalid watch.")
        selected = next((watch for watch in watches if watch.id == watch_id), None)
        if selected is None:
            abort(404)
        return watches, selected

    @login_required
    def metrics():
        if not (
            dependencies.is_admin_user(current_user)
            or getattr(current_user, "role", "") in ("editor", "admin")
        ):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        today = date.today()
        default_start = dependencies.financial_year_start(today)
        start_day = date.fromisoformat(
            request.args.get("start", default_start.isoformat())
        )
        end_day = date.fromisoformat(request.args.get("end", today.isoformat()))
        watches, selected_watch = watch_selection()
        staff_metrics, totals, annotation_columns = dependencies.compute_metrics_range(
            start_day, end_day, selected_watch.id if selected_watch else None
        )
        return render_template(
            "metrics.html",
            start=start_day,
            end=end_day,
            staff_metrics=staff_metrics,
            totals=totals,
            annotation_columns=annotation_columns,
            watches=watches,
            selected_watch=selected_watch,
        )

    @login_required
    def metrics_export():
        if not dependencies.consume_rate_limit(
            "metrics-export",
            current_user.id,
            limit=20,
            window=timedelta(hours=1),
        ):
            abort(429)
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        today = date.today()
        default_start = dependencies.financial_year_start(today)
        start_day = date.fromisoformat(
            request.args.get("start", default_start.isoformat())
        )
        end_day = date.fromisoformat(request.args.get("end", today.isoformat()))
        watches, selected_watch = watch_selection()
        staff_metrics, totals, annotation_columns = dependencies.compute_metrics_range(
            start_day, end_day, selected_watch.id if selected_watch else None
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["ATCO", "Staff #", "Watch"]
            + [f"{column['label']} ({column['code']})" for column in annotation_columns]
        )
        for row in staff_metrics:
            person = row["staff"]
            watch = person.watch.name.replace("Watch ", "") if person.watch else "-"
            writer.writerow(
                csv_safe_cell(value)
                for value in (
                    [person.name, person.staff_no, watch]
                    + [
                        row["annotations"].get(column["code"], 0)
                        for column in annotation_columns
                    ]
                )
            )
        writer.writerow([])
        writer.writerow(
            [selected_watch.name if selected_watch else "Entire unit", "", ""]
            + [
                totals["annotations"].get(column["code"], 0)
                for column in annotation_columns
            ]
        )
        watch_suffix = f"_watch-{selected_watch.id}" if selected_watch else ""
        filename = f"annotation-totals_{start_day.isoformat()}_to_{end_day.isoformat()}{watch_suffix}.csv"
        return Response(
            output.getvalue().encode("utf-8"),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @login_required
    def report_leave(ym):
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        year, month = dependencies.parse_year_month(ym)
        dependencies.ensure_month_requirement(year, month)
        dependencies.generate_month(year, month)
        watches, selected_watch = watch_selection()
        rows, codes, totals, grand_total, _days = dependencies.leave_summary_for_month(
            year, month, selected_watch.id if selected_watch else None
        )
        month_title = datetime(year, month, 1).strftime("%B %Y")
        return render_template(
            "report_leave.html",
            ym=ym,
            year=year,
            month=month,
            month_title=month_title,
            rows=rows,
            codes=codes,
            totals=totals,
            grand_total=grand_total,
            watches=watches,
            selected_watch=selected_watch,
        )

    @login_required
    def report_leave_csv():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        ym = request.args.get("ym")
        if not ym:
            abort(400)
        year, month = dependencies.parse_year_month(ym)
        dependencies.ensure_month_requirement(year, month)
        dependencies.generate_month(year, month)
        watches, selected_watch = watch_selection()
        rows, codes, totals, grand_total, _days = dependencies.leave_summary_for_month(
            year, month, selected_watch.id if selected_watch else None
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Name", "Staff #", "Watch", *codes, "Total"])
        for row in rows:
            person = row["staff"]
            watch = person.watch.name.replace("Watch ", "") if person.watch else "-"
            writer.writerow(
                csv_safe_cell(value)
                for value in [
                    person.name,
                    person.staff_no,
                    watch,
                    *[row["counts"].get(code, 0) for code in codes],
                    row["total"],
                ]
            )
        writer.writerow([])
        writer.writerow(
            ["Totals", "", "", *[totals.get(code, 0) for code in codes], grand_total]
        )
        watch_suffix = f"_watch-{selected_watch.id}" if selected_watch else ""
        filename = f"leave_{year:04d}-{month:02d}{watch_suffix}.csv"
        return Response(
            output.getvalue().encode("utf-8"),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    @login_required
    def report_leave_year():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        today = date.today()
        raw_end_date = (request.args.get("end_date") or "").strip()
        if raw_end_date:
            try:
                report_end = date.fromisoformat(raw_end_date)
            except ValueError:
                abort(400, "Invalid report end date.")
        else:
            report_end = today
        unit_id = dependencies.current_unit_id()
        watches, selected_watch = watch_selection()
        people_query = dependencies.Staff.query.filter(
            dependencies.Staff.unit_id == unit_id,
            dependencies.Staff.role != "position_monitor",
        ).outerjoin(
            dependencies.Watch,
            dependencies.Staff.watch_id == dependencies.Watch.id,
        )
        if selected_watch:
            people_query = people_query.filter(
                dependencies.Staff.watch_id == selected_watch.id
            )
        people = people_query.order_by(
            dependencies.Watch.order_index, dependencies.Staff.name
        ).all()
        rows = []
        for person in people:
            start, leave_year_end = dependencies.current_leave_year_window(
                person, report_end
            )
            assignments = dependencies.Assignment.query.filter(
                dependencies.Assignment.staff_id == person.id,
                dependencies.Assignment.day >= start,
                dependencies.Assignment.day <= report_end,
            ).all()
            al_taken = sum(1 for assignment in assignments if assignment.code == "AL")
            entitlement = person.leave_entitlement_days or 0
            public_holidays = person.leave_public_holidays or 0
            carryover = person.leave_carryover_days or 0
            accrued, used = dependencies.toil_accrued_used(
                person.id, start, report_end
            )
            rows.append(
                {
                    "staff": person,
                    "watch": person.watch.name.replace("Watch ", "")
                    if person.watch
                    else "-",
                    "leave_year_start": start,
                    "leave_year_end": leave_year_end,
                    "entitlement": entitlement,
                    "public_holidays": public_holidays,
                    "carryover": carryover,
                    "al_taken": al_taken,
                    "remaining": entitlement + public_holidays + carryover - al_taken,
                    "toil_accrued_days": accrued / 2.0,
                    "toil_used_days": used / 2.0,
                    "toil_balance_days": (person.toil_half_days or 0) / 2.0,
                }
            )
        return render_template(
            "report_leave_year.html",
            rows=rows,
            today=today,
            report_end=report_end,
            watches=watches,
            selected_watch=selected_watch,
        )

    @login_required
    def report_sickness():
        if not dependencies.is_admin_user(current_user):
            abort(403)
        if acknowledgement := require_acknowledgement():
            return acknowledgement
        today = date.today()
        start = today - timedelta(days=365)
        unit_id = dependencies.current_unit_id()
        watches, selected_watch = watch_selection()
        people_query = dependencies.Staff.query.filter(
            dependencies.Staff.unit_id == unit_id,
            dependencies.Staff.role != "position_monitor",
        ).outerjoin(
            dependencies.Watch,
            dependencies.Staff.watch_id == dependencies.Watch.id,
        )
        if selected_watch:
            people_query = people_query.filter(
                dependencies.Staff.watch_id == selected_watch.id
            )
        people = people_query.order_by(
            dependencies.Watch.order_index, dependencies.Staff.name
        ).all()
        sickness_types = dependencies.get_absence_types("sickness", active_only=True)
        codes = [item["code"] for item in sickness_types]
        rows = []
        totals = Counter()
        for person in people:
            assignments = [
                assignment
                for assignment in dependencies.Assignment.query.filter(
                    dependencies.Assignment.unit_id == unit_id,
                    dependencies.Assignment.staff_id == person.id,
                    dependencies.Assignment.day >= start,
                    dependencies.Assignment.day <= today,
                ).all()
                if assignment.code in codes
            ]
            sick_days = sorted(assignment.day for assignment in assignments)
            counts = Counter(assignment.code for assignment in assignments)
            totals.update(counts)
            rows.append(
                {
                    "staff": person,
                    "watch": person.watch.name.replace("Watch ", "")
                    if person.watch
                    else "-",
                    "total": len(sick_days),
                    "groups": dependencies.group_consecutive_days(set(sick_days)),
                    "counts": counts,
                }
            )
        return render_template(
            "report_sickness.html",
            start=start,
            end=today,
            rows=rows,
            sickness_types=sickness_types,
            totals=totals,
            watches=watches,
            selected_watch=selected_watch,
            has_sickness=any(row["total"] > 0 for row in rows),
        )

    @login_required
    def reports_index():
        can_view = dependencies.is_admin_user(current_user) or getattr(
            current_user, "role", ""
        ) in ("editor", "admin")
        if not can_view:
            abort(403)
        if request.method == "POST":
            dependencies.validate_csrf()
            session["reports_sensitive_data_ack"] = acknowledgement_key()
            session["reports_sensitive_data_hub_entry"] = acknowledgement_key()
            return redirect(url_for("reports_index"))
        if not sensitive_data_acknowledged():
            return render_template("reports_index.html", requires_acknowledgement=True)
        if (
            session.pop("reports_sensitive_data_hub_entry", None)
            != acknowledgement_key()
        ):
            session.pop("reports_sensitive_data_ack", None)
            return render_template("reports_index.html", requires_acknowledgement=True)
        if dependencies.is_admin_user(current_user):
            today = date.today()
            return render_template(
                "reports_index.html",
                ym=f"{today.year}-{today.month:02d}",
                year=today.year,
                month=today.month,
                month_title=datetime(today.year, today.month, 1).strftime("%B %Y"),
                months=[],
                links={
                    "leave_year": url_for("report_leave_year"),
                    "sickness": url_for("report_sickness"),
                    "roster": url_for(
                        "roster_month", ym=f"{today.year}-{today.month:02d}"
                    ),
                    "metrics": url_for("metrics"),
                },
                page_title="Annotation Totals",
                requires_acknowledgement=False,
            )
        if getattr(current_user, "role", "") in ("editor", "admin"):
            return redirect(url_for("metrics"))
        abort(403)

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/metrics", "metrics", metrics, ["GET"]),
            ("/metrics/export", "metrics_export", metrics_export, ["GET"]),
            ("/reports/leave/<ym>", "report_leave", report_leave, ["GET"]),
            ("/reports/leave.csv", "report_leave_csv", report_leave_csv, ["GET"]),
            ("/reports/leave-year", "report_leave_year", report_leave_year, ["GET"]),
            ("/reports/sickness", "report_sickness", report_sickness, ["GET"]),
            ("/reports", "reports_index", reports_index, ["GET", "POST"]),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
