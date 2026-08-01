"""Reusable report calculations, independent of Flask route handling."""

from __future__ import annotations

from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any, Callable, Iterable


SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


def csv_safe_cell(value: object) -> object:
    """Neutralise text that spreadsheet applications may execute as a formula."""
    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" \t\r\n")
    if candidate.startswith(SPREADSHEET_FORMULA_PREFIXES):
        return "'" + value
    return value


def compute_annotation_metrics(
    start_day: date,
    end_day: date,
    *,
    Assignment: Any,
    Staff: Any,
    Watch: Any,
    annotation_items: Iterable[dict[str, object]],
    parse_annotation: Callable[[str], dict[str, str] | None],
):
    assignments = Assignment.query.filter(
        Assignment.day >= start_day,
        Assignment.day <= end_day,
    ).all()
    items = list(annotation_items)
    label_map = {item["code"]: item["label"] for item in items}
    excluded = {
        item["code"]
        for item in items
        if "report_exclude" in set(item.get("tags") or ())
    }
    columns = [
        {
            "code": item["code"],
            "label": item["label"] or item["code"],
            "active": bool(item["is_active"]),
        }
        for item in items
        if item["is_active"] and item["code"] not in excluded
    ]
    order = [column["code"] for column in columns]
    known = set(order)
    staff_by_id = {person.id: person for person in Staff.query.all()}
    metrics: dict[int, dict[str, object]] = {}

    for assignment in assignments:
        person = staff_by_id.get(assignment.staff_id)
        if not person:
            continue
        metrics.setdefault(
            person.id,
            {
                "staff": person,
                "annotations": {code: 0 for code in order},
            },
        )
        parsed = parse_annotation(assignment.annotation)
        if not parsed or parsed["type"] in excluded:
            continue
        code = parsed["type"]
        if code not in known:
            known.add(code)
            order.append(code)
            columns.append(
                {
                    "code": code,
                    "label": label_map.get(code, code),
                    "active": False,
                }
            )
        annotations = metrics[person.id]["annotations"]
        annotations.setdefault(code, 0)
        annotations[code] += 1

    ordered_people = (
        Staff.query.outerjoin(Watch, Staff.watch_id == Watch.id)
        .order_by(Watch.order_index, Staff.name)
        .all()
    )
    rows = []
    for person in ordered_people:
        row = metrics.get(
            person.id,
            {
                "staff": person,
                "annotations": {},
            },
        )
        row["annotations"] = {code: row["annotations"].get(code, 0) for code in order}
        rows.append(row)
    totals = {
        "annotations": {
            code: sum(row["annotations"].get(code, 0) for row in rows) for code in order
        }
    }
    return rows, totals, columns


def leave_summary_for_month(
    year: int,
    month: int,
    watch_id: int | None,
    *,
    unit_id: int,
    Assignment: Any,
    Staff: Any,
    Watch: Any,
    month_range: Callable,
    active_leave_types: Iterable[dict[str, object]],
):
    start, days = month_range(year, month)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)
    assignments: dict[int, dict[date, str]] = defaultdict(dict)
    for row in Assignment.query.filter(
        Assignment.unit_id == unit_id,
        Assignment.day >= start,
        Assignment.day < month_end,
    ):
        assignments[row.staff_id][row.day] = row.code

    query = Staff.query.filter(Staff.unit_id == unit_id).outerjoin(
        Watch, Staff.watch_id == Watch.id
    )
    if watch_id is not None:
        query = query.filter(Staff.watch_id == watch_id)
    people = query.order_by(Watch.order_index, Staff.name).all()
    codes = [item["code"] for item in active_leave_types]
    rows = []
    totals = Counter()
    for person in people:
        counts = {code: 0 for code in codes}
        for day in days:
            code = assignments[person.id].get(day)
            if code in counts:
                counts[code] += 1
        total = sum(counts.values())
        totals.update(counts)
        rows.append({"staff": person, "counts": counts, "total": total})
    return rows, codes, totals, sum(totals.values()), days


def financial_year_start(day: date) -> date:
    return date(day.year if day.month >= 4 else day.year - 1, 4, 1)


def current_leave_year_window(person: Any, today: date | None = None):
    today = today or date.today()
    start_month = person.leave_year_start_month or 4
    start_year = today.year if today.month >= start_month else today.year - 1
    start = date(start_year, start_month, 1)
    end_month = start_month - 1 if start_month > 1 else 12
    _, end_days = monthrange(start_year + 1, end_month)
    return start, date(start_year + 1, end_month, end_days)


def group_consecutive_days(days: Iterable[date]) -> int:
    groups = 0
    previous = None
    for day in sorted(set(days)):
        if previous is None or (day - previous).days > 1:
            groups += 1
        previous = day
    return groups
