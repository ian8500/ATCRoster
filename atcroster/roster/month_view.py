"""Pure-ish display calculations for the monthly roster surface."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from calendar import monthrange
from datetime import date, timedelta
from typing import Any, Callable


@dataclass(frozen=True)
class MonthDisplayDependencies:
    staff_is_countable_on: Callable[[Any, date], bool]
    shift_counter_group_for_day: Callable[[str, date, int], str | None]


@dataclass(frozen=True)
class MonthRosterLoadDependencies:
    db: Any
    Assignment: Any
    Requirement: Any
    Staff: Any
    Watch: Any
    ensure_month_requirement: Callable[[int, int], Any]
    log_exception: Callable[..., None]


def load_month_roster(
    unit_id: int,
    year: int,
    month: int,
    dependencies: MonthRosterLoadDependencies,
) -> tuple[list[date], list[Any], dict[int, dict[date, tuple]], Any]:
    """Load the narrow monthly roster projection used by the roster view."""
    deps = dependencies
    try:
        start = date(year, month, 1)
        days = [
            start + timedelta(days=offset)
            for offset in range(monthrange(year, month)[1])
        ]
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        end = date(next_year, next_month, 1)
        try:
            staff = (
                deps.Staff.query.outerjoin(
                    deps.Watch, deps.Staff.watch_id == deps.Watch.id
                )
                .filter(deps.Staff.role != "position_monitor")
                .order_by(deps.Watch.order_index, deps.Staff.name)
                .all()
            )
        except Exception:
            staff = (
                deps.Staff.query.filter(deps.Staff.role != "position_monitor")
                .order_by(deps.Staff.id)
                .all()
            )
        rows = (
            deps.db.session.query(
                deps.Assignment.staff_id,
                deps.Assignment.day,
                deps.Assignment.effective_code.label("effective_code"),
                deps.Assignment.source,
                deps.Assignment.annotation,
                deps.Assignment.annotation_note,
            )
            .filter(deps.Assignment.day >= start, deps.Assignment.day < end)
            .all()
        )
        assignment_map: dict[int, dict[date, tuple]] = {}
        for staff_id, day, code, source, annotation, note in rows:
            assignment_map.setdefault(staff_id, {})[day] = (
                code,
                source,
                annotation,
                note or "",
            )
        requirement = deps.Requirement.query.filter_by(year=year, month=month).first()
        if not requirement:
            requirement = deps.ensure_month_requirement(year, month)
        return days, staff, assignment_map, requirement
    except Exception as exc:
        try:
            deps.log_exception(
                "Failed load_month_roster(%s,%s,%s): %s",
                unit_id,
                year,
                month,
                exc,
            )
        except Exception:
            pass
        return [], [], {}, deps.ensure_month_requirement(year, month)


class RosterMonthViewService:
    """Build template-only monthly display state from already-loaded data."""

    def __init__(self, dependencies: MonthDisplayDependencies) -> None:
        self.dependencies = dependencies

    def build(
        self,
        *,
        staff: list[Any],
        days: list[date],
        assignment_map: dict[int, dict[date, str]],
        capability_matrix: dict,
        excluded: set[str],
        training_codes: set[str],
        requirements: dict[date, dict[str, int]],
        night_active: dict[date, bool],
        unit_id: int,
        display_watch_by_staff: dict[int, int | None],
        watch_order: dict[int, int],
        today: date,
    ) -> dict[str, Any]:
        def rank(person: Any) -> int:
            return (
                0
                if getattr(person, "is_wm", False)
                else 1
                if getattr(person, "is_dwm", False)
                else 2
            )

        def watch_rank(person: Any) -> int:
            watch_id = display_watch_by_staff.get(person.id)
            return watch_order.get(watch_id, 9999) if watch_id is not None else 9999

        staff.sort(key=lambda person: (watch_rank(person), rank(person), person.name))
        counters: dict[date, Counter[str]] = {day: Counter() for day in days}
        for person in staff:
            if not getattr(person, "is_operational", True):
                continue
            assignments = assignment_map.get(person.id, {})
            for duty_day in days:
                capability = capability_matrix.get((person.id, duty_day))
                countable = (
                    capability.counts_as_operational
                    if capability is not None
                    else self.dependencies.staff_is_countable_on(person, duty_day)
                )
                code = (assignments.get(duty_day) or "").upper()
                if (
                    not countable
                    or not code
                    or code in excluded | training_codes | {"AL", "NOPS"}
                ):
                    continue
                group = self.dependencies.shift_counter_group_for_day(
                    code, duty_day, unit_id
                )
                if group:
                    counters[duty_day][group] += 1
        rag = {
            duty_day: {
                code: "green"
                if counters[duty_day][code]
                >= (
                    0
                    if code == "N" and not night_active[duty_day]
                    else requirements[duty_day][code]
                )
                else "amber"
                if counters[duty_day][code]
                >= max(
                    0,
                    (
                        0
                        if code == "N" and not night_active[duty_day]
                        else requirements[duty_day][code]
                    )
                    - 1,
                )
                else "red"
                for code in ("M", "D", "A", "N")
            }
            for duty_day in days
        }

        def expiry_class(expiry: date | None, under_training: bool = False) -> str:
            if under_training:
                return "exp-amber"
            if not expiry:
                return ""
            return (
                "exp-red"
                if (expiry - today).days < 0
                else "exp-amber"
                if (expiry - today).days <= 90
                else "exp-green"
            )

        expiry_classes = {
            person.id: {
                "medical": expiry_class(person.medical_expiry),
                "tower": expiry_class(person.tower_ue_expiry, person.tower_ut),
                "radar": expiry_class(person.radar_ue_expiry, person.radar_ut),
                "met": expiry_class(person.met_ue_expiry, person.met_ut),
            }
            for person in staff
        }
        watch_break_after_ids: list[int] = []
        for previous, current in zip(staff, staff[1:]):
            if display_watch_by_staff.get(previous.id) != display_watch_by_staff.get(
                current.id
            ):
                watch_break_after_ids.append(previous.id)
        return {
            "counters": counters,
            "rag": rag,
            "expiry_classes": expiry_classes,
            "watch_break_after_ids": watch_break_after_ids,
        }
