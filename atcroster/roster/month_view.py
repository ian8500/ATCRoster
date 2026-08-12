"""Pure-ish display calculations for the monthly roster surface."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable


@dataclass(frozen=True)
class MonthDisplayDependencies:
    staff_is_countable_on: Callable[[Any, date], bool]
    shift_counter_group_for_day: Callable[[str, date, int], str | None]


class RosterMonthViewService:
    """Build template-only monthly display state from already-loaded data."""

    def __init__(self, dependencies: MonthDisplayDependencies) -> None:
        self.dependencies = dependencies

    def build(
        self, *, staff: list[Any], days: list[date], assignment_map: dict[int, dict[date, str]],
        capability_matrix: dict, excluded: set[str], training_codes: set[str],
        requirements: dict[date, dict[str, int]], night_active: dict[date, bool],
        unit_id: int, display_watch_by_staff: dict[int, int | None], watch_order: dict[int, int],
        today: date,
    ) -> dict[str, Any]:
        def rank(person: Any) -> int:
            return 0 if getattr(person, "is_wm", False) else 1 if getattr(person, "is_dwm", False) else 2

        staff.sort(key=lambda person: (
            watch_order.get(display_watch_by_staff.get(person.id), 9999), rank(person), person.name,
        ))
        counters = {day: Counter() for day in days}
        for person in staff:
            if not getattr(person, "is_operational", True):
                continue
            assignments = assignment_map.get(person.id, {})
            for duty_day in days:
                capability = capability_matrix.get((person.id, duty_day))
                countable = capability.counts_as_operational if capability is not None else self.dependencies.staff_is_countable_on(person, duty_day)
                code = (assignments.get(duty_day) or "").upper()
                if not countable or not code or code in excluded | training_codes | {"AL", "NOPS"}:
                    continue
                group = self.dependencies.shift_counter_group_for_day(code, duty_day, unit_id)
                if group:
                    counters[duty_day][group] += 1
        rag = {
            duty_day: {
                code: "green" if counters[duty_day][code] >= (0 if code == "N" and not night_active[duty_day] else requirements[duty_day][code])
                else "amber" if counters[duty_day][code] >= max(0, (0 if code == "N" and not night_active[duty_day] else requirements[duty_day][code]) - 1) else "red"
                for code in ("M", "D", "A", "N")
            }
            for duty_day in days
        }
        def expiry_class(expiry: date | None, under_training: bool = False) -> str:
            if under_training:
                return "exp-amber"
            if not expiry:
                return ""
            return "exp-red" if (expiry - today).days < 0 else "exp-amber" if (expiry - today).days <= 90 else "exp-green"
        expiry_classes = {
            person.id: {
                "medical": expiry_class(person.medical_expiry),
                "tower": expiry_class(person.tower_ue_expiry, person.tower_ut),
                "radar": expiry_class(person.radar_ue_expiry, person.radar_ut),
                "met": expiry_class(person.met_ue_expiry, person.met_ut),
            } for person in staff
        }
        watch_break_after_ids: list[int] = []
        for previous, current in zip(staff, staff[1:]):
            if display_watch_by_staff.get(previous.id) != display_watch_by_staff.get(current.id):
                watch_break_after_ids.append(previous.id)
        return {"counters": counters, "rag": rag, "expiry_classes": expiry_classes, "watch_break_after_ids": watch_break_after_ids}
