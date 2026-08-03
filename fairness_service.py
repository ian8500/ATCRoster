"""Roster fairness calculations independent of Flask and SQLAlchemy.

The service deliberately accepts plain records.  This keeps fairness policy out
of routes and lets the future optimiser consume exactly the same calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FairnessAssignment:
    staff_id: int
    day: date
    shift_code: str
    minutes: int
    start_time: time | None
    source: str = ""
    is_night: bool = False


@dataclass(frozen=True)
class FairnessStaff:
    staff_id: int
    name: str
    expected_minutes: int
    eligible_nights: bool = True
    eligible_early: bool = True


@dataclass(frozen=True)
class StaffFairnessMetrics:
    staff_id: int
    name: str
    contracted_ratio: float
    actual_minutes: int
    target_minutes: int
    difference_minutes: int
    night_count: int
    target_night_count: float
    weekend_count: int
    target_weekend_count: float
    bank_holiday_count: int
    target_bank_holiday_count: float
    early_count: int
    target_early_count: float
    overtime_minutes: int
    pattern_deviations: int
    preference_breaches: int
    manual_roster_changes: int


def calculate_fairness(
    staff: Sequence[FairnessStaff],
    assignments: Iterable[FairnessAssignment],
    *,
    expected_code_for: Callable[[int, date], str | None] | None = None,
    preference_breach_for: Callable[[int, date, str], bool] | None = None,
    bank_holidays: Iterable[date] = (),
    manual_change_counts: Mapping[int, int] | None = None,
    early_before: time = time(8, 0),
) -> list[StaffFairnessMetrics]:
    """Return transparent actual-versus-target metrics for each staff member.

    Burdens are shared in proportion to expected minutes. Night and early
    targets only include staff who are eligible for that duty type.
    """
    people = {person.staff_id: person for person in staff}
    rows = [row for row in assignments if row.staff_id in people and row.minutes > 0]
    holidays = set(bank_holidays)
    manual_counts = manual_change_counts or {}

    actual = {staff_id: 0 for staff_id in people}
    nights = {staff_id: 0 for staff_id in people}
    weekends = {staff_id: 0 for staff_id in people}
    holiday_counts = {staff_id: 0 for staff_id in people}
    earlies = {staff_id: 0 for staff_id in people}
    deviations = {staff_id: 0 for staff_id in people}
    breaches = {staff_id: 0 for staff_id in people}

    for row in rows:
        actual[row.staff_id] += row.minutes
        code = row.shift_code.upper()
        if row.is_night or code == "N":
            nights[row.staff_id] += 1
        if row.day.weekday() >= 5:
            weekends[row.staff_id] += 1
        if row.day in holidays:
            holiday_counts[row.staff_id] += 1
        if row.start_time is not None and row.start_time < early_before:
            earlies[row.staff_id] += 1
        if expected_code_for is not None:
            expected = (expected_code_for(row.staff_id, row.day) or "").upper()
            if expected and expected != code:
                deviations[row.staff_id] += 1
        if preference_breach_for and preference_breach_for(
            row.staff_id, row.day, code
        ):
            breaches[row.staff_id] += 1

    total_expected = sum(max(0, person.expected_minutes) for person in staff)
    total_nights = sum(nights.values())
    total_weekends = sum(weekends.values())
    total_holidays = sum(holiday_counts.values())
    total_earlies = sum(earlies.values())

    def proportional_target(total: int, eligible_ids: list[int], staff_id: int) -> float:
        eligible_expected = sum(
            max(0, people[item].expected_minutes) for item in eligible_ids
        )
        if staff_id not in eligible_ids or eligible_expected <= 0:
            return 0.0
        return total * max(0, people[staff_id].expected_minutes) / eligible_expected

    all_eligible = list(people)
    night_eligible = [sid for sid, p in people.items() if p.eligible_nights]
    early_eligible = [sid for sid, p in people.items() if p.eligible_early]
    result = []
    for person in staff:
        staff_id = person.staff_id
        target_minutes = max(0, person.expected_minutes)
        result.append(StaffFairnessMetrics(
            staff_id=staff_id,
            name=person.name,
            contracted_ratio=(target_minutes / total_expected if total_expected else 0.0),
            actual_minutes=actual[staff_id],
            target_minutes=target_minutes,
            difference_minutes=actual[staff_id] - target_minutes,
            night_count=nights[staff_id],
            target_night_count=proportional_target(total_nights, night_eligible, staff_id),
            weekend_count=weekends[staff_id],
            target_weekend_count=proportional_target(total_weekends, all_eligible, staff_id),
            bank_holiday_count=holiday_counts[staff_id],
            target_bank_holiday_count=proportional_target(total_holidays, all_eligible, staff_id),
            early_count=earlies[staff_id],
            target_early_count=proportional_target(total_earlies, early_eligible, staff_id),
            overtime_minutes=max(0, actual[staff_id] - target_minutes),
            pattern_deviations=deviations[staff_id],
            preference_breaches=breaches[staff_id],
            manual_roster_changes=max(0, int(manual_counts.get(staff_id, 0))),
        ))
    return result
