"""Roster fairness query orchestration and reporting totals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Any, Callable


@dataclass(frozen=True)
class FairnessDependencies:
    Assignment: Any
    BankHoliday: Any
    ChangeLog: Any
    ShiftType: Any
    Staff: Any
    FairnessAssignment: Any
    FairnessStaff: Any
    current_unit_id: Callable[[], int]
    work_pattern_service: Any
    code_from_pattern: Callable[[Any, Any], str | None]
    shift_duration_minutes: Callable[[Any], int]
    calculate_fairness: Callable[..., list[Any]]


class FairnessReportService:
    """Adapt persisted roster state for the canonical fairness engine."""

    def __init__(self, dependencies: FairnessDependencies):
        self.dependencies = dependencies

    def compute(self, start_day: Any, end_day: Any):
        deps = self.dependencies
        unit_id = deps.current_unit_id()
        people = (
            deps.Staff.query.filter_by(
                unit_id=unit_id,
                is_operational=True,
                membership_status="active",
            )
            .order_by(deps.Staff.name)
            .all()
        )
        shifts = {
            shift.code.upper(): shift
            for shift in deps.ShiftType.query.filter_by(unit_id=unit_id).all()
        }
        assignments = deps.Assignment.query.filter(
            deps.Assignment.unit_id == unit_id,
            deps.Assignment.day >= start_day,
            deps.Assignment.day <= end_day,
        ).all()
        days = [
            start_day + timedelta(days=offset)
            for offset in range((end_day - start_day).days + 1)
        ]
        context = deps.work_pattern_service.build_eligibility_context(
            unit_id, [person.id for person in people], start_day, end_day
        )
        expected, eligible_nights, eligible_early = self._eligibility(
            people, days, shifts, context
        )
        fairness_rows, assignment_ids = self._assignment_rows(assignments, shifts)
        manual_changes = {
            staff_id: deps.ChangeLog.query.filter(
                deps.ChangeLog.unit_id == unit_id,
                deps.ChangeLog.entity_type == "Assignment",
                deps.ChangeLog.entity_id.in_(ids),
            ).count()
            for staff_id, ids in assignment_ids.items()
        }
        people_by_id = {person.id: person for person in people}
        holidays = {
            holiday.day
            for holiday in deps.BankHoliday.query.filter(
                deps.BankHoliday.unit_id == unit_id,
                deps.BankHoliday.is_active.is_(True),
                deps.BankHoliday.day >= start_day,
                deps.BankHoliday.day <= end_day,
            ).all()
        }

        def expected_code(staff_id, day):
            resolution = deps.work_pattern_service.get_pattern_day_from_context(
                staff_id, day, context
            )
            if resolution and resolution.pattern_day.day_type != "FIXED_SHIFT":
                return None
            return deps.code_from_pattern(people_by_id[staff_id], day)

        def preference_breach(staff_id, day, code):
            shift = shifts.get(code.upper())
            return bool(
                shift
                and deps.work_pattern_service.is_staff_eligible_for_shift(
                    staff_id, day, shift.id, context=context
                ).soft_penalty
            )

        metrics = deps.calculate_fairness(
            [
                deps.FairnessStaff(
                    person.id,
                    person.name,
                    expected[person.id],
                    eligible_nights[person.id],
                    eligible_early[person.id],
                )
                for person in people
            ],
            fairness_rows,
            expected_code_for=expected_code,
            preference_breach_for=preference_breach,
            bank_holidays=holidays,
            manual_change_counts=manual_changes,
        )
        totals = {
            "actual_minutes": sum(row.actual_minutes for row in metrics),
            "target_minutes": sum(row.target_minutes for row in metrics),
            "nights": sum(row.night_count for row in metrics),
            "weekends": sum(row.weekend_count for row in metrics),
            "earlies": sum(row.early_count for row in metrics),
            "overtime_minutes": sum(row.overtime_minutes for row in metrics),
        }
        return metrics, totals

    def _eligibility(self, people, days, shifts, context):
        deps = self.dependencies
        expected_minutes = {}
        eligible_nights = {}
        eligible_early = {}
        night_shift = shifts.get("N")
        for person in people:
            expected = 0.0
            night_possible = early_possible = False
            for day in days:
                resolution = deps.work_pattern_service.get_pattern_day_from_context(
                    person.id, day, context
                )
                if resolution:
                    contracted = (
                        resolution.assignment.contracted_minutes_override
                        if resolution.assignment.contracted_minutes_override is not None
                        else resolution.pattern.contracted_minutes_per_cycle
                    )
                    expected += contracted / max(
                        1, resolution.pattern.cycle_length_days
                    )
                shift = shifts.get((deps.code_from_pattern(person, day) or "").upper())
                if shift and shift.is_working:
                    if not resolution:
                        expected += deps.shift_duration_minutes(shift)
                    night_possible |= shift.code.upper() == "N"
                    early_possible |= bool(
                        shift.start_time and shift.start_time < time(8)
                    )
            if night_shift and all(
                not deps.work_pattern_service.is_staff_eligible_for_shift(
                    person.id, day, night_shift.id, context=context
                ).eligible
                for day in days
            ):
                night_possible = False
            expected_minutes[person.id] = int(round(expected))
            eligible_nights[person.id] = night_possible
            eligible_early[person.id] = early_possible
        return expected_minutes, eligible_nights, eligible_early

    def _assignment_rows(self, assignments, shifts):
        deps = self.dependencies
        rows = []
        assignment_ids = defaultdict(list)
        for assignment in assignments:
            shift = shifts.get((assignment.code or "").upper())
            if not shift or not shift.is_working:
                continue
            rows.append(
                deps.FairnessAssignment(
                    assignment.staff_id,
                    assignment.day,
                    shift.code,
                    deps.shift_duration_minutes(shift),
                    shift.start_time,
                    assignment.source or "",
                    bool(
                        shift.start_time
                        and shift.end_time
                        and shift.start_time >= time(18)
                        and shift.end_time <= time(10)
                    ),
                )
            )
            assignment_ids[assignment.staff_id].append(assignment.id)
        return rows, assignment_ids
