"""Safe, deterministic gap-filling proposals for roster assignments.

The allocator accepts plain records and callbacks.  It never writes to the
database, which keeps proposal generation separate from proposal acceptance
and makes the strategy replaceable by a future CP-SAT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence


class ProposalStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    INFEASIBLE = "INFEASIBLE"
    TIMED_OUT_WITH_SOLUTION = "TIMED_OUT_WITH_SOLUTION"
    TIMED_OUT_NO_SOLUTION = "TIMED_OUT_NO_SOLUTION"


@dataclass(frozen=True)
class AllocationWeights:
    """Centralised soft-objective priorities; hard rules are never weighted."""

    uncovered_shift: int = 1_000_000
    overtime_minute: int = 1_000
    contracted_minute_deviation: int = 10
    night_deviation: int = 500
    weekend_deviation: int = 350
    pattern_deviation: int = 250
    preference_breach: int = 200
    soft_lock_change: int = 100_000


@dataclass(frozen=True)
class AllocationStaff:
    staff_id: int
    name: str
    target_minutes: int
    actual_minutes: int = 0
    night_count: int = 0
    target_night_count: float = 0.0
    weekend_count: int = 0
    target_weekend_count: float = 0.0


@dataclass(frozen=True)
class AllocationShift:
    shift_type_id: int
    code: str
    minutes: int
    is_night: bool = False


@dataclass(frozen=True)
class ExistingAllocation:
    assignment_id: int
    staff_id: int
    day: date
    shift_type_id: int
    shift_code: str
    lock_status: str = "UNLOCKED"


@dataclass(frozen=True)
class StaffingNeed:
    day: date
    shift_type_id: int
    required_count: int


@dataclass(frozen=True)
class HardConstraintResult:
    allowed: bool
    reason_code: str = "ELIGIBLE"
    explanation: str = "Employee is eligible for this shift."


@dataclass(frozen=True)
class SoftConstraintResult:
    penalty: int = 0
    reason_codes: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProposedAssignment:
    staff_id: int
    staff_name: str
    day: date
    shift_type_id: int
    shift_code: str
    score: int
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class UncoveredShift:
    day: date
    shift_type_id: int
    shift_code: str
    missing_count: int
    reason_codes: tuple[str, ...]
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class RosterProposal:
    status: ProposalStatus
    proposed_assignments: tuple[ProposedAssignment, ...]
    retained_assignments: tuple[ExistingAllocation, ...]
    uncovered_shifts: tuple[UncoveredShift, ...]
    objective_score: int
    warnings: tuple[str, ...]
    solve_duration_seconds: float
    configuration: Mapping[str, object]
    fairness_impact: Mapping[int, int] = field(default_factory=dict)


HardConstraint = Callable[[int, date, int], HardConstraintResult]
SoftConstraint = Callable[[int, date, int], SoftConstraintResult]
SelectionCallback = Callable[[int, date, int], None]


def generate_roster_proposal(
    start_date: date,
    end_date: date,
    *,
    staff: Sequence[AllocationStaff],
    shifts: Sequence[AllocationShift],
    staffing_needs: Iterable[StaffingNeed],
    existing_assignments: Iterable[ExistingAllocation] = (),
    hard_constraint: HardConstraint,
    soft_constraint: SoftConstraint | None = None,
    on_assignment_selected: SelectionCallback | None = None,
    staff_ids: Iterable[int] | None = None,
    shift_type_ids: Iterable[int] | None = None,
    preserve_existing: bool = True,
    allow_overtime: bool = False,
    fairness_lookback_days: int = 180,
    max_solve_seconds: float = 30,
    weights: AllocationWeights | None = None,
) -> RosterProposal:
    """Return a gap-filling proposal without mutating any supplied record.

    Coverage is processed in stable date/shift order.  Each slot selects the
    lowest-cost legal candidate using current contractual, night, weekend and
    preference balances.  Hard constraints and one-duty-per-day are absolute.
    """
    started = monotonic()
    if end_date < start_date:
        raise ValueError("Proposal end date cannot be before its start date.")
    if max_solve_seconds <= 0:
        raise ValueError("Maximum solve time must be positive.")
    policy = weights or AllocationWeights()
    selected_staff_ids = set(staff_ids) if staff_ids is not None else None
    selected_shift_ids = (
        set(shift_type_ids) if shift_type_ids is not None else None
    )
    people = {
        person.staff_id: person for person in staff
        if selected_staff_ids is None or person.staff_id in selected_staff_ids
    }
    shift_by_id = {
        shift.shift_type_id: shift for shift in shifts
        if selected_shift_ids is None or shift.shift_type_id in selected_shift_ids
    }
    retained = tuple(
        row for row in existing_assignments
        if start_date <= row.day <= end_date
    )
    if not preserve_existing and retained:
        raise ValueError(
            "Initial gap filling requires preserve_existing=True; existing "
            "assignments cannot yet be regenerated."
        )

    occupied = {(row.staff_id, row.day) for row in retained}
    covered: dict[tuple[date, int], int] = {}
    for row in retained:
        key = (row.day, row.shift_type_id)
        covered[key] = covered.get(key, 0) + 1

    actual_minutes = {sid: person.actual_minutes for sid, person in people.items()}
    night_counts = {sid: person.night_count for sid, person in people.items()}
    weekend_counts = {sid: person.weekend_count for sid, person in people.items()}
    proposals: list[ProposedAssignment] = []
    uncovered: list[UncoveredShift] = []
    objective = 0
    timed_out = False

    needs = sorted(
        (
            need for need in staffing_needs
            if start_date <= need.day <= end_date
            and need.shift_type_id in shift_by_id
            and need.required_count > 0
        ),
        key=lambda need: (need.day, need.shift_type_id),
    )
    for need in needs:
        shift = shift_by_id[need.shift_type_id]
        missing = max(
            0, int(need.required_count) - covered.get((need.day, need.shift_type_id), 0)
        )
        rejection_codes: set[str] = set()
        rejection_text: set[str] = set()
        while missing:
            if monotonic() - started >= max_solve_seconds:
                timed_out = True
                break
            candidates: list[tuple[int, int, tuple[str, ...]]] = []
            for staff_id, person in people.items():
                if (staff_id, need.day) in occupied:
                    rejection_codes.add("ONE_DUTY_PER_DAY")
                    rejection_text.add("Employee already has a duty on this date.")
                    continue
                hard = hard_constraint(staff_id, need.day, shift.shift_type_id)
                if not hard.allowed:
                    rejection_codes.add(hard.reason_code)
                    rejection_text.add(hard.explanation)
                    continue
                projected_minutes = actual_minutes[staff_id] + shift.minutes
                overtime = max(0, projected_minutes - person.target_minutes)
                if overtime and not allow_overtime:
                    rejection_codes.add("OVERTIME_NOT_ALLOWED")
                    rejection_text.add(
                        "Assignment would exceed the employee's target minutes."
                    )
                    continue
                soft = (
                    soft_constraint(staff_id, need.day, shift.shift_type_id)
                    if soft_constraint else SoftConstraintResult()
                )
                score = soft.penalty * policy.preference_breach
                score += overtime * policy.overtime_minute
                score += abs(projected_minutes - person.target_minutes) * (
                    policy.contracted_minute_deviation
                )
                if shift.is_night:
                    score += round(
                        abs((night_counts[staff_id] + 1) - person.target_night_count)
                        * policy.night_deviation
                    )
                if need.day.weekday() >= 5:
                    score += round(
                        abs(
                            (weekend_counts[staff_id] + 1)
                            - person.target_weekend_count
                        ) * policy.weekend_deviation
                    )
                explanations = (
                    hard.explanation,
                    "Assignment fills an uncovered staffing requirement.",
                    "Employee has the lowest configured legal allocation cost.",
                    *soft.explanations,
                )
                candidates.append((score, staff_id, tuple(dict.fromkeys(explanations))))
            if not candidates:
                break
            score, staff_id, explanations = min(candidates)
            person = people[staff_id]
            proposals.append(ProposedAssignment(
                staff_id=staff_id,
                staff_name=person.name,
                day=need.day,
                shift_type_id=shift.shift_type_id,
                shift_code=shift.code,
                score=score,
                explanations=explanations,
            ))
            occupied.add((staff_id, need.day))
            actual_minutes[staff_id] += shift.minutes
            night_counts[staff_id] += int(shift.is_night)
            weekend_counts[staff_id] += int(need.day.weekday() >= 5)
            if on_assignment_selected is not None:
                on_assignment_selected(staff_id, need.day, shift.shift_type_id)
            objective += score
            missing -= 1
        if missing:
            objective += missing * policy.uncovered_shift
            uncovered.append(UncoveredShift(
                day=need.day,
                shift_type_id=shift.shift_type_id,
                shift_code=shift.code,
                missing_count=missing,
                reason_codes=tuple(sorted(rejection_codes)) or ("NO_CANDIDATE",),
                explanations=tuple(sorted(rejection_text)) or (
                    "No eligible employee was available for this requirement.",
                ),
            ))
        if timed_out:
            break

    if timed_out:
        status = (
            ProposalStatus.TIMED_OUT_WITH_SOLUTION
            if proposals else ProposalStatus.TIMED_OUT_NO_SOLUTION
        )
    elif uncovered and proposals:
        status = ProposalStatus.PARTIALLY_COVERED
    elif uncovered:
        status = ProposalStatus.INFEASIBLE
    else:
        status = ProposalStatus.FEASIBLE
    warnings = (
        ("Solver time limit reached before every requirement was considered.",)
        if timed_out else ()
    )
    return RosterProposal(
        status=status,
        proposed_assignments=tuple(proposals),
        retained_assignments=retained,
        uncovered_shifts=tuple(uncovered),
        objective_score=objective,
        warnings=warnings,
        solve_duration_seconds=monotonic() - started,
        configuration={
            "preserve_existing": preserve_existing,
            "allow_overtime": allow_overtime,
            "fairness_lookback_days": fairness_lookback_days,
            "max_solve_seconds": max_solve_seconds,
            "strategy": "deterministic_constraint_first_gap_fill",
            "weights": policy,
        },
        fairness_impact={
            sid: actual_minutes[sid] - person.actual_minutes
            for sid, person in people.items()
        },
    )
