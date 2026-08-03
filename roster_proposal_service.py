"""Database adapter and review workflow for automatic roster proposals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Callable

from roster_allocation_service import (
    AllocationShift,
    AllocationStaff,
    ExistingAllocation,
    HardConstraintResult,
    SoftConstraintResult,
    StaffingNeed,
    generate_roster_proposal,
)


@dataclass(frozen=True)
class RosterProposalDependencies:
    db: Any
    Staff: Any
    ShiftType: Any
    Assignment: Any
    Sickness: Any
    Requirement: Any
    SpecialRequirement: Any
    RosterProposal: Any
    RosterProposalAssignment: Any
    ChangeLog: Any
    work_pattern_service: Any
    requirements_for_day: Callable[..., dict[str, int]]
    shift_group_for_day: Callable[[str, date, int], str | None]
    shift_minutes: Callable[[Any], int]
    staff_is_countable_on: Callable[[Any, date], bool]
    staff_has_qualification: Callable[[Any, Any, date], bool]
    would_trigger_fatigue: Callable[[Any, date, str, dict[date, str]], list[str]]
    compute_fairness_range: Callable[[date, date], tuple[list[Any], dict[str, Any]]]
    utcnow: Callable[[], Any]


def index_fairness_rows(
    result: tuple[list[Any], dict[str, Any]],
) -> dict[int, Any]:
    """Index the shared fairness helper's ``(rows, totals)`` result."""
    rows, _totals = result
    return {row.staff_id: row for row in rows}


class RosterProposalService:
    def __init__(self, dependencies: RosterProposalDependencies) -> None:
        self.dependencies = dependencies

    def generate(
        self,
        unit_id: int,
        start_date: date,
        end_date: date,
        actor_id: int,
        *,
        allow_overtime: bool = False,
        fairness_lookback_days: int = 180,
        max_solve_seconds: float = 30,
    ) -> Any:
        if end_date < start_date or (end_date - start_date).days > 92:
            raise ValueError("Choose a proposal period between 1 and 93 days.")
        staff_rows = self.dependencies.Staff.query.filter_by(
            unit_id=unit_id, is_operational=True, membership_status="active"
        ).filter(self.dependencies.Staff.role != "position_monitor").order_by(
            self.dependencies.Staff.id
        ).all()
        shift_rows = self.dependencies.ShiftType.query.filter_by(
            unit_id=unit_id, is_active=True, is_working=True
        ).order_by(self.dependencies.ShiftType.id).all()
        shift_by_code = {(row.code or "").upper(): row for row in shift_rows}
        shift_by_id = {row.id: row for row in shift_rows}
        days = [
            start_date + timedelta(days=offset)
            for offset in range((end_date - start_date).days + 1)
        ]
        requirements = {
            (row.year, row.month): row
            for row in self.dependencies.Requirement.query.filter(
                self.dependencies.Requirement.unit_id == unit_id
            ).all()
        }
        specials = {
            row.day: row for row in self.dependencies.SpecialRequirement.query.filter(
                self.dependencies.SpecialRequirement.unit_id == unit_id,
                self.dependencies.SpecialRequirement.day >= start_date,
                self.dependencies.SpecialRequirement.day <= end_date,
            ).all()
        }
        canonical_shift: dict[tuple[date, str], Any] = {}
        for day in days:
            for shift in shift_rows:
                group = self.dependencies.shift_group_for_day(
                    shift.code, day, unit_id
                )
                if group and (
                    (day, group) not in canonical_shift
                    or shift.code.upper() == group
                ):
                    canonical_shift[(day, group)] = shift
        needs = []
        for day in days:
            requirement = requirements.get((day.year, day.month))
            counts = self.dependencies.requirements_for_day(
                requirement, day, specials.get(day)
            )
            for group, required_count in counts.items():
                shift = canonical_shift.get((day, group))
                if shift and required_count:
                    needs.append(StaffingNeed(day, shift.id, required_count))

        assignment_rows = self.dependencies.Assignment.query.filter(
            self.dependencies.Assignment.unit_id == unit_id,
            self.dependencies.Assignment.day >= start_date,
            self.dependencies.Assignment.day <= end_date,
        ).all()
        existing = []
        for row in assignment_rows:
            shift = shift_by_code.get((row.code or "").upper())
            group = (
                self.dependencies.shift_group_for_day(row.code, row.day, unit_id)
                if shift else None
            )
            coverage_shift = canonical_shift.get((row.day, group)) if group else None
            existing.append(ExistingAllocation(
                assignment_id=row.id,
                staff_id=row.staff_id,
                day=row.day,
                shift_type_id=coverage_shift.id if coverage_shift else 0,
                shift_code=row.code,
                lock_status=row.lock_status or "UNLOCKED",
            ))
        lookback_start = start_date - timedelta(days=fairness_lookback_days)
        fairness = index_fairness_rows(
            self.dependencies.compute_fairness_range(lookback_start, end_date)
        )
        people = [
            AllocationStaff(
                staff_id=row.id,
                name=row.name,
                target_minutes=max(
                    int(getattr(fairness.get(row.id), "target_minutes", 0)),
                    int(getattr(fairness.get(row.id), "actual_minutes", 0)),
                ),
                actual_minutes=int(
                    getattr(fairness.get(row.id), "actual_minutes", 0)
                ),
                night_count=int(getattr(fairness.get(row.id), "night_count", 0)),
                target_night_count=float(
                    getattr(fairness.get(row.id), "target_night_count", 0)
                ),
                weekend_count=int(
                    getattr(fairness.get(row.id), "weekend_count", 0)
                ),
                target_weekend_count=float(
                    getattr(fairness.get(row.id), "target_weekend_count", 0)
                ),
            )
            for row in staff_rows
        ]
        allocation_shifts = [
            AllocationShift(
                shift.id,
                shift.code,
                self.dependencies.shift_minutes(shift),
                any(
                    self.dependencies.shift_group_for_day(
                        shift.code, day, unit_id
                    ) == "N"
                    for day in days
                ),
            )
            for shift in shift_rows
        ]
        staff_by_id = {row.id: row for row in staff_rows}
        sickness = self.dependencies.Sickness.query.filter(
            self.dependencies.Sickness.unit_id == unit_id,
            self.dependencies.Sickness.start <= end_date,
            self.dependencies.Sickness.end >= start_date,
        ).all()
        eligibility_context = (
            self.dependencies.work_pattern_service.build_eligibility_context(
                unit_id, list(staff_by_id), start_date, end_date
            )
        )
        planned_codes: dict[int, dict[date, str]] = {
            staff_id: {} for staff_id in staff_by_id
        }

        def hard_constraint(staff_id: int, day: date, shift_id: int):
            person = staff_by_id[staff_id]
            shift = shift_by_id[shift_id]
            if any(
                row.staff_id == staff_id and row.start <= day <= row.end
                for row in sickness
            ):
                return HardConstraintResult(
                    False, "SICKNESS", "Employee is recorded sick on this date."
                )
            if not self.dependencies.staff_is_countable_on(person, day):
                return HardConstraintResult(
                    False, "MEDICAL_OR_ENDORSEMENT_INVALID",
                    "Employee does not have the required current medical and endorsement.",
                )
            if shift.required_qualification and not (
                self.dependencies.staff_has_qualification(person, shift, day)
            ):
                return HardConstraintResult(
                    False, "QUALIFICATION_INVALID",
                    "Employee does not hold the qualification required for this shift.",
                )
            eligibility = (
                self.dependencies.work_pattern_service.is_staff_eligible_for_shift(
                    staff_id, day, shift_id, context=eligibility_context
                )
            )
            if not eligibility.eligible:
                return HardConstraintResult(
                    False, eligibility.reason_code, eligibility.explanation
                )
            fatigue = self.dependencies.would_trigger_fatigue(
                person, day, shift.code, planned_codes[staff_id]
            )
            if fatigue:
                return HardConstraintResult(
                    False, "FATIGUE_RULE", "; ".join(fatigue)
                )
            return HardConstraintResult(True)

        def soft_constraint(staff_id: int, day: date, shift_id: int):
            result = self.dependencies.work_pattern_service.is_staff_eligible_for_shift(
                staff_id, day, shift_id, context=eligibility_context
            )
            reasons = tuple(
                reason for reason in result.reasons if reason.code.startswith("SOFT_")
            )
            return SoftConstraintResult(
                penalty=result.soft_penalty,
                reason_codes=tuple(reason.code for reason in reasons),
                explanations=tuple(reason.explanation for reason in reasons),
            )

        result = generate_roster_proposal(
            start_date,
            end_date,
            staff=people,
            shifts=allocation_shifts,
            staffing_needs=needs,
            existing_assignments=existing,
            hard_constraint=hard_constraint,
            soft_constraint=soft_constraint,
            on_assignment_selected=lambda staff_id, day, shift_id: (
                planned_codes[staff_id].__setitem__(day, shift_by_id[shift_id].code)
            ),
            allow_overtime=allow_overtime,
            fairness_lookback_days=fairness_lookback_days,
            max_solve_seconds=max_solve_seconds,
        )
        proposal = self.dependencies.RosterProposal(
            unit_id=unit_id,
            start_date=start_date,
            end_date=end_date,
            status=result.status.value,
            objective_score=result.objective_score,
            configuration_json=json.dumps(
                {
                    key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
                    for key, value in result.configuration.items()
                }, default=str,
            ),
            warnings_json=json.dumps(result.warnings),
            uncovered_json=json.dumps(
                [asdict(row) for row in result.uncovered_shifts], default=str
            ),
            created_by_user_id=actor_id,
        )
        self.dependencies.db.session.add(proposal)
        self.dependencies.db.session.flush()
        for row in result.proposed_assignments:
            self.dependencies.db.session.add(
                self.dependencies.RosterProposalAssignment(
                    unit_id=unit_id,
                    proposal_id=proposal.id,
                    staff_id=row.staff_id,
                    day=row.day,
                    shift_type_id=row.shift_type_id,
                    shift_code=row.shift_code,
                    score=row.score,
                    explanations_json=json.dumps(row.explanations),
                )
            )
        self.dependencies.db.session.commit()
        return proposal

    def apply(self, proposal: Any, actor_id: int) -> int:
        if proposal.workflow_state != "draft":
            raise ValueError("Only a draft proposal can be applied.")
        rows = self.dependencies.RosterProposalAssignment.query.filter_by(
            unit_id=proposal.unit_id,
            proposal_id=proposal.id,
            review_state="accepted",
        ).order_by(self.dependencies.RosterProposalAssignment.id).all()
        applied = 0
        for row in rows:
            existing = self.dependencies.Assignment.query.filter_by(
                unit_id=proposal.unit_id, staff_id=row.staff_id, day=row.day
            ).with_for_update().first()
            if existing:
                raise ValueError(
                    "The live roster changed after proposal generation; regenerate it."
                )
            assignment = self.dependencies.Assignment(
                unit_id=proposal.unit_id,
                staff_id=row.staff_id,
                day=row.day,
                code=row.shift_code,
                source="proposal",
                note=f"Accepted from proposal {proposal.id}",
            )
            self.dependencies.db.session.add(assignment)
            self.dependencies.db.session.flush()
            row.applied_assignment_id = assignment.id
            row.review_state = "applied"
            self.dependencies.db.session.add(self.dependencies.ChangeLog(
                when=self.dependencies.utcnow(),
                who_user_id=actor_id,
                entity_type="Assignment",
                entity_id=assignment.id,
                field="code",
                old_value="",
                new_value=row.shift_code,
                context_month=f"{row.day.year:04d}-{row.day.month:02d}",
                note=f"Accepted from automatic proposal {proposal.id}",
            ))
            applied += 1
        proposal.workflow_state = "applied"
        proposal.applied_by_user_id = actor_id
        proposal.applied_at = self.dependencies.utcnow()
        self.dependencies.db.session.commit()
        return applied
