"""Read-only validation of assigned roster duties against staff constraints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class RosterValidationFinding:
    staff_id: int
    staff_name: str
    day: date
    shift_code: str
    severity: str
    reason_code: str
    explanation: str

    @property
    def blocks_publication(self) -> bool:
        return self.severity == "blocking"


@dataclass(frozen=True)
class RosterValidationResult:
    findings: tuple[RosterValidationFinding, ...]

    @property
    def blocking_count(self) -> int:
        return sum(item.blocks_publication for item in self.findings)

    @property
    def advisory_count(self) -> int:
        return len(self.findings) - self.blocking_count

    @property
    def can_publish(self) -> bool:
        return self.blocking_count == 0

    def by_cell(self) -> dict[tuple[int, date], tuple[RosterValidationFinding, ...]]:
        grouped: dict[tuple[int, date], list[RosterValidationFinding]] = {}
        for finding in self.findings:
            grouped.setdefault((finding.staff_id, finding.day), []).append(finding)
        return {key: tuple(value) for key, value in grouped.items()}


@dataclass(frozen=True)
class RosterValidationDependencies:
    Staff: Any
    ShiftType: Any
    Assignment: Any
    StaffPatternAssignment: Any
    StaffRule: Any
    work_pattern_service: Any


class RosterValidationService:
    def __init__(self, dependencies: RosterValidationDependencies) -> None:
        self.dependencies = dependencies

    def validate_range(
        self, unit_id: int, start: date, end: date
    ) -> RosterValidationResult:
        """Validate inclusive dates without mutating any roster record."""
        relevant_staff_ids = self._relevant_staff_ids(unit_id, start, end)
        if not relevant_staff_ids:
            return RosterValidationResult(())
        staff = {
            row.id: row
            for row in self.dependencies.Staff.query.filter(
                self.dependencies.Staff.unit_id == unit_id,
                self.dependencies.Staff.id.in_(relevant_staff_ids),
            ).all()
        }
        shifts = {
            row.code.upper(): row
            for row in self.dependencies.ShiftType.query.filter_by(
                unit_id=unit_id, is_active=True, is_working=True
            ).all()
        }
        assignments = self.dependencies.Assignment.query.filter(
            self.dependencies.Assignment.unit_id == unit_id,
            self.dependencies.Assignment.staff_id.in_(relevant_staff_ids),
            self.dependencies.Assignment.day >= start,
            self.dependencies.Assignment.day <= end,
        ).order_by(
            self.dependencies.Assignment.day,
            self.dependencies.Assignment.staff_id,
        ).all()
        eligibility_context = (
            self.dependencies.work_pattern_service.build_eligibility_context(
                unit_id, relevant_staff_ids, start, end
            )
        )
        findings: list[RosterValidationFinding] = []
        for assignment in assignments:
            person = staff.get(assignment.staff_id)
            shift = shifts.get((assignment.code or "").upper())
            if not person or not shift:
                continue
            result = self.dependencies.work_pattern_service.is_staff_eligible_for_shift(
                person.id, assignment.day, shift.id,
                existing_assignment=True,
                context=eligibility_context,
            )
            if not result.eligible:
                findings.append(RosterValidationFinding(
                    staff_id=person.id,
                    staff_name=person.name,
                    day=assignment.day,
                    shift_code=shift.code,
                    severity="blocking",
                    reason_code=result.reason_code,
                    explanation=result.explanation,
                ))
                continue
            for reason in result.reasons:
                if not reason.code.startswith("SOFT_"):
                    continue
                findings.append(RosterValidationFinding(
                    staff_id=person.id,
                    staff_name=person.name,
                    day=assignment.day,
                    shift_code=shift.code,
                    severity="advisory",
                    reason_code=reason.code,
                    explanation=reason.explanation,
                ))
        return RosterValidationResult(tuple(findings))

    def _relevant_staff_ids(
        self, unit_id: int, start: date, end: date
    ) -> set[int]:
        assignment_ids = {
            row[0]
            for row in self.dependencies.StaffPatternAssignment.query.with_entities(
                self.dependencies.StaffPatternAssignment.staff_id
            ).filter(
                self.dependencies.StaffPatternAssignment.unit_id == unit_id,
                self.dependencies.StaffPatternAssignment.effective_from <= end,
                (
                    self.dependencies.StaffPatternAssignment.effective_to.is_(None)
                    | (self.dependencies.StaffPatternAssignment.effective_to >= start)
                ),
            ).all()
        }
        rule_ids = {
            row[0]
            for row in self.dependencies.StaffRule.query.with_entities(
                self.dependencies.StaffRule.staff_id
            ).filter(
                self.dependencies.StaffRule.unit_id == unit_id,
                self.dependencies.StaffRule.is_active.is_(True),
                self.dependencies.StaffRule.effective_from <= end,
                (
                    self.dependencies.StaffRule.effective_to.is_(None)
                    | (self.dependencies.StaffRule.effective_to >= start)
                ),
            ).all()
        }
        return assignment_ids | rule_ids
