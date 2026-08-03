"""Classify editor overrides after deterministic baseline recalculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable


VALID_CLASSIFICATIONS = frozenset({
    "VALID", "REDUNDANT_MATCHES_BASELINE", "AFTER_UNIT_LEAVING_DATE",
    "OUTSIDE_EMPLOYMENT", "CONFLICTS_WITH_HARD_RESTRICTION", "REQUIRES_REVIEW",
})
INVALID_CLASSIFICATIONS = frozenset({
    "AFTER_UNIT_LEAVING_DATE", "OUTSIDE_EMPLOYMENT",
    "CONFLICTS_WITH_HARD_RESTRICTION", "REQUIRES_REVIEW",
})


@dataclass(frozen=True)
class OverrideFinding:
    assignment: Any
    classification: str
    description: str


@dataclass(frozen=True)
class OverrideClassificationResult:
    classified: int
    redundant: int
    invalid: int
    findings: tuple[OverrideFinding, ...]


@dataclass(frozen=True)
class OverrideClassificationDependencies:
    Assignment: Any
    Staff: Any
    ShiftType: Any
    work_pattern_service: Any


class OverrideClassificationService:
    def __init__(self, dependencies: OverrideClassificationDependencies) -> None:
        self.dependencies = dependencies

    def classify_range(
        self, unit_id: int, start: date, end: date,
        *, staff_ids: Iterable[int] = (), preserve_redundant: bool = True,
    ) -> OverrideClassificationResult:
        dep = self.dependencies
        query = dep.Assignment.query.filter(
            dep.Assignment.unit_id == unit_id,
            dep.Assignment.day >= start,
            dep.Assignment.day <= end,
            dep.Assignment.override_code.isnot(None),
        )
        selected = tuple(sorted({int(value) for value in staff_ids}))
        if selected:
            query = query.filter(dep.Assignment.staff_id.in_(selected))
        assignments = query.all()
        staff_by_id = {
            row.id: row for row in dep.Staff.query.filter(
                dep.Staff.unit_id == unit_id,
                dep.Staff.id.in_({row.staff_id for row in assignments}),
            ).all()
        } if assignments else {}
        shifts_by_code = {
            row.code: row for row in dep.ShiftType.query.filter_by(
                unit_id=unit_id, is_active=True
            ).all()
        }
        findings: list[OverrideFinding] = []
        redundant = invalid = 0
        for assignment in assignments:
            person = staff_by_id.get(assignment.staff_id)
            classification, description = self._classify(
                assignment, person, shifts_by_code.get(assignment.override_code)
            )
            assignment.override_classification = classification
            assignment.override_classified_at = date.today()
            if classification == "REDUNDANT_MATCHES_BASELINE":
                redundant += 1
                if not preserve_redundant:
                    # The default is preservation. If a unit later opts into
                    # cleanup, the materialised audit metadata remains.
                    assignment.clear_editor_override()
                    assignment.override_classification = classification
            if classification in INVALID_CLASSIFICATIONS:
                invalid += 1
                findings.append(OverrideFinding(assignment, classification, description))
        return OverrideClassificationResult(
            classified=len(assignments), redundant=redundant,
            invalid=invalid, findings=tuple(findings),
        )

    def _classify(self, assignment: Any, person: Any, shift: Any) -> tuple[str, str]:
        if not person:
            return "REQUIRES_REVIEW", "Override references an unavailable staff record."
        leaving = person.final_unit_date or person.final_operational_duty_date
        if leaving and assignment.day > leaving:
            return "AFTER_UNIT_LEAVING_DATE", "Editor override exists after the final unit date."
        if (
            person.employment_start_date and assignment.day < person.employment_start_date
        ) or (person.employment_end_date and assignment.day > person.employment_end_date):
            return "OUTSIDE_EMPLOYMENT", "Editor override falls outside the employment period."
        if assignment.override_code == assignment.generated_code:
            return "REDUNDANT_MATCHES_BASELINE", "Override now matches the generated baseline."
        if shift and shift.is_working:
            eligibility = self.dependencies.work_pattern_service.is_staff_eligible_for_shift(
                person.id, assignment.day, shift.id, existing_assignment=True
            )
            if not eligibility.eligible:
                return "CONFLICTS_WITH_HARD_RESTRICTION", eligibility.explanation
        return "VALID", "Override remains distinct from the generated baseline."
