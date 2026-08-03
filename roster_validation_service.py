"""Structured, reusable validation for live rosters and future proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Iterable, Mapping


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ValidationAssignment:
    assignment_id: int | None
    staff_id: int
    day: date
    shift_type_id: int
    shift_code: str
    coverage_group: str | None


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    day: date | None = None
    staff_id: int | None = None
    shift_type_id: int | None = None
    assignment_id: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationSummary:
    issues: tuple[ValidationIssue, ...]

    @property
    def has_blocking_issues(self) -> bool:
        return any(issue.severity in {Severity.ERROR, Severity.CRITICAL} for issue in self.issues)

    @property
    def counts_by_severity(self) -> dict[str, int]:
        counts = Counter(issue.severity.value for issue in self.issues)
        return {severity.value: counts[severity.value] for severity in Severity}


def validate_roster(
    assignments: Iterable[ValidationAssignment],
    *,
    eligibility_for: Callable[[ValidationAssignment], tuple[bool, str, str]],
    absence_for: Callable[[int, date], str | None],
    medical_expiry_for: Callable[[int], date | None],
    requirements: Mapping[tuple[date, str], int],
) -> ValidationSummary:
    rows = sorted(assignments, key=lambda row: (row.day, row.staff_id, row.assignment_id or 0))
    issues: list[ValidationIssue] = []
    seen: set[tuple[int, date]] = set()
    coverage: Counter[tuple[date, str]] = Counter()
    for row in rows:
        key = (row.staff_id, row.day)
        if key in seen:
            issues.append(ValidationIssue(Severity.CRITICAL, "DUPLICATE_ASSIGNMENT", "More than one duty exists for this employee on this date.", row.day, row.staff_id, row.shift_type_id, row.assignment_id))
        seen.add(key)
        absence = absence_for(row.staff_id, row.day)
        if absence:
            issues.append(ValidationIssue(Severity.ERROR, "APPROVED_ABSENCE", f"Employee has {absence} recorded on this date.", row.day, row.staff_id, row.shift_type_id, row.assignment_id))
        medical_expiry = medical_expiry_for(row.staff_id)
        if medical_expiry is None or medical_expiry < row.day:
            issues.append(ValidationIssue(Severity.ERROR, "MEDICAL_INVALID", "Employee does not have an in-date medical.", row.day, row.staff_id, row.shift_type_id, row.assignment_id))
        eligible, code, explanation = eligibility_for(row)
        if not eligible:
            issues.append(ValidationIssue(Severity.ERROR, code, explanation, row.day, row.staff_id, row.shift_type_id, row.assignment_id))
        if row.coverage_group:
            coverage[(row.day, row.coverage_group)] += 1
    for (day, group), required in sorted(requirements.items()):
        actual = coverage[(day, group)]
        if actual < required:
            issues.append(ValidationIssue(Severity.ERROR, "UNCOVERED_REQUIREMENT", f"{group} requires {required} staff but has {actual}.", day, metadata={"group": group, "required": required, "actual": actual}))
        elif actual > required:
            issues.append(ValidationIssue(Severity.WARNING, "OVERSTAFFED", f"{group} requires {required} staff but has {actual}.", day, metadata={"group": group, "required": required, "actual": actual}))
    return ValidationSummary(tuple(issues))
