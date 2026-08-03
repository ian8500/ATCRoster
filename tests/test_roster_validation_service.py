from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_validation_service import (  # noqa: E402
    Severity, ValidationAssignment, validate_roster,
)


def test_validation_returns_structured_blocking_and_coverage_issues():
    day = date(2026, 8, 3)
    rows = [
        ValidationAssignment(1, 10, day, 2, "M", "M"),
        ValidationAssignment(2, 10, day, 2, "M", "M"),
    ]
    summary = validate_roster(
        rows,
        eligibility_for=lambda row: (False, "NO_EARLY_RULE", "Hard restriction"),
        absence_for=lambda staff_id, on_day: "approved leave",
        medical_expiry_for=lambda staff_id: date(2026, 8, 2),
        requirements={(day, "M"): 3},
    )
    codes = {issue.code for issue in summary.issues}
    assert {"DUPLICATE_ASSIGNMENT", "APPROVED_ABSENCE", "MEDICAL_INVALID", "NO_EARLY_RULE", "UNCOVERED_REQUIREMENT"} <= codes
    assert summary.has_blocking_issues
    assert summary.counts_by_severity[Severity.CRITICAL.value] == 1


def test_warning_only_overstaffing_does_not_block():
    day = date(2026, 8, 3)
    summary = validate_roster(
        [ValidationAssignment(1, 10, day, 2, "M", "M")],
        eligibility_for=lambda row: (True, "ELIGIBLE", ""),
        absence_for=lambda staff_id, on_day: None,
        medical_expiry_for=lambda staff_id: day,
        requirements={(day, "M"): 0},
    )
    assert [issue.code for issue in summary.issues] == ["OVERSTAFFED"]
    assert not summary.has_blocking_issues
