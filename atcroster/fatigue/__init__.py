"""Fatigue-policy adapters for roster workflows."""

from .policy import assignment_is_fatigue_safe
from .analysis import (
    configured_findings,
    new_findings_for_proposed_assignment,
    roster_findings_matrix,
    findings_for_range,
    visible_working_findings,
    proposed_plan_findings,
    segments_for_staff as load_staff_segments,
    segments_from_assignments,
)

__all__ = (
    "assignment_is_fatigue_safe",
    "configured_findings",
    "new_findings_for_proposed_assignment",
    "roster_findings_matrix",
    "findings_for_range",
    "visible_working_findings",
    "proposed_plan_findings",
    "load_staff_segments",
    "segments_from_assignments",
)
