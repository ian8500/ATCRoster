"""Roster assignment mutation orchestration."""

from __future__ import annotations

from typing import Any, Callable


def set_assignment_code(assignment: Any, code: str, source: str, note: str, invalidate_month: Callable[[Any], None], record_change: Callable[..., None]) -> Any:
    """Apply an editor override and keep cache and audit state consistent."""
    old_code = assignment.effective_code
    if old_code == code and assignment.source == source:
        return assignment
    assignment.set_editor_override(code, reason=note or "Allocation proposal", override_type="ALLOCATION")
    assignment.annotation = None
    assignment.source = source
    invalidate_month(assignment.day)
    record_change(
        "Assignment", assignment.id, "code", old_code, code,
        note=note, context_day=assignment.day,
    )
    return assignment
