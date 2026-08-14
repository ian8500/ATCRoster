"""Roster assignment edit-protection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from atcroster.fatigue import assignment_is_fatigue_safe
from atcroster.workforce import has_leave_or_sickness

from .assignments import assignment_for_day
from .codes import is_non_working, is_working_with_prefix, normalize_code
from .dates import is_sunday
from .mutations import set_assignment_code


LOCKED_SOURCES = frozenset({"manual", "leave", "sickness"})


def cell_is_protected(assignment: Any) -> bool:
    """Return whether a materialised assignment may not be overwritten."""
    return bool(assignment.effective_code and assignment.source in LOCKED_SOURCES)


@dataclass(frozen=True)
class RosterEditingDependencies:
    db: Any
    Assignment: Any
    Leave: Any
    Sickness: Any
    current_unit_id: Callable[[], int]
    invalidate_month_for_day: Callable[[Any], None]
    log_change: Callable[..., None]
    would_trigger_fatigue: Callable[..., Any]
    non_working_codes: Callable[[], set[str]]
    get_shift: Callable[[str], Any]


def create_roster_editing_dependencies(
    *, db: Any, operational_models: Any, **services: Any
) -> RosterEditingDependencies:
    """Bind roster-editing records within the roster domain."""
    return RosterEditingDependencies(
        db=db,
        Assignment=operational_models.Assignment,
        Leave=operational_models.Leave,
        Sickness=operational_models.Sickness,
        **services,
    )


class RosterEditingRuntime:
    """Own assignment mutation guards and shift-code editing policy."""

    def __init__(self, dependencies: RosterEditingDependencies):
        self.dependencies = dependencies

    def assignment(self, staff_id: int, day: Any):
        deps = self.dependencies
        return assignment_for_day(
            deps.db, deps.Assignment, deps.current_unit_id(), staff_id, day
        )

    @staticmethod
    def cell_is_protected(assignment: Any) -> bool:
        return cell_is_protected(assignment)

    def set_code(
        self,
        assignment: Any,
        code: str,
        source: str,
        note: str = "",
        ctx_month: str | None = None,
    ):
        del ctx_month
        deps = self.dependencies
        return set_assignment_code(
            assignment,
            code,
            source,
            note,
            deps.invalidate_month_for_day,
            deps.log_change,
        )

    def has_leave_or_sickness(self, staff_id: int, day: Any) -> bool:
        deps = self.dependencies
        return has_leave_or_sickness(deps.Leave, deps.Sickness, staff_id, day)

    def fatigue_ok(self, staff: Any, day: Any, code: str) -> bool:
        return assignment_is_fatigue_safe(
            staff, day, code, self.dependencies.would_trigger_fatigue
        )

    @staticmethod
    def weekday_is_sunday(day: Any) -> bool:
        return is_sunday(day)

    @staticmethod
    def normalize_code(code: object) -> str:
        return normalize_code(code)

    def code_is_non_working(self, code: str) -> bool:
        return is_non_working(code, self.dependencies.non_working_codes)

    def working_code_prefix(self, code: str, prefix: str) -> bool:
        deps = self.dependencies
        return is_working_with_prefix(
            code, prefix, deps.non_working_codes, deps.get_shift
        )

    def working_day_code(self, code: str) -> bool:
        return self.working_code_prefix(code, "D")

    def working_morning_code(self, code: str) -> bool:
        return self.working_code_prefix(code, "M")

    def working_night_code(self, code: str) -> bool:
        return self.working_code_prefix(code, "N")
