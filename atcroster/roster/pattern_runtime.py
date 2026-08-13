"""Bound effective-watch and recurring-pattern resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Callable

from atcroster.workforce.watches import (
    effective_watch,
    watch_id_for_staff_on,
    watch_ids_for_staff_on,
)

from .patterns import (
    code_for_day,
    expand,
    leave_code_for,
    night_active_on,
    pattern_context,
    unit_pattern_context,
    validate,
)


@dataclass(frozen=True)
class PatternRuntimeDependencies:
    db: Any
    Staff: Any
    StaffWatchHistory: Any
    authenticated_unit_id: Callable[[], int]
    settings_snapshot: Callable[[int], dict[str, Any]]
    expand_pattern: Callable[[str | None], list[str]]
    validated_pattern: Callable[[str | None], list[str]]
    default_pattern: str
    today: Callable[[], date] = date.today


def create_pattern_runtime_dependencies(
    *, db: Any, operational_models: Any, **services: Any
) -> PatternRuntimeDependencies:
    """Bind pattern resolution records within the roster domain."""
    return PatternRuntimeDependencies(
        db=db,
        Staff=operational_models.Staff,
        StaffWatchHistory=operational_models.StaffWatchHistory,
        **services,
    )


class PatternRuntime:
    """Resolve tenant-scoped watch membership and dated roster patterns."""

    def __init__(self, dependencies: PatternRuntimeDependencies) -> None:
        self.dependencies = dependencies

    def watch_id(self, staff_id: int, on_date: date) -> int | None:
        return self.cached_watch_id(
            self.dependencies.authenticated_unit_id(), staff_id, on_date
        )

    def watch_ids(self, staff: list[Any], on_date: date) -> dict[int, int | None]:
        deps = self.dependencies
        return watch_ids_for_staff_on(
            deps.StaffWatchHistory, staff, deps.authenticated_unit_id(), on_date
        )

    @lru_cache(maxsize=4096)
    def cached_watch_id(
        self, unit_id: int, staff_id: int, on_date: date
    ) -> int | None:
        deps = self.dependencies
        return watch_id_for_staff_on(
            deps.db, deps.StaffWatchHistory, deps.Staff,
            unit_id, staff_id, on_date,
        )

    def expand(self, raw_value: str | None) -> list[str]:
        return expand(raw_value, self.dependencies.expand_pattern)

    def validate(self, raw_value: str | None) -> list[str]:
        return validate(raw_value, self.dependencies.validated_pattern)

    def effective_watch(self, staff: Any, on_date: date) -> Any:
        deps = self.dependencies
        return effective_watch(deps.db, deps.StaffWatchHistory, staff, on_date)

    def unit_context(self, unit_id: int) -> tuple[list[str], date]:
        deps = self.dependencies
        return unit_pattern_context(
            unit_id,
            settings_snapshot=deps.settings_snapshot,
            validate_pattern=self.validate,
            default_pattern=deps.default_pattern,
        )

    def context(self, staff: Any, on_date: date) -> tuple[list[str], date]:
        deps = self.dependencies
        return pattern_context(
            staff,
            on_date,
            db=deps.db,
            StaffWatchHistory=deps.StaffWatchHistory,
            effective_watch=self.effective_watch,
            validate_pattern=self.validate,
            unit_context=self.unit_context,
        )

    def pattern_for(self, staff: Any, on_date: date | None = None) -> list[str]:
        return self.context(staff, on_date or self.dependencies.today())[0]

    def night_active(self, unit_id: int, on_date: date) -> bool:
        return night_active_on(
            unit_id, on_date, settings_snapshot=self.dependencies.settings_snapshot
        )

    @staticmethod
    def leave_for(staff: Any, on_date: date) -> Any:
        return leave_code_for(staff, on_date)

    def code_for(self, staff: Any, on_date: date) -> str:
        return code_for_day(
            staff, on_date,
            resolve_context=self.context,
            night_active=self.night_active,
        )

    def effective_watch_id(self, staff: Any, on_date: date) -> int | None:
        watch = self.effective_watch(staff, on_date)
        return watch.id if watch else None

    def cycle_day(self, staff: Any, on_date: date) -> int | None:
        pattern, anchor = self.context(staff, on_date)
        if not pattern:
            return None
        return ((on_date - anchor).days % len(pattern)) + 1
