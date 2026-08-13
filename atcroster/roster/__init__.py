"""Roster domain services independent of the legacy application module."""

from .cache_adapter import invalidate_month_for_day, memoize
from .annotations import parse_annotation
from .periods import is_month_locked, lock_date_for_month, month_add
from .existence import month_has_data
from .locking import lock_roster_month
from .shifts import (
    counter_group,
    counter_group_for_day,
    duration_minutes,
    ShiftCounterService,
    shift_groups_snapshot,
)
from .requirements import ensure_month_requirement, requirements_for_day
from .patterns import expand, validate
from .dates import is_sunday, parse_hhmm, parse_iso_date, parse_year_month
from .editing import LOCKED_SOURCES, cell_is_protected
from .assignments import assignment_for_day
from .mutations import set_assignment_code
from .codes import is_non_working, is_working_with_prefix, normalize_code
from .pattern_runtime import (
    PatternRuntime,
    PatternRuntimeDependencies,
    create_pattern_runtime_dependencies,
)
from .months import RosterMonthService, create_roster_month_service
from .lookups import ShiftLookupService, create_shift_lookup_service

__all__ = (
    "LOCKED_SOURCES",
    "assignment_for_day",
    "cell_is_protected",
    "counter_group",
    "counter_group_for_day",
    "duration_minutes",
    "ensure_month_requirement",
    "expand",
    "invalidate_month_for_day",
    "is_month_locked",
    "is_non_working",
    "is_sunday",
    "is_working_with_prefix",
    "lock_date_for_month",
    "lock_roster_month",
    "memoize",
    "month_add",
    "month_has_data",
    "normalize_code",
    "parse_annotation",
    "parse_hhmm",
    "parse_iso_date",
    "parse_year_month",
    "PatternRuntime",
    "PatternRuntimeDependencies",
    "create_pattern_runtime_dependencies",
    "RosterMonthService",
    "create_roster_month_service",
    "ShiftLookupService",
    "create_shift_lookup_service",
    "requirements_for_day",
    "set_assignment_code",
    "shift_groups_snapshot",
    "ShiftCounterService",
    "validate",
)
