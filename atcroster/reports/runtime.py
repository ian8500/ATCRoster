"""Bound reporting computations used by report and roster routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from reporting import (
    compute_annotation_metrics,
    current_leave_year_window,
    financial_year_start,
    group_consecutive_days,
    leave_summary_for_month,
)

from atcroster.roster.fairness import FairnessDependencies, FairnessReportService


@dataclass(frozen=True)
class ReportingRuntimeDependencies:
    Assignment: Any
    Staff: Any
    Watch: Any
    BankHoliday: Any
    ChangeLog: Any
    ShiftType: Any
    FairnessAssignment: Any
    FairnessStaff: Any
    current_unit_id: Callable[[], int]
    annotation_snapshot: Callable[[int], dict[str, Any]]
    parse_annotation: Callable[[str], Any]
    work_pattern_service: Callable[[], Any]
    code_from_pattern: Callable[..., Any]
    shift_duration_minutes: Callable[..., int]
    calculate_fairness: Callable[..., Any]
    month_range: Callable[..., Any]
    get_absence_types: Callable[..., list[dict[str, Any]]]


class ReportingRuntime:
    """Own annotation, fairness, leave, and financial-year report calculations."""

    def __init__(self, dependencies: ReportingRuntimeDependencies):
        self.dependencies = dependencies

    def compute_metrics(
        self, start_day: Any, end_day: Any, watch_id: int | None = None
    ):
        deps = self.dependencies
        return compute_annotation_metrics(
            start_day,
            end_day,
            watch_id=watch_id,
            Assignment=deps.Assignment,
            Staff=deps.Staff,
            Watch=deps.Watch,
            annotation_items=deps.annotation_snapshot(int(deps.current_unit_id() or 1))[
                "items"
            ],
            parse_annotation=deps.parse_annotation,
        )

    def compute_fairness(self, start_day: Any, end_day: Any):
        deps = self.dependencies
        return FairnessReportService(
            FairnessDependencies(
                Assignment=deps.Assignment,
                BankHoliday=deps.BankHoliday,
                ChangeLog=deps.ChangeLog,
                ShiftType=deps.ShiftType,
                Staff=deps.Staff,
                FairnessAssignment=deps.FairnessAssignment,
                FairnessStaff=deps.FairnessStaff,
                current_unit_id=deps.current_unit_id,
                work_pattern_service=deps.work_pattern_service(),
                code_from_pattern=deps.code_from_pattern,
                shift_duration_minutes=deps.shift_duration_minutes,
                calculate_fairness=deps.calculate_fairness,
            )
        ).compute(start_day, end_day)

    @staticmethod
    def financial_year_start(day: Any):
        return financial_year_start(day)

    def leave_summary(self, year: int, month: int, watch_id: int | None = None):
        deps = self.dependencies
        return leave_summary_for_month(
            year,
            month,
            watch_id,
            unit_id=deps.current_unit_id(),
            Assignment=deps.Assignment,
            Staff=deps.Staff,
            Watch=deps.Watch,
            month_range=deps.month_range,
            active_leave_types=deps.get_absence_types("leave", active_only=True),
        )

    @staticmethod
    def current_leave_year_window(staff: Any, today: Any = None):
        return current_leave_year_window(staff, today)

    @staticmethod
    def group_consecutive_days(days: set[Any]):
        return group_consecutive_days(days)
