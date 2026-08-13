"""Bound fatigue analysis runtime for roster workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .analysis import (
    configured_findings,
    findings_for_range,
    new_findings_for_proposed_assignment,
    proposed_plan_findings,
    roster_findings_matrix,
    segments_for_staff,
    segments_from_assignments,
    visible_working_findings,
)


@dataclass(frozen=True)
class FatigueRuntimeDependencies:
    Assignment: Any
    get_shift: Callable[..., Any]
    is_working: Callable[..., bool]
    span: Callable[..., Any]
    is_night_duty: Callable[..., bool]
    is_early_start: Callable[..., tuple[bool, bool]]
    is_morning_duty: Callable[..., bool]
    analyze_segments: Callable[..., Any]
    custom_fatigue_flags: Callable[..., Any]
    fatigue_rule_config: Callable[[int], dict[str, Any]]


def create_fatigue_runtime_dependencies(
    *, operational_models: Any, **services: Any
) -> FatigueRuntimeDependencies:
    """Bind fatigue analysis to operational assignment records."""
    return FatigueRuntimeDependencies(
        Assignment=operational_models.Assignment,
        **services,
    )


class FatigueRuntime:
    """Own segment construction and fatigue finding projections."""

    def __init__(self, dependencies: FatigueRuntimeDependencies):
        self.dependencies = dependencies

    def segments_from_assignments(self, staff: Any, assignments: Any, definitions: Any):
        deps = self.dependencies
        return segments_from_assignments(
            staff,
            assignments,
            definitions,
            get_shift=deps.get_shift,
            is_working=deps.is_working,
            span=deps.span,
            is_night_duty=deps.is_night_duty,
            is_early_start=deps.is_early_start,
            is_morning_duty=deps.is_morning_duty,
        )

    def configured_findings(self, segments: Any, config: Any, observation_start: Any):
        deps = self.dependencies
        return configured_findings(
            segments,
            config,
            observation_start,
            analyze_segments=deps.analyze_segments,
            custom_fatigue_flags=deps.custom_fatigue_flags,
        )

    def segments_for_staff(self, staff: Any, start_day: Any, end_day: Any):
        deps = self.dependencies
        return segments_for_staff(
            staff,
            start_day,
            end_day,
            Assignment=deps.Assignment,
            fatigue_rule_config=deps.fatigue_rule_config,
            build_segments=self.segments_from_assignments,
        )

    def findings_for_range(self, staff: Any, day_list: Any, lookback_days: int = 30):
        return findings_for_range(
            staff,
            day_list,
            lookback_days=lookback_days,
            segments_for_staff=self.segments_for_staff,
            fatigue_rule_config=self.dependencies.fatigue_rule_config,
            configured_findings=self.configured_findings,
        )

    def visible_findings(
        self, staff: Any, day_list: Any, code_by_day: Any, unit_id: int | None = None
    ):
        return visible_working_findings(
            staff,
            day_list,
            code_by_day,
            int(unit_id or staff.unit_id),
            range_findings=self.findings_for_range,
            get_shift=self.dependencies.get_shift,
        )

    def findings_matrix(
        self, staff: Any, day_list: Any, code_by_staff: Any, unit_id: int
    ):
        deps = self.dependencies
        return roster_findings_matrix(
            staff,
            day_list,
            code_by_staff,
            unit_id,
            Assignment=deps.Assignment,
            segments_from_assignments=self.segments_from_assignments,
            fatigue_rule_config=deps.fatigue_rule_config,
            configured_findings=self.configured_findings,
            get_shift=deps.get_shift,
        )

    def proposed_plan(
        self, staff: Any, day: Any, code: str, proposed_codes: Any = None
    ):
        deps = self.dependencies
        return proposed_plan_findings(
            staff,
            day,
            code,
            proposed_codes or {},
            get_shift=deps.get_shift,
            is_working=deps.is_working,
            segments_for_staff=self.segments_for_staff,
            fatigue_rule_config=deps.fatigue_rule_config,
            configured_findings=self.configured_findings,
            span=deps.span,
            is_early_start=deps.is_early_start,
            is_night_duty=deps.is_night_duty,
            is_morning_duty=deps.is_morning_duty,
        )

    def new_findings(
        self,
        staff: Any,
        proposed_day: Any,
        proposed_code: str,
        lookback_days: int = 30,
        lookahead_days: int = 14,
    ):
        deps = self.dependencies
        return new_findings_for_proposed_assignment(
            staff,
            proposed_day,
            proposed_code,
            lookback_days=lookback_days,
            lookahead_days=lookahead_days,
            get_shift=deps.get_shift,
            is_working=deps.is_working,
            segments_for_staff=self.segments_for_staff,
            fatigue_rule_config=deps.fatigue_rule_config,
            configured_fatigue_findings=self.configured_findings,
            span=deps.span,
            is_early_start=deps.is_early_start,
            is_night_duty=deps.is_night_duty,
            is_morning_duty=deps.is_morning_duty,
        )
