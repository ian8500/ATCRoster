"""Compatibility-facing fatigue policy composed from the fatigue runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from .analysis import proposed_plan_findings, visible_working_findings


@dataclass(frozen=True)
class FatigueCompatibilityService:
    """Expose legacy fatigue call shapes without putting policy in bootstrap."""

    range_findings: Callable[..., dict[date, list[str]]]
    get_shift: Callable[..., Any]
    is_working: Callable[..., bool]
    segments_for_staff: Callable[..., Any]
    fatigue_rule_config: Callable[..., Any]
    configured_findings: Callable[..., Any]
    span: Callable[..., Any]
    is_early_start: Callable[..., Any]
    is_night_duty: Callable[..., Any]
    is_morning_duty: Callable[..., Any]

    def roster_findings(
        self,
        staff: Any,
        day_list: Any,
        code_by_day: dict[date, str],
        unit_id: int | None = None,
    ) -> dict[date, list[str]]:
        return visible_working_findings(
            staff,
            day_list,
            code_by_day,
            int(unit_id or staff.unit_id),
            range_findings=self.range_findings,
            get_shift=self.get_shift,
        )

    def proposed_findings(
        self, staff: Any, day: date, code: str, proposed_codes: dict[date, str]
    ) -> Any:
        return proposed_plan_findings(
            staff,
            day,
            code,
            proposed_codes,
            get_shift=self.get_shift,
            is_working=self.is_working,
            segments_for_staff=self.segments_for_staff,
            fatigue_rule_config=self.fatigue_rule_config,
            configured_findings=self.configured_findings,
            span=self.span,
            is_early_start=self.is_early_start,
            is_night_duty=self.is_night_duty,
            is_morning_duty=self.is_morning_duty,
        )

    def would_trigger(self, staff: Any, day: date, code: str) -> Any:
        return self.proposed_findings(staff, day, code, {})
