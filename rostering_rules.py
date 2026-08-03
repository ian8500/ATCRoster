"""Framework-neutral roster pattern and staff-rule policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class PatternResolution:
    assignment_id: int
    pattern_id: int
    cycle_index: int
    day_type: str
    fixed_shift_type_id: int | None
    allowed_shift_type_ids: frozenset[int]
    required_work: bool
    contracted_minutes: int


@dataclass(frozen=True)
class EffectiveRule:
    rule_id: int
    rule_type: str
    hardness: str
    shift_type_id: int | None
    shift_group: str | None
    penalty_weight: int
    reason: str


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason_code: str = "ELIGIBLE"
    explanation: str = "Employee is eligible for this duty."
    soft_penalty: int = 0


def cycle_index(anchor_date: date, anchor_day_index: int, day: date, length: int) -> int:
    if length <= 0:
        raise ValueError("Pattern cycle length must be positive.")
    return (anchor_day_index + (day - anchor_date).days) % length


def evaluate_eligibility(
    pattern: PatternResolution | None,
    rules: Iterable[EffectiveRule],
    *,
    shift_type_id: int,
    shift_code: str,
    shift_group: str | None = None,
) -> EligibilityResult:
    code = (shift_code or "").upper()
    group = (shift_group or "").upper()
    if pattern:
        if pattern.day_type in {"OFF", "PROTECTED_NON_OPERATIONAL"}:
            return EligibilityResult(False, "PATTERN_OFF_DAY", "The working pattern does not permit an operational duty on this day.")
        if pattern.day_type == "FIXED_SHIFT" and pattern.fixed_shift_type_id != shift_type_id:
            return EligibilityResult(False, "PATTERN_FIXED_SHIFT", "The working pattern requires a different fixed shift.")
        if pattern.day_type == "WORK_ALLOWED_SET" and shift_type_id not in pattern.allowed_shift_type_ids:
            return EligibilityResult(False, "PATTERN_SHIFT_NOT_ALLOWED", "This shift is not in the pattern day's allowed set.")
    soft_penalty = 0
    for rule in rules:
        applies = (
            (rule.rule_type in {"NO_NIGHT", "AVOID_NIGHT"} and code == "N")
            or (rule.rule_type in {"NO_EARLY", "AVOID_EARLY"} and group == "EARLY")
            or (rule.rule_type == "DISALLOWED_SHIFT" and rule.shift_type_id == shift_type_id)
            or (rule.rule_type == "ALLOWED_SHIFT" and rule.shift_type_id != shift_type_id)
        )
        if not applies:
            continue
        if rule.hardness == "HARD":
            return EligibilityResult(False, f"{rule.rule_type}_RULE", rule.reason or "An active hard staff rule prevents this duty.")
        soft_penalty += max(0, rule.penalty_weight)
    return EligibilityResult(True, soft_penalty=soft_penalty)
