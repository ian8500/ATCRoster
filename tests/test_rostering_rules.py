from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rostering_rules import (  # noqa: E402
    EffectiveRule, PatternResolution, cycle_index, evaluate_eligibility,
)


def test_cycle_index_supports_historical_and_future_dates():
    anchor = date(2026, 1, 10)
    assert cycle_index(anchor, 2, anchor, 10) == 2
    assert cycle_index(anchor, 2, date(2026, 1, 18), 10) == 0
    assert cycle_index(anchor, 2, date(2026, 1, 8), 10) == 0


def test_hard_no_night_rule_blocks_night_and_explains_why():
    result = evaluate_eligibility(
        None,
        [EffectiveRule(1, "NO_NIGHT", "HARD", None, None, 0, "Medical restriction")],
        shift_type_id=4, shift_code="N",
    )
    assert not result.eligible
    assert result.reason_code == "NO_NIGHT_RULE"
    assert result.explanation == "Medical restriction"


def test_soft_rule_adds_penalty_without_making_staff_ineligible():
    result = evaluate_eligibility(
        None,
        [EffectiveRule(1, "AVOID_NIGHT", "SOFT", None, None, 7, "Preference")],
        shift_type_id=4, shift_code="N",
    )
    assert result.eligible
    assert result.soft_penalty == 7


def test_allowed_set_and_off_days_are_hard_pattern_constraints():
    allowed = PatternResolution(1, 1, 0, "WORK_ALLOWED_SET", None, frozenset({2, 3}), True, 2400)
    assert evaluate_eligibility(allowed, [], shift_type_id=2, shift_code="M").eligible
    assert not evaluate_eligibility(allowed, [], shift_type_id=4, shift_code="N").eligible
    off = PatternResolution(1, 1, 4, "OFF", None, frozenset(), False, 2400)
    assert not evaluate_eligibility(off, [], shift_type_id=2, shift_code="M").eligible
