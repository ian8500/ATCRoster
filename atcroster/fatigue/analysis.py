"""Core fatigue segment construction and configured-rule evaluation."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any, Callable


def segments_from_assignments(
    staff: Any,
    assignments: Any,
    definitions: dict[str, Any],
    *,
    get_shift: Callable[..., Any],
    is_working: Callable[[Any], bool],
    span: Callable[..., tuple[Any, Any]],
    is_night_duty: Callable[..., bool],
    is_early_start: Callable[..., tuple[bool, bool]],
    is_morning_duty: Callable[[Any], bool],
) -> list[dict[str, Any]]:
    """Convert effective working assignments into fatigue-engine segments."""
    segments = []
    for assignment in assignments:
        code = (assignment.effective_code or "").upper()
        if code in ("SC", "SSC"):
            continue
        shift = get_shift(code, staff.unit_id) if code else None
        if not is_working(shift):
            continue
        start, end = span(assignment.day, shift)
        if not start:
            continue
        early, pre0600 = is_early_start(start, definitions)
        segments.append(
            {
                "day": assignment.day,
                "start": start,
                "end": end,
                "mins": int((end - start).total_seconds() // 60),
                "night": is_night_duty(start, end, definitions),
                "early": early,
                "early_pre0600": pre0600,
                "morning": is_morning_duty(start),
            }
        )
    return segments


def configured_findings(
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    observation_start: Any,
    *,
    analyze_segments: Callable[..., dict[Any, list[str]]],
    custom_fatigue_flags: Callable[..., dict[Any, list[str]]],
) -> dict[Any, list[str]]:
    """Run enabled system rules and airport-local rules as one policy set."""
    findings = analyze_segments(segments, config, observation_start=observation_start)
    enabled_system = {
        code for code, rule in config["system"].items() if rule["enabled"]
    }
    filtered = {
        finding_day: [
            message
            for message in messages
            if not (match := re.search(r"\b(D\d{2})\b", message))
            or match.group(1) in enabled_system
        ]
        for finding_day, messages in findings.items()
    }
    for finding_day, messages in custom_fatigue_flags(
        segments, config["custom"]
    ).items():
        filtered.setdefault(finding_day, []).extend(messages)
    return {
        finding_day: messages for finding_day, messages in filtered.items() if messages
    }


def new_findings_for_proposed_assignment(
    staff: Any,
    proposed_day: Any,
    proposed_code: str,
    *,
    lookback_days: int,
    lookahead_days: int,
    get_shift: Callable[..., Any],
    is_working: Callable[[Any], bool],
    segments_for_staff: Callable[[Any, Any, Any], list[dict[str, Any]]],
    fatigue_rule_config: Callable[[int], dict[str, Any]],
    configured_fatigue_findings: Callable[..., dict[Any, list[str]]],
    span: Callable[..., tuple[Any, Any]],
    is_early_start: Callable[..., tuple[bool, bool]],
    is_night_duty: Callable[..., bool],
    is_morning_duty: Callable[[Any], bool],
) -> dict[Any, list[str]]:
    """Return only fatigue findings introduced by a proposed duty."""
    shift = get_shift(proposed_code, staff.unit_id)
    if not is_working(shift):
        return {}
    start_day = proposed_day - timedelta(days=lookback_days)
    end_day = proposed_day + timedelta(days=lookahead_days)
    baseline_segments = segments_for_staff(staff, start_day, end_day)
    observation_start = datetime.combine(start_day, time.min)
    config = fatigue_rule_config(staff.unit_id)
    baseline = configured_fatigue_findings(baseline_segments, config, observation_start)
    start, end = span(proposed_day, shift)
    if not start:
        return {}
    definitions = config["definitions"]
    early, pre0600 = is_early_start(start, definitions)
    proposed_segments = [
        segment for segment in baseline_segments if segment["day"] != proposed_day
    ]
    proposed_segments.append(
        {
            "day": proposed_day,
            "start": start,
            "end": end,
            "mins": int((end - start).total_seconds() // 60),
            "night": is_night_duty(start, end, definitions),
            "early": early,
            "early_pre0600": pre0600,
            "morning": is_morning_duty(start),
        }
    )
    proposed = configured_fatigue_findings(proposed_segments, config, observation_start)
    result = {}
    for finding_day, findings in proposed.items():
        if finding_day < proposed_day:
            continue
        new_findings = sorted(set(findings) - set(baseline.get(finding_day, [])))
        if new_findings:
            result[finding_day] = new_findings
    return result
