"""Core fatigue segment construction and configured-rule evaluation."""

from __future__ import annotations

import re
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
