"""Core fatigue segment construction and configured-rule evaluation."""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from typing import Any, Callable
from collections import defaultdict


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


def roster_findings_matrix(
    staff: list[Any],
    days: list[Any],
    codes_by_staff: dict[int, dict[Any, str]],
    unit_id: int,
    *,
    Assignment: Any,
    segments_from_assignments: Callable[..., list[dict[str, Any]]],
    fatigue_rule_config: Callable[[int], dict[str, Any]],
    configured_findings: Callable[..., dict[Any, list[str]]],
    get_shift: Callable[..., Any],
) -> dict[int, dict[Any, list[str]]]:
    """Calculate displayed fatigue findings using one assignment query."""
    if not staff or not days:
        return {}
    ordered_days = sorted(days)
    start = ordered_days[0] - timedelta(days=30)
    end = ordered_days[-1]
    staff_ids = [person.id for person in staff if person.id is not None]
    assignments = (
        Assignment.query.filter(
            Assignment.unit_id == unit_id,
            Assignment.staff_id.in_(staff_ids or [0]),
            Assignment.day >= start,
            Assignment.day <= end,
        )
        .order_by(Assignment.staff_id, Assignment.day)
        .all()
    )
    assignments_by_staff = defaultdict(list)
    for assignment in assignments:
        assignments_by_staff[assignment.staff_id].append(assignment)
    config = fatigue_rule_config(unit_id)
    target_days = set(ordered_days)
    result = {}
    for person in staff:
        segments = segments_from_assignments(
            person,
            assignments_by_staff.get(person.id, ()),
            config["definitions"],
        )
        findings = configured_findings(
            segments, config, datetime.combine(start, time.min)
        )
        visible = {}
        for finding_day, messages in findings.items():
            shift = get_shift(
                codes_by_staff.get(person.id, {}).get(finding_day), unit_id
            )
            if (
                finding_day in target_days
                and messages
                and shift
                and shift.is_active
                and shift.is_working
            ):
                visible[finding_day] = messages
        result[person.id] = visible
    return result


def findings_for_range(
    staff,
    days,
    *,
    lookback_days,
    segments_for_staff,
    fatigue_rule_config,
    configured_findings,
):
    if not days:
        return {}
    ordered = sorted(days)
    start = ordered[0] - timedelta(days=lookback_days)
    findings = configured_findings(
        segments_for_staff(staff, start, ordered[-1]),
        fatigue_rule_config(staff.unit_id),
        datetime.combine(start, time.min),
    )
    target = set(ordered)
    return {
        day: messages
        for day, messages in findings.items()
        if day in target and messages
    }


def visible_working_findings(
    staff, days, codes_by_day, unit_id, *, range_findings, get_shift
):
    findings = range_findings(staff, days)
    return {
        day: messages
        for day, messages in findings.items()
        if (shift := get_shift(codes_by_day.get(day), unit_id or staff.unit_id))
        and shift.is_active
        and shift.is_working
    }


def proposed_plan_findings(
    staff,
    day,
    code,
    proposed_codes,
    *,
    get_shift,
    is_working,
    segments_for_staff,
    fatigue_rule_config,
    configured_findings,
    span,
    is_early_start,
    is_night_duty,
    is_morning_duty,
):
    shift = get_shift(code, staff.unit_id)
    if not is_working(shift):
        return []
    start_day = min([day, *proposed_codes], default=day) - timedelta(days=30)
    end_day = max([day, *proposed_codes], default=day)
    proposed_days = set(proposed_codes) | {day}
    segments = [
        segment
        for segment in segments_for_staff(staff, start_day, end_day)
        if segment["day"] not in proposed_days
    ]
    config = fatigue_rule_config(staff.unit_id)
    definitions = config["definitions"]
    for proposed_day, proposed_code in {**proposed_codes, day: code}.items():
        proposed_shift = get_shift(proposed_code, staff.unit_id)
        start, end = span(proposed_day, proposed_shift)
        if not start:
            continue
        early, pre0600 = is_early_start(start, definitions)
        segments.append(
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
    segments.sort(key=lambda item: item["start"])
    return configured_findings(
        segments, config, datetime.combine(start_day, time.min)
    ).get(day, [])


def segments_for_staff(
    staff,
    start_day,
    end_day,
    *,
    Assignment,
    fatigue_rule_config,
    build_segments,
):
    """Load one staff member's dated assignments and construct duty segments."""
    assignments = (
        Assignment.query.filter(
            Assignment.staff_id == staff.id,
            Assignment.day >= start_day,
            Assignment.day <= end_day,
        )
        .order_by(Assignment.day.asc())
        .all()
    )
    definitions = fatigue_rule_config(staff.unit_id)["definitions"]
    return build_segments(staff, assignments, definitions)
