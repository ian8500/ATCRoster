"""Pure fatigue-rule configuration and SRATCOH analysis engine."""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, time, timedelta
from typing import Tuple

SYSTEM_FATIGUE_RULES = [
    {"code": "D21", "name": "Duty duration and rolling hours", "severity": "critical", "parameters": {
        "max_duty_hours": {"label": "Maximum single duty", "value": 10, "unit": "hours"},
        "max_rolling_hours": {"label": "Maximum rolling duty", "value": 200, "unit": "hours"},
        "rolling_days": {"label": "Rolling period", "value": 30, "unit": "days"},
    }},
    {"code": "D22", "name": "Minimum rest between duties", "severity": "critical", "parameters": {
        "normal_rest_hours": {"label": "Normal minimum rest", "value": 12, "unit": "hours"},
        "absolute_min_rest_hours": {"label": "Absolute minimum rest", "value": 11, "unit": "hours"},
        "reduced_rest_window_days": {"label": "Reduced-rest review period", "value": 30, "unit": "days"},
    }},
    {"code": "D23", "name": "Recovery after consecutive duties", "severity": "warning", "parameters": {
        "max_consecutive_duties": {"label": "Consecutive-duty trigger", "value": 6, "unit": "duties"},
        "max_consecutive_hours": {"label": "Consecutive-hours trigger", "value": 50, "unit": "hours"},
        "recovery_hours": {"label": "Required recovery", "value": 60, "unit": "hours"},
        "hard_recovery_hours": {"label": "Warning threshold", "value": 54, "unit": "hours"},
    }},
    {"code": "D24", "name": "Qualifying rest in rolling period", "severity": "warning", "parameters": {
        "qualifying_rest_hours": {"label": "Rest needed to qualify", "value": 54, "unit": "hours"},
        "required_rest_hours": {"label": "Total qualifying rest required", "value": 180, "unit": "hours"},
        "rest_window_days": {"label": "Review period", "value": 30, "unit": "days"},
    }},
    {"code": "D30", "name": "Night-duty limits", "severity": "critical", "parameters": {
        "max_night_hours": {"label": "Maximum night duty", "value": 9.5, "unit": "hours"},
        "max_consecutive_nights": {"label": "Maximum consecutive nights", "value": 2, "unit": "nights"},
    }},
    {"code": "D31", "name": "Recovery after night duties", "severity": "warning", "parameters": {
        "single_night_recovery_hours": {"label": "Recovery after one night", "value": 48, "unit": "hours"},
        "night_block_recovery_hours": {"label": "Recovery after two nights", "value": 54, "unit": "hours"},
    }},
    {"code": "D39", "name": "Early-start frequency", "severity": "warning", "parameters": {
        "max_early_starts": {"label": "Maximum early starts", "value": 2, "unit": "starts"},
        "early_window_hours": {"label": "Review period", "value": 144, "unit": "hours"},
    }},
    {"code": "D40", "name": "Early-start duty length", "severity": "warning", "parameters": {
        "max_early_duty_hours": {"label": "Maximum early-start duty", "value": 8, "unit": "hours"},
    }},
    {"code": "D43", "name": "Morning-duty limits", "severity": "warning", "parameters": {
        "max_morning_points": {"label": "Maximum consecutive morning points", "value": 5, "unit": "points"},
        "max_morning_duty_hours": {"label": "Maximum morning duty", "value": 8.5, "unit": "hours"},
    }},
]

CUSTOM_FATIGUE_RULE_TYPES = {
    "max_duty_hours": {
        "label": "Maximum single duty length",
        "unit": "hours", "default": 10, "uses_window": False,
    },
    "min_rest_hours": {
        "label": "Minimum rest between duties",
        "unit": "hours", "default": 11, "uses_window": False,
    },
    "max_consecutive_duties": {
        "label": "Maximum consecutive duties",
        "unit": "duties", "default": 6, "uses_window": False,
    },
    "max_consecutive_nights": {
        "label": "Maximum consecutive night duties",
        "unit": "nights", "default": 2, "uses_window": False,
    },
    "max_early_starts_in_window": {
        "label": "Maximum early starts in a period",
        "unit": "starts", "default": 2, "uses_window": True,
        "default_window": 6,
    },
    "max_hours_in_window": {
        "label": "Maximum duty hours in a period",
        "unit": "hours", "default": 200, "uses_window": True,
        "default_window": 30,
    },
}

def default_fatigue_rule_config() -> dict:
    return {
        "system": {
            item["code"]: {
                **item,
                "parameters": {
                    key: dict(parameter)
                    for key, parameter in item["parameters"].items()
                },
                "enabled": True,
            }
            for item in SYSTEM_FATIGUE_RULES
        },
        "custom": [],
        "definitions": {
            "early_start_before": "06:30",
            "night_period_start": "01:30",
            "night_period_end": "05:30",
        },
    }

def _custom_fatigue_flags(segs: list, rules: list) -> dict:
    flags: dict[date, list[str]] = {}
    ordered = sorted(segs, key=lambda item: item["start"])
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rule_type = rule.get("rule_type")
        code = str(rule.get("code") or "CUSTOM")
        name = str(rule.get("name") or "Custom fatigue rule")
        try:
            threshold = float(rule.get("threshold"))
            window_days = max(1, int(rule.get("window_days") or 1))
        except (TypeError, ValueError):
            continue
        if rule_type == "max_duty_hours":
            for seg in ordered:
                hours = seg["mins"] / 60
                if hours > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {hours:g}h exceeds {threshold:g}h"
                    )
        elif rule_type == "min_rest_hours":
            for previous, current in zip(ordered, ordered[1:]):
                rest = (current["start"] - previous["end"]).total_seconds() / 3600
                if rest < threshold:
                    flags.setdefault(current["day"], []).append(
                        f"{code}: {name} — {rest:g}h is below {threshold:g}h"
                    )
        elif rule_type in {"max_consecutive_duties", "max_consecutive_nights"}:
            streak = 0
            previous_day = None
            for seg in ordered:
                qualifies = (
                    True if rule_type == "max_consecutive_duties"
                    else bool(seg["night"])
                )
                consecutive = (
                    previous_day is not None
                    and (seg["day"] - previous_day).days == 1
                )
                streak = streak + 1 if qualifies and consecutive else (1 if qualifies else 0)
                if streak > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {streak} exceeds {threshold:g}"
                    )
                previous_day = seg["day"] if qualifies else None
        elif rule_type in {"max_hours_in_window", "max_early_starts_in_window"}:
            window = deque()
            running = 0.0
            for seg in ordered:
                value = (
                    seg["mins"] / 60
                    if rule_type == "max_hours_in_window"
                    else (1.0 if seg["early"] else 0.0)
                )
                window.append((seg["start"], value))
                running += value
                while window and (
                    seg["start"] - window[0][0]
                ) > timedelta(days=window_days):
                    running -= window.popleft()[1]
                if running > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {running:g} exceeds "
                        f"{threshold:g} in {window_days} days"
                    )
    return flags


def _span(d: date, sh):
    if not (sh and sh.start_time and sh.end_time):
        return None, None
    start_dt = datetime.combine(d, sh.start_time)
    end_dt = datetime.combine(d, sh.end_time)
    if sh.end_time <= sh.start_time:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _overlap_window(start_dt: datetime, end_dt: datetime, w_start_h: int, w_start_m: int, w_end_h: int, w_end_m: int) -> int:
    base = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    wnd_start = base.replace(hour=w_start_h, minute=w_start_m)
    wnd_end = base.replace(hour=w_end_h, minute=w_end_m)
    if wnd_end <= wnd_start:
        wnd_end += timedelta(days=1)
    total = 0
    for k in (-1, 0, 1):
        a = max(start_dt, wnd_start + timedelta(days=k))
        b = min(end_dt,  wnd_end + timedelta(days=k))
        if b > a:
            total += int((b - a).total_seconds() // 60)
    return total


def _is_working(sh) -> bool:
    return bool(sh and sh.is_working)


def _configured_time(value: str, fallback: time) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except (TypeError, ValueError):
        return fallback


def _is_night_duty(start_dt: datetime, end_dt: datetime, definitions: dict) -> bool:
    start = _configured_time(definitions.get("night_period_start"), time(1, 30))
    end = _configured_time(definitions.get("night_period_end"), time(5, 30))
    return _overlap_window(
        start_dt, end_dt, start.hour, start.minute, end.hour, end.minute
    ) > 0


def _is_early_start(start_dt: datetime, definitions: dict) -> Tuple[bool, bool]:
    hm = start_dt.time()
    threshold = _configured_time(
        definitions.get("early_start_before"), time(6, 30)
    )
    is_early = time(0, 0) <= hm < threshold
    is_pre0600 = is_early and (hm < time(6, 0))
    return is_early, is_pre0600


def _is_morning_duty(start_dt: datetime) -> bool:
    hm = start_dt.time()
    return time(6, 30) <= hm <= time(7, 59)


def _analyze_segments(segs, rule_config=None, observation_start=None):
    segs = sorted(segs, key=lambda x: x["start"])
    flags = {}
    if not segs:
        return flags

    config = rule_config or default_fatigue_rule_config()
    system = config["system"]

    def parameter(code, name):
        return float(system[code]["parameters"][name]["value"])

    d21_window = timedelta(days=parameter("D21", "rolling_days"))
    d22_window = timedelta(days=parameter("D22", "reduced_rest_window_days"))
    d24_window = timedelta(days=parameter("D24", "rest_window_days"))
    early_window_span = timedelta(hours=parameter("D39", "early_window_hours"))

    win30 = deque()
    duty_30 = 0
    reduced_intervals_30 = deque()
    night_block_count = 0
    last_night_end = None

    consec_queue = deque()

    morning_streak_points = 0
    early_window = deque()
    last_duty_day = None
    last_was_night = False
    last_was_early_pre0600 = False

    prev_end = None

    for seg in segs:
        start = seg["start"]
        end = seg["end"]
        mins = seg["mins"]
        night = seg["night"]
        early = seg["early"]
        early_pre0600 = seg["early_pre0600"]
        morning = seg["morning"]
        the_day = seg["day"]

        if night_block_count > 0 and not night and last_night_end is not None:
            gap = start - last_night_end
            req_hours = (
                parameter("D31", "single_night_recovery_hours")
                if night_block_count == 1
                else parameter("D31", "night_block_recovery_hours")
            )
            if gap < timedelta(hours=req_hours):
                flags.setdefault(the_day, []).append(
                    f"<{req_hours}h after {'single' if night_block_count == 1 else 'two consecutive'} night(s) (D31: {int(gap.total_seconds()//3600)}h)"
                )
            night_block_count = 0
            last_night_end = None

        if prev_end is not None:
            gap = start - prev_end
            while reduced_intervals_30 and (
                start - reduced_intervals_30[0]
            ) > d22_window:
                reduced_intervals_30.popleft()
            normal_rest = parameter("D22", "normal_rest_hours")
            absolute_rest = parameter("D22", "absolute_min_rest_hours")
            if gap < timedelta(hours=normal_rest):
                if gap >= timedelta(hours=absolute_rest):
                    if len(reduced_intervals_30) == 0:
                        reduced_intervals_30.append(start)
                    else:
                        flags.setdefault(the_day, []).append(
                            f"<{normal_rest:g}h between duties (D22) and "
                            f"{absolute_rest:g}–{normal_rest:g}h allowance "
                            f"already used within last "
                            f"{parameter('D22', 'reduced_rest_window_days'):g} days"
                        )
                else:
                    flags.setdefault(the_day, []).append(
                        f"<{absolute_rest:g}h between duties "
                        f"(D22: {int(gap.total_seconds()//3600)}h)"
                    )

        qualifying_rest = parameter("D24", "qualifying_rest_hours")
        required_rest = parameter("D24", "required_rest_hours")
        rest_window_start = start - d24_window
        has_complete_rest_window = (
            observation_start is not None
            and observation_start <= rest_window_start
        )
        qual_hours = 0.0
        if has_complete_rest_window:
            rest_start = rest_window_start
            for prior in segs:
                if prior["start"] >= start:
                    break
                if prior["end"] <= rest_window_start:
                    continue
                duty_start = max(prior["start"], rest_window_start)
                if duty_start > rest_start:
                    rest = duty_start - rest_start
                    if rest >= timedelta(hours=qualifying_rest):
                        qual_hours += rest.total_seconds() / 3600.0
                rest_start = max(rest_start, prior["end"])
            if start > rest_start:
                rest = start - rest_start
                if rest >= timedelta(hours=qualifying_rest):
                    qual_hours += rest.total_seconds() / 3600.0
        if has_complete_rest_window and qual_hours < required_rest:
            flags.setdefault(the_day, []).append(
                f"D24: qualifying rest {int(round(qual_hours))}h "
                f"(<{required_rest:g}h) in last "
                f"{parameter('D24', 'rest_window_days'):g}d"
            )

        prior_consec_count = len(consec_queue)
        prior_consec_minutes = sum(m for (_, _, m) in consec_queue)
        max_consecutive = parameter("D23", "max_consecutive_duties")
        max_consecutive_hours = parameter("D23", "max_consecutive_hours")
        if (
            prior_consec_count >= max_consecutive
            or prior_consec_minutes >= max_consecutive_hours * 60
        ):
            if prev_end is not None:
                gap = start - prev_end
                recovery = parameter("D23", "recovery_hours")
                hard_recovery = parameter("D23", "hard_recovery_hours")
                if gap < timedelta(hours=recovery):
                    if gap < timedelta(hours=hard_recovery):
                        flags.setdefault(the_day, []).append(
                            f"<{recovery:g}h after {max_consecutive:g} "
                            f"consecutive duties or ≥{max_consecutive_hours:g}h "
                            f"across consecutive duties "
                            f"(D23: {int(gap.total_seconds()//3600)}h)"
                        )

        max_duty_hours = parameter("D21", "max_duty_hours")
        if mins > max_duty_hours * 60:
            flags.setdefault(the_day, []).append(
                f"Duty > {max_duty_hours:g}h (D21)"
            )

        while win30 and (end - win30[0][1]) > d21_window:
            _, _, mo = win30.popleft()
            duty_30 -= mo
        win30.append((start, end, mins))
        duty_30 += mins
        max_rolling_hours = parameter("D21", "max_rolling_hours")
        if duty_30 > max_rolling_hours * 60:
            flags.setdefault(the_day, []).append(
                f">{max_rolling_hours:g}h duty in last "
                f"{parameter('D21', 'rolling_days'):g} days (D21)")

        if night:
            max_night_hours = parameter("D30", "max_night_hours")
            if mins > max_night_hours * 60:
                flags.setdefault(the_day, []).append(
                    f"Night duty > {max_night_hours:g}h (D30)"
                )
            if end.time() > time(7, 30):
                flags.setdefault(the_day, []).append(
                    "Night duty ends after 07:30 (D30)")

            if last_duty_day and (the_day - last_duty_day).days == 1 and last_was_night:
                night_block_count += 1
            else:
                night_block_count = 1

            max_nights = parameter("D30", "max_consecutive_nights")
            if night_block_count > max_nights:
                flags.setdefault(the_day, []).append(
                    f"More than {max_nights:g} consecutive night duties (D30)"
                )

            last_night_end = end

        if early:
            early_window.append(start)
            while early_window and (
                start - early_window[0]
            ) > early_window_span:
                early_window.popleft()
            max_early_starts = parameter("D39", "max_early_starts")
            if len(early_window) > max_early_starts:
                flags.setdefault(the_day, []).append(
                    f"More than {max_early_starts:g} early starts in "
                    f"{parameter('D39', 'early_window_hours'):g}h (D39)"
                )
            if early_pre0600 and last_was_early_pre0600 and last_duty_day and (the_day - last_duty_day).days == 1:
                flags.setdefault(the_day, []).append(
                    "Consecutive early starts both before 06:00 not permitted (D39)"
                )
            max_early_hours = parameter("D40", "max_early_duty_hours")
            if mins > max_early_hours * 60:
                flags.setdefault(the_day, []).append(
                    f"Early start duty > {max_early_hours:g}h (D40)"
                )

        if early or morning:
            points_today = 2 if early_pre0600 else 1
            if last_duty_day and (the_day - last_duty_day).days == 1 and (morning_streak_points > 0):
                morning_streak_points += points_today
            else:
                morning_streak_points = points_today
            max_morning_points = parameter("D43", "max_morning_points")
            if morning_streak_points > max_morning_points:
                flags.setdefault(the_day, []).append(
                    f"More than {max_morning_points:g} consecutive "
                    f"morning-duty points (D43)"
                )
        else:
            morning_streak_points = 0

        max_morning_hours = parameter("D43", "max_morning_duty_hours")
        if morning and mins > max_morning_hours * 60:
            flags.setdefault(the_day, []).append(
                f"Morning duty > {max_morning_hours:g}h (D43)"
            )

        if (last_duty_day is None) or ((the_day - last_duty_day).days >= 2):
            consec_queue.clear()
        consec_queue.append((start, end, mins))

        prev_end = end
        last_duty_day = the_day
        last_was_night = night
        last_was_early_pre0600 = early_pre0600

    return flags

