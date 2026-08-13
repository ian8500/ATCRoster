"""Airport-owned roster settings persistence and normalization."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Mapping


def normalise_codes(values: Iterable[str]) -> list[str]:
    result = []
    for value in values:
        code = (value or "").strip().upper()
        if code and code not in result:
            result.append(code)
    return result


def load_code_setting(
    key: str,
    default: list[str],
    unit_id: int,
    *,
    settings_snapshot: Callable[[int], Mapping[str, str]],
    db: Any,
    ShiftType: Any,
) -> set[str]:
    raw = settings_snapshot(unit_id).get(key)
    try:
        parsed = json.loads(raw) if raw else default
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = default
    configured = set(normalise_codes(parsed if isinstance(parsed, list) else default))
    existing = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(unit_id=unit_id).all()
    }
    return configured & existing


def load_absence_types(
    raw: str | None,
    defaults: list[dict[str, object]],
    *,
    category: str | None = None,
    active_only: bool = True,
) -> list[dict[str, object]]:
    try:
        parsed = json.loads(raw) if raw else defaults
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = defaults
    if not isinstance(parsed, list):
        parsed = defaults
    result = []
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().upper()
        item_category = str(item.get("category") or "").strip().lower()
        if (
            not re.fullmatch(r"[A-Z0-9]{1,10}", code)
            or item_category not in {"leave", "sickness"}
            or code in seen
        ):
            continue
        seen.add(code)
        normalized = {
            "code": code,
            "label": str(item.get("label") or code).strip()[:80] or code,
            "category": item_category,
            "active": bool(item.get("active", True)),
        }
        if category and item_category != category:
            continue
        if active_only and not normalized["active"]:
            continue
        result.append(normalized)
    return result


def save_setting(
    key: str,
    value: str,
    *,
    unit_id: int,
    db: Any,
    RosterSetting: Any,
    refresh_cache: Callable[[], None],
) -> None:
    row = RosterSetting.query.filter_by(unit_id=unit_id, key=key).first()
    if row is None:
        row = RosterSetting(unit_id=unit_id, key=key, value=value)
        db.session.add(row)
    else:
        row.value = value
    refresh_cache()


def prune_code_settings(
    unit_id: int,
    *,
    db: Any,
    ShiftType: Any,
    RosterSetting: Any,
    setting_keys: Iterable[str],
    refresh_cache: Callable[[], None],
) -> int:
    valid_codes = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(unit_id=unit_id).all()
    }
    changed = 0
    rows = RosterSetting.query.filter(
        RosterSetting.unit_id == unit_id,
        RosterSetting.key.in_(setting_keys),
    ).all()
    for row in rows:
        try:
            values = json.loads(row.value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        normalized = normalise_codes(values if isinstance(values, list) else [])
        cleaned = [code for code in normalized if code in valid_codes]
        if cleaned != normalized:
            row.value = json.dumps(cleaned)
            changed += 1
    if changed:
        refresh_cache()
    return changed


def decode_counter_map(raw: str | None) -> dict[str, str]:
    try:
        values = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        values = {}
    if not isinstance(values, dict):
        return {}
    return {
        str(code).upper(): str(group).upper()
        for code, group in values.items()
        if str(group).upper() in {"", "M", "D", "A", "N"}
    }
