"""Airport-owned roster settings persistence and normalization."""

from __future__ import annotations

import json
import re
from functools import lru_cache
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


def parse_codes_input(raw: str) -> list[str]:
    return normalise_codes(re.split(r"[\s,]+", raw or ""))


def save_code_setting(
    key: str,
    values: list[str],
    *,
    unit_id: int,
    db: Any,
    RosterSetting: Any,
    refresh_cache: Callable[[], None],
) -> None:
    save_setting(
        key,
        json.dumps(normalise_codes(values)),
        unit_id=unit_id,
        db=db,
        RosterSetting=RosterSetting,
        refresh_cache=refresh_cache,
    )
    db.session.commit()


def save_absence_catalogue(
    items: list[dict[str, object]],
    *,
    unit_id: int,
    db: Any,
    RosterSetting: Any,
    refresh_cache: Callable[[], None],
) -> None:
    save_setting(
        "absence_types",
        json.dumps(items, separators=(",", ":")),
        unit_id=unit_id,
        db=db,
        RosterSetting=RosterSetting,
        refresh_cache=refresh_cache,
    )
    db.session.commit()


class RosterSettingsCatalogue:
    """Tenant-scoped cached roster settings and code catalogue."""

    def __init__(
        self,
        *,
        db: Any,
        RosterSetting: Any,
        ShiftType: Any,
        current_unit_id: Callable[[], int],
        defaults: Mapping[str, Any],
        absence_defaults: list[dict[str, object]],
        working_codes: list[str],
        banned_codes: list[str],
        excluded_codes: list[str],
        non_working_codes: list[str],
    ):
        self.db = db
        self.RosterSetting = RosterSetting
        self.ShiftType = ShiftType
        self.current_unit_id = current_unit_id
        self.defaults = defaults
        self.absence_defaults = absence_defaults
        self.working_codes = working_codes
        self.banned_codes = banned_codes
        self.excluded_codes = excluded_codes
        self.non_working_codes = non_working_codes
        self._secondary_cache_clear: Callable[[], None] | None = None

    def _unit_id(self, unit_id: int | None = None) -> int:
        return int(unit_id or self.current_unit_id() or 1)

    @lru_cache(maxsize=128)
    def snapshot(self, unit_id: int) -> dict[str, str]:
        rows = self.RosterSetting.query.filter_by(unit_id=unit_id).all()
        return {row.key: row.value for row in rows}

    def set_secondary_cache_clear(self, clear: Callable[[], None]) -> None:
        self._secondary_cache_clear = clear

    def refresh(self) -> None:
        self.snapshot.cache_clear()
        if self._secondary_cache_clear is not None:
            self._secondary_cache_clear()

    @staticmethod
    def normalise(values: Iterable[str]) -> list[str]:
        return normalise_codes(values)

    def load_codes(
        self, key: str, default: list[str], unit_id: int | None = None
    ) -> set[str]:
        resolved_unit_id = self._unit_id(unit_id)
        return load_code_setting(
            key,
            default,
            resolved_unit_id,
            settings_snapshot=self.snapshot,
            db=self.db,
            ShiftType=self.ShiftType,
        )

    def get_working_codes(self) -> set[str]:
        return self.load_codes("working_codes", self.working_codes)

    def get_banned_codes(self) -> set[str]:
        return self.load_codes("banned_codes", self.banned_codes)

    def get_excluded_counter_codes(self) -> set[str]:
        return self.load_codes("exclude_from_counters", self.excluded_codes)

    def get_non_working_codes(self) -> set[str]:
        return self.load_codes("non_working_codes", self.non_working_codes)

    def get_absence_types(
        self,
        category: str | None = None,
        active_only: bool = True,
        unit_id: int | None = None,
    ) -> list[dict[str, object]]:
        return load_absence_types(
            self.snapshot(self._unit_id(unit_id)).get("absence_types"),
            self.absence_defaults,
            category=category,
            active_only=active_only,
        )

    def save_absence_types(self, items: list[dict[str, object]]) -> None:
        return save_absence_catalogue(
            items,
            unit_id=self._unit_id(),
            db=self.db,
            RosterSetting=self.RosterSetting,
            refresh_cache=self.refresh,
        )

    def get_shift_counter_map(self, unit_id: int | None = None) -> dict[str, str]:
        return decode_counter_map(
            self.snapshot(self._unit_id(unit_id)).get("shift_counter_map", "{}")
        )

    @staticmethod
    def parse_codes_input(raw: str) -> list[str]:
        return parse_codes_input(raw)

    def save_codes_setting(self, key: str, values: list[str]) -> None:
        return save_code_setting(
            key,
            values,
            unit_id=self._unit_id(),
            db=self.db,
            RosterSetting=self.RosterSetting,
            refresh_cache=self.refresh,
        )

    def prune_code_settings(self, unit_id: int) -> int:
        return prune_code_settings(
            unit_id,
            db=self.db,
            ShiftType=self.ShiftType,
            RosterSetting=self.RosterSetting,
            setting_keys=self.defaults,
            refresh_cache=self.refresh,
        )

    def save_setting(self, key: str, value: str) -> None:
        return save_setting(
            key,
            value,
            unit_id=self._unit_id(),
            db=self.db,
            RosterSetting=self.RosterSetting,
            refresh_cache=self.refresh,
        )
