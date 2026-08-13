"""Roster annotation parsing against the unit-owned annotation catalogue."""

from __future__ import annotations

from typing import Any, Callable


def parse_annotation(
    value: str,
    *,
    get_annotation_config: Callable[[str], Any],
    annotation_snapshot: Callable[[int], dict[str, Any]],
    current_unit_id: Callable[[], int],
) -> dict[str, str | None] | None:
    """Parse a configured base code or a configured single-character suffix."""
    if not value:
        return None
    value = value.strip().upper()
    if info := get_annotation_config(value):
        return {"type": info["code"], "suffix": None}
    for item in annotation_snapshot(int(current_unit_id() or 1))["items"]:
        if not item["allow_suffix"] or not (code := item["code"]):
            continue
        if value.startswith(code) and len(value) == len(code) + 1:
            suffix = value[len(code):]
            if suffix in set(item["suffixes"]):
                return {"type": code, "suffix": suffix}
    return None
