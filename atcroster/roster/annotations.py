"""Roster annotation parsing against the unit-owned annotation catalogue."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable


def build_annotation_snapshot(AnnotationType: Any, unit_id: int) -> dict[str, Any]:
    """Build the normalized annotation catalogue for one airport."""
    rows = (
        AnnotationType.query.filter(AnnotationType.unit_id == unit_id)
        .order_by(AnnotationType.code)
        .all()
    )
    items = []
    for row in rows:
        tags = tuple(
            sorted(
                tag.strip().lower()
                for tag in (row.tags or "").split(",")
                if tag.strip()
            )
        )
        suffixes = "".join(sorted(set((row.suffixes or "").upper())))
        items.append(
            {
                "id": row.id,
                "code": (row.code or "").upper(),
                "label": row.label or row.code.upper(),
                "category": row.category or "Other",
                "colour": row.colour or "#6c757d",
                "description": row.description or "",
                "allow_suffix": bool(row.allow_suffix),
                "suffixes": suffixes,
                "toil_half_days": int(row.toil_half_days or 0),
                "tags": tags,
                "note_required": bool(row.note_required),
                "admin_only": bool(row.admin_only),
                "is_active": bool(row.is_active),
                "sort_order": row.sort_order if row.sort_order is not None else 0,
            }
        )
    return {"items": items, "by_code": {item["code"]: item for item in items}}


def annotation_types(
    snapshot: dict[str, Any], active_only: bool = True
) -> list[dict[str, Any]]:
    items = snapshot["items"]
    return [item for item in items if item["is_active"]] if active_only else items


def annotation_config(
    snapshot: dict[str, Any], code: str | None
) -> dict[str, Any] | None:
    if not code:
        return None
    return snapshot["by_code"].get(code.strip().upper())


def annotation_groups(
    items: list[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in items:
        groups.setdefault(item["category"], []).append(item)
    return groups


def tags_for(config: dict[str, Any] | None) -> set[str]:
    return set(config.get("tags") or ()) if config else set()


def codes_for_tag(items: list[dict[str, Any]], tag: str) -> list[str]:
    needle = (tag or "").lower().strip()
    if not needle:
        return []
    return [item["code"] for item in items if needle in set(item.get("tags") or ())]


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
            suffix = value[len(code) :]
            if suffix in set(item["suffixes"]):
                return {"type": code, "suffix": suffix}
    return None
