"""Administration service for configurable absence types."""

from __future__ import annotations

import re
from typing import Callable


def update_absence_types(
    form: str,
    values: dict[str, str],
    *,
    load: Callable[..., list[dict]],
    save: Callable[[list[dict]], None],
) -> tuple[str, str]:
    """Apply a create/deactivate action and return a flash message/category."""
    types = load(active_only=False)
    code = (values.get("code") or "").strip().upper()
    if form == "absence_type_add":
        label = (values.get("label") or "").strip()
        category = (values.get("category") or "").strip().lower()
        if not re.fullmatch(r"[A-Z0-9]{1,10}", code) or category not in {"leave", "sickness"} or not label:
            return "Enter a name, category and a 1–10 character code.", "error"
        existing = next((item for item in types if item["code"] == code), None)
        if existing:
            existing.update(label=label[:80], category=category, active=True)
        else:
            types.append({"code": code, "label": label[:80], "category": category, "active": True})
        save(types)
        return f"{label} is now available for this airport.", "ok"
    item = next((item for item in types if item["code"] == code), None)
    if not item:
        return "That absence type does not exist.", "error"
    item["active"] = False
    save(types)
    return f"{item['label']} was removed from new records and reports. Historical records were retained.", "ok"
