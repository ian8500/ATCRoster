"""Central role and unit-permission policy for ATCRoster."""

from __future__ import annotations

import json


def is_admin(user) -> bool:
    return bool(
        getattr(user, "is_admin", False) or getattr(user, "role", "") == "admin"
    )


def is_editor(user) -> bool:
    return getattr(user, "role", "") in ("editor", "admin")


def permissions_for(user) -> dict[str, bool]:
    try:
        raw = json.loads(getattr(user, "permissions_json", "") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items() if isinstance(key, str)}


def has_permission(user, permission: str) -> bool:
    return bool(permissions_for(user).get(permission, False))


def is_trainee(user) -> bool:
    return bool(
        getattr(user, "is_trainee", False)
        or getattr(user, "tower_ut", False)
        or getattr(user, "radar_ut", False)
        or getattr(user, "met_ut", False)
    )


def may_record_training(user) -> bool:
    return bool(
        is_admin(user)
        or getattr(user, "has_ojti", False)
        or getattr(user, "has_assessor", False)
    )


def may_manage_training(user) -> bool:
    return bool(
        is_admin(user)
        or getattr(user, "has_assessor", False)
        or getattr(user, "is_wm", False)
        or getattr(user, "is_dwm", False)
    )


def may_edit_roster(user) -> bool:
    return bool(
        is_admin(user)
        or is_editor(user)
        or (
            (getattr(user, "is_wm", False) or getattr(user, "is_dwm", False))
            and has_permission(user, "edit_roster")
        )
    )


def may_apply_annotations(user) -> bool:
    return bool(
        is_admin(user) or is_editor(user) or has_permission(user, "apply_annotations")
    )


def may_send_unit_messages(user) -> bool:
    return bool(
        is_admin(user)
        or getattr(user, "is_wm", False)
        or getattr(user, "is_dwm", False)
    )


def may_override_roster_conflicts(user) -> bool:
    return is_admin(user) or has_permission(user, "override_roster_conflicts")
