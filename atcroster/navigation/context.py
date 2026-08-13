"""Authenticated navigation and template-shell context construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from flask import Flask, request
from flask_login import current_user


@dataclass(frozen=True)
class NavigationContextDependencies:
    db: Any
    Unit: Any
    Staff: Any
    ShiftRequest: Any
    FeatureFlag: Any
    Notification: Any
    BriefingDelivery: Any
    BriefingItem: Any
    is_admin_user: Callable[[Any], bool]
    is_editor_user: Callable[[Any], bool]
    briefing_enabled: Callable[[int], bool]
    training_enabled: Callable[[int], bool]
    competency_enabled: Callable[[int], bool]
    live_position_enabled: Callable[[int], bool]
    briefing_local_now: Callable[[int], Any]


def create_navigation_context_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any,
    briefing_models: Any, **services: Any
) -> NavigationContextDependencies:
    """Bind template navigation records at the navigation boundary."""
    return NavigationContextDependencies(
        db=db,
        Unit=operational_models.Unit,
        Staff=operational_models.Staff,
        ShiftRequest=operational_models.ShiftRequest,
        FeatureFlag=saas_models.FeatureFlag,
        Notification=operational_models.Notification,
        BriefingDelivery=briefing_models.BriefingDelivery,
        BriefingItem=briefing_models.BriefingItem,
        **services,
    )


def build_navigation_context(
    user: Any,
    endpoint: str | None,
    dependencies: NavigationContextDependencies,
) -> dict[str, Any]:
    """Build the permission, module, branding, and unread navigation state."""
    deps = dependencies
    authenticated_user = user if getattr(user, "is_authenticated", False) else None
    current_unit = (
        deps.db.session.get(
            deps.Unit, int(getattr(authenticated_user, "unit_id", 0) or 0)
        )
        if authenticated_user
        and getattr(authenticated_user, "role", "") != "superadmin"
        else None
    )
    branding = _branding_for(current_unit)
    pending_request_count = 0
    unread_notification_count = 0
    unread_briefing_count = 0
    active_admin_count = 0
    enabled_feature_keys: set[str] = set()
    has_briefing_module = False
    has_training_module = False
    has_competency_module = False
    has_live_position_module = False

    if current_unit and authenticated_user and deps.is_admin_user(authenticated_user):
        active_admin_count = deps.Staff.query.filter(
            deps.Staff.unit_id == current_unit.id,
            deps.Staff.membership_status == "active",
            deps.db.or_(deps.Staff.role == "admin", deps.Staff.is_admin.is_(True)),
        ).count()
        pending_request_count = deps.ShiftRequest.query.filter(
            deps.ShiftRequest.unit_id == current_unit.id,
            deps.ShiftRequest.status.in_(("pending", "approved")),
        ).count()

    if current_unit and authenticated_user:
        enabled_feature_keys = {
            row.key
            for row in deps.FeatureFlag.query.filter_by(
                unit_id=current_unit.id, enabled=True
            ).all()
        }
        unread_notification_count = deps.Notification.query.filter_by(
            unit_id=current_unit.id,
            recipient_id=authenticated_user.id,
            read_at=None,
        ).count()
        has_briefing_module = deps.briefing_enabled(current_unit.id)
        has_training_module = deps.training_enabled(current_unit.id)
        has_competency_module = deps.competency_enabled(current_unit.id)
        has_live_position_module = deps.live_position_enabled(current_unit.id)
        if has_briefing_module and _shows_briefing_count(endpoint):
            briefing_now = deps.briefing_local_now(current_unit.id)
            unread_briefing_count = (
                deps.db.session.query(deps.BriefingDelivery.id)
                .join(
                    deps.BriefingItem,
                    deps.BriefingItem.id == deps.BriefingDelivery.briefing_id,
                )
                .filter(
                    deps.BriefingDelivery.unit_id == current_unit.id,
                    deps.BriefingDelivery.recipient_id == authenticated_user.id,
                    deps.BriefingDelivery.acknowledged_at.is_(None),
                    deps.BriefingDelivery.archived_at.is_(None),
                    deps.BriefingDelivery.deleted_at.is_(None),
                    deps.BriefingItem.status == "published",
                    deps.BriefingItem.kind != "daily",
                    deps.BriefingItem.effective_at <= briefing_now,
                    deps.BriefingItem.expires_at >= briefing_now,
                )
                .count()
            )

    return {
        "is_admin": bool(authenticated_user) and deps.is_admin_user(authenticated_user),
        "is_editor": bool(authenticated_user)
        and deps.is_editor_user(authenticated_user),
        "pending_request_count": pending_request_count,
        "unread_notification_count": unread_notification_count,
        "unread_briefing_count": unread_briefing_count,
        "has_briefing_module": has_briefing_module,
        "has_training_module": has_training_module,
        "has_competency_module": has_competency_module,
        "has_live_position_module": has_live_position_module,
        "has_handover_module": "handover_module" in enabled_feature_keys,
        "enabled_feature_keys": enabled_feature_keys,
        "active_admin_count": active_admin_count,
        "current_unit": current_unit,
        "unit_branding": branding,
    }


def register_navigation_context(
    app: Flask, dependencies: NavigationContextDependencies
) -> Callable[[], dict[str, Any]]:
    """Register the authenticated navigation context with Jinja."""

    def inject_navigation_context() -> dict[str, Any]:
        return build_navigation_context(current_user, request.endpoint, dependencies)

    inject_navigation_context.__name__ = "inject_perms"
    app.context_processor(inject_navigation_context)
    return inject_navigation_context


def _branding_for(unit: Any) -> dict[str, str]:
    branding: dict[str, Any] = {}
    if unit:
        try:
            candidate = json.loads(unit.branding_json or "{}")
            if isinstance(candidate, dict):
                branding = candidate
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    primary = branding.get("primary_colour", "")
    accent = branding.get("accent_colour", "")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", primary):
        primary = ""
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", accent):
        accent = ""
    return {
        "primary_colour": primary,
        "accent_colour": accent,
        "display_name": (branding.get("display_name") or (unit.name if unit else ""))[
            :120
        ],
    }


def _shows_briefing_count(endpoint: str | None) -> bool:
    return endpoint in {"index", "module_home"} or bool(
        endpoint and endpoint.startswith("briefing.")
    )
