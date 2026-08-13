"""Deferred briefing-module loading for application composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from types import SimpleNamespace


@dataclass(frozen=True)
class BriefingDependencies:
    """Cross-domain collaborators used by the optional briefing module."""

    FeatureFlag: Any
    Unit: Any
    Staff: Any
    Watch: Any
    ShiftType: Any
    Assignment: Any
    PlatformIdentity: Any
    active_roster_publication: Callable[[int, int], Any]


def load_briefing_module() -> SimpleNamespace:
    """Load briefing after the canonical database and model globals exist."""
    from briefing_module import (
        BriefingAssuranceRun,
        BriefingAudit,
        BriefingDelivery,
        BriefingItem,
        BriefingMessageType,
        briefing_blueprint,
        briefing_enabled,
        briefing_local_now,
        configure_briefing_dependencies,
    )

    return SimpleNamespace(
        BriefingAssuranceRun=BriefingAssuranceRun,
        BriefingAudit=BriefingAudit,
        BriefingDelivery=BriefingDelivery,
        BriefingItem=BriefingItem,
        BriefingMessageType=BriefingMessageType,
        blueprint=briefing_blueprint,
        enabled=briefing_enabled,
        local_now=briefing_local_now,
        configure_dependencies=configure_briefing_dependencies,
    )
