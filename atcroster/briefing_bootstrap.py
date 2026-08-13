"""Deferred briefing-module loading for application composition."""

from __future__ import annotations

from types import SimpleNamespace


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
    )
