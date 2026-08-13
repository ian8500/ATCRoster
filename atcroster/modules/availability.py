"""Tenant module availability policy."""

from __future__ import annotations

from typing import Any


class ModuleAvailability:
    def __init__(self, FeatureFlag: Any) -> None:
        self.FeatureFlag = FeatureFlag

    def enabled(self, unit_id: int, key: str) -> bool:
        return bool(self.FeatureFlag.query.filter_by(
            unit_id=unit_id, key=key, enabled=True,
        ).first())

    def training(self, unit_id: int) -> bool:
        return self.enabled(unit_id, "training_module")

    def competency(self, unit_id: int) -> bool:
        row = self.FeatureFlag.query.filter_by(
            unit_id=unit_id, key="competency_module",
        ).first()
        # Existing airports inherit the combined-module entitlement until set.
        return bool(row.enabled) if row else self.training(unit_id)

    def live_position(self, unit_id: int) -> bool:
        return self.enabled(unit_id, "live_position_monitoring")
