"""Platform-level feature flag definitions."""

from __future__ import annotations


PLATFORM_FEATURE_FLAGS = frozenset({
    "advanced_coverage", "scenario_planning", "calendar_exports",
    "fatigue_reporting", "custom_branding", "briefing_module",
    "training_module", "competency_module", "live_position_monitoring",
    "handover_module",
})

# Supporting capabilities stay available internally, but only these flags are
# exposed as launchable product modules in Super Admin controls.
PLATFORM_MODULE_FLAGS = frozenset({
    "briefing_module", "training_module", "competency_module",
    "live_position_monitoring", "handover_module",
})
