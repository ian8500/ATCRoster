"""Contract for application-module exports retained during incremental extraction."""

from __future__ import annotations

import ast
from pathlib import Path

import app


def test_public_application_compatibility_exports_are_available():
    """Keep established integration imports explicit and independently tested."""
    expected_callables = (
        "_active_roster_publication",
        "_lock_date_for_target_month",
        "_request_date_bounds",
        "_roster_snapshot",
        "_send_account_email",
        "_send_sms_via_messagemedia",
        "_shift_by_code",
        "bootstrap_reference_data",
        "get_shift",
        "lock_date_for_month",
        "month_has_data",
        "roster_edit_required",
        "shift_counter_group",
        "shift_counter_group_for_day",
        "tenant_get",
        "would_trigger_fatigue",
        "would_trigger_fatigue_with_plan",
    )
    expected_models = (
        "Assignment",
        "DatabaseRoutingMetadata",
        "FeatureFlag",
        "PlatformIdentity",
        "RosterPublication",
        "Staff",
        "Unit",
        "UnitMembership",
    )

    for name in expected_callables:
        assert callable(getattr(app, name)), name
    for name in expected_models:
        assert getattr(app, name) is not None, name

    assert app.application is app.app
    assert "special_requirement" in app.OPERATIONAL_TABLE_NAMES


def test_production_modules_do_not_import_the_composition_root():
    """Domains must receive collaborators explicitly, never via ``app``."""
    repository = Path(__file__).resolve().parents[1]
    production_sources = [
        *repository.glob("atcroster/**/*.py"),
        repository / "briefing_module.py",
    ]
    offenders = [
        str(source.relative_to(repository))
        for source in production_sources
        if source.name != "application.py"
        and any(
            (isinstance(node, ast.ImportFrom) and node.module in {"app", "atcroster.application"})
            or (
                isinstance(node, ast.Import)
                and any(alias.name in {"app", "atcroster.application"} for alias in node.names)
            )
            for node in ast.walk(ast.parse(source.read_text()))
        )
    ]
    assert offenders == []
