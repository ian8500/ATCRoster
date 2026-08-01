"""Compatibility contract for every registered Flask route."""

from __future__ import annotations

import json
from pathlib import Path

import app


FIXTURE = Path(__file__).parent / "fixtures" / "route_map.json"
IGNORED_AUTOMATIC_METHODS = {"HEAD", "OPTIONS"}


def current_route_map() -> list[dict[str, object]]:
    """Return stable route facts, excluding Flask's automatic methods."""
    routes = [
        {
            "endpoint": rule.endpoint,
            "methods": sorted(set(rule.methods or ()) - IGNORED_AUTOMATIC_METHODS),
            "rule": rule.rule,
        }
        for rule in app.app.url_map.iter_rules()
    ]
    return sorted(
        routes,
        key=lambda route: (route["rule"], route["endpoint"], route["methods"]),
    )


def test_route_map_matches_compatibility_snapshot():
    expected = json.loads(FIXTURE.read_text())
    assert current_route_map() == expected

