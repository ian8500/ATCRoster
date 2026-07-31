ENDPOINTS = {
    "operations_assurance": ("/operations/<ym>", {"GET", "POST"}),
    "coverage_heatmap": ("/planning/coverage/<ym>", {"GET"}),
    "scenarios_page": ("/planning/scenarios", {"GET", "POST"}),
}


def test_operations_blueprint_preserves_global_endpoint_contract():
    import app

    rules = {rule.endpoint: rule for rule in app.app.url_map.iter_rules()}
    for endpoint, (path, methods) in ENDPOINTS.items():
        assert endpoint in rules
        assert rules[endpoint].rule == path
        assert methods <= rules[endpoint].methods


def test_operations_routes_are_owned_outside_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for path, _methods in ENDPOINTS.values():
        assert f'@app.route("{path}"' not in source


def test_operations_handler_is_no_longer_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def operations_assurance(" not in source


def test_coverage_handler_is_no_longer_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def coverage_heatmap(" not in source


def test_scenarios_handler_is_no_longer_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def scenarios_page():" not in source
