ENDPOINTS = {
    "training_home": ("/training/", {"GET"}),
    "training_profile": ("/training/<int:sid>", {"GET", "POST"}),
    "competency_home": ("/competency/", {"GET"}),
    "competency_profile": ("/competency/<int:sid>", {"GET", "POST"}),
    "training_admin": ("/training/admin", {"GET", "POST"}),
    "training_analytics": ("/training/analytics", {"GET"}),
}


def test_training_blueprint_preserves_global_endpoint_contract():
    import app

    rules = {rule.endpoint: rule for rule in app.app.url_map.iter_rules()}
    for endpoint, (path, methods) in ENDPOINTS.items():
        assert endpoint in rules
        assert rules[endpoint].rule == path
        assert methods <= rules[endpoint].methods


def test_training_routes_are_owned_outside_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for path, _methods in ENDPOINTS.values():
        assert f'@app.route("{path}"' not in source
        assert f'@app.get("{path}"' not in source


def test_training_dashboard_handlers_are_no_longer_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def training_home():" not in source
    assert "def training_profile(" not in source
