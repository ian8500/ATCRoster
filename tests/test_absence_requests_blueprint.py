ENDPOINTS = {
    "leave": ("/leave", {"GET", "POST"}),
    "requests_page": ("/requests", {"GET", "POST"}),
    "admin_request_respond": ("/admin/requests/<int:rid>/respond", {"POST"}),
}


def test_absence_request_blueprint_preserves_global_endpoint_contract():
    import app

    rules = {rule.endpoint: rule for rule in app.app.url_map.iter_rules()}
    for endpoint, (path, methods) in ENDPOINTS.items():
        assert endpoint in rules
        assert rules[endpoint].rule == path
        assert methods <= rules[endpoint].methods


def test_absence_request_routes_are_owned_outside_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for path, _methods in ENDPOINTS.values():
        assert f'@app.route("{path}"' not in source


def test_leave_handler_is_no_longer_implemented_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def leave():" not in source


def test_shift_request_handler_is_no_longer_implemented_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def requests_page():" not in source


def test_manager_request_handler_is_no_longer_implemented_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "def admin_request_respond(" not in source
