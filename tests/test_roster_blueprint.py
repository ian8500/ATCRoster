ROSTER_ENDPOINTS = {
    "roster_month_publish": ("/roster/<ym>/publish", {"POST"}),
    "roster_month_unpublish": ("/roster/<ym>/unpublish", {"POST"}),
    "roster_month": ("/roster/<ym>", {"GET"}),
    "assign_cell": ("/assign/<int:staff_id>/<ym>/<day>", {"POST"}),
    "roster_export_csv": ("/roster/<ym>/export", {"GET"}),
    "roster_print_view": ("/roster/<ym>/print", {"GET"}),
}


def test_roster_blueprint_preserves_global_endpoint_contract():
    import app

    rules = {rule.endpoint: rule for rule in app.app.url_map.iter_rules()}
    for endpoint, (path, methods) in ROSTER_ENDPOINTS.items():
        assert endpoint in rules
        assert rules[endpoint].rule == path
        assert methods <= rules[endpoint].methods


def test_roster_routes_are_no_longer_declared_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for path, _methods in ROSTER_ENDPOINTS.values():
        assert f'@app.route("{path}"' not in source


def test_completed_roster_handlers_are_no_longer_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for handler in (
        "roster_month_publish",
        "roster_month_unpublish",
        "roster_export_csv",
        "roster_print_view",
        "assign_cell",
    ):
        assert f"def {handler}(" not in source
