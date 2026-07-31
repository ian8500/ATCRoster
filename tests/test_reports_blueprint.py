REPORT_ENDPOINTS = {
    "metrics": ("/metrics", {"GET"}),
    "metrics_export": ("/metrics/export", {"GET"}),
    "report_leave": ("/reports/leave/<ym>", {"GET"}),
    "report_leave_csv": ("/reports/leave.csv", {"GET"}),
    "report_leave_year": ("/reports/leave-year", {"GET"}),
    "report_sickness": ("/reports/sickness", {"GET"}),
    "reports_index": ("/reports", {"GET", "POST"}),
}


def test_reports_blueprint_preserves_global_endpoint_contract():
    import app

    rules = {rule.endpoint: rule for rule in app.app.url_map.iter_rules()}
    for endpoint, (path, methods) in REPORT_ENDPOINTS.items():
        assert endpoint in rules
        assert rules[endpoint].rule == path
        assert methods <= rules[endpoint].methods
        assert rules[endpoint].endpoint == endpoint


def test_report_routes_are_no_longer_declared_in_legacy_module():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "app.py").read_text()
    for path, _methods in REPORT_ENDPOINTS.values():
        assert f'@app.route("{path}"' not in source
