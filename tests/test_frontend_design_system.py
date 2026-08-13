from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_design_system_modules_define_semantic_operational_states():
    tokens = (ROOT / "static/css/tokens.css").read_text()
    for token in (
        "--color-normal", "--color-warning", "--color-blocking",
        "--color-selected", "--color-published", "--color-advisory",
    ):
        assert token in tokens


def test_base_loads_modular_styles_in_design_system_order():
    base = (ROOT / "templates/base.html").read_text()
    expected = ("tokens.css", "layout.css", "components.css", "roster.css", "operations.css", "print.css")
    offsets = [base.index(name) for name in expected]
    assert offsets == sorted(offsets)


def test_roster_and_operations_modules_preserve_print_and_kiosk_contracts():
    roster = (ROOT / "static/css/roster.css").read_text()
    operations = (ROOT / "static/css/operations.css").read_text()
    assert "@media print" in roster
    assert ".live-position-kiosk" in operations


def test_visual_regression_contract_covers_roster_zoom_mobile_dialogs_and_tables():
    roster_template = (ROOT / "templates/roster_month.html").read_text()
    stylesheet = (ROOT / "static/styles.css").read_text()
    roster_layer = (ROOT / "static/css/roster.css").read_text()
    component_layer = (ROOT / "static/css/components.css").read_text()
    for zoom in ('data-roster-zoom="0.75"', 'data-roster-zoom="0.90"', 'data-roster-zoom="1"', 'data-roster-zoom="fit"'):
        assert zoom in roster_template
    assert "@media print" in roster_layer
    assert "@media(max-width:700px)" in stylesheet
    assert ".roster-command-palette" in stylesheet
    assert "table:not(.roster)" in stylesheet
    assert ":focus-visible" in component_layer
