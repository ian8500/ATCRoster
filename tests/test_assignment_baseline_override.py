import app


def test_effective_code_prefers_override_then_generated_then_legacy():
    row = app.Assignment(code="LEGACY", generated_code="M", override_code="A")
    assert row.effective_code == "A"
    row.override_code = None
    assert row.effective_code == "M"
    row.generated_code = None
    assert row.effective_code == "LEGACY"


def test_empty_override_is_distinct_from_no_override():
    row = app.Assignment(code="LEGACY", generated_code="M", override_code="")
    assert row.effective_code == ""


def test_generated_recalculation_preserves_override_and_legacy_value():
    row = app.Assignment(code="A", generated_code="M", override_code="A")
    row.set_generated_baseline("N", generation_version="test-v1")
    assert row.generated_code == "N"
    assert row.override_code == "A"
    assert row.effective_code == "A"
    assert row.code == "A"


def test_clearing_override_reveals_and_materialises_baseline():
    row = app.Assignment(code="A", generated_code="M", override_code="A")
    row.clear_editor_override()
    assert row.override_code is None
    assert row.effective_code == "M"
    assert row.code == "M"
