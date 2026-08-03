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
