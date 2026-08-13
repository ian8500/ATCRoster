from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metar_fetch_rejects_non_json_non_success_and_unbounded_responses():
    source = (ROOT / "handover_blueprint.py").read_text()

    assert "200 <= response.status < 300" in source
    assert '"application/json" not in content_type' in source
    assert "response.read(65_537)" in source
    assert "len(payload_bytes) > 65_536" in source
    assert "except (http.client.HTTPException, OSError, UnicodeDecodeError, json.JSONDecodeError)" in source
    assert "metar_fetch_failed" in source
