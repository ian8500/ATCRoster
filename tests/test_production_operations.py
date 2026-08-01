from __future__ import annotations

import json
import logging

import app
from production_operations import JsonFormatter, MetricsRegistry, structured_event


def test_structured_security_event_keeps_only_allowlisted_fields():
    class Capture(logging.Handler):
        record = None

        def emit(self, record):
            self.record = record

    logger = logging.getLogger("structured-event-test")
    logger.handlers.clear()
    logger.propagate = False
    capture = Capture()
    logger.addHandler(capture)
    structured_event(
        logger,
        "login_failed",
        request_id="request-7",
        unit_id=3,
        password="must-not-appear",
        sickness_description="must-not-appear",
    )
    assert capture.record.structured_fields == {
        "event": "login_failed",
        "request_id": "request-7",
        "unit_id": 3,
    }


def test_json_formatter_emits_required_production_envelope(monkeypatch):
    monkeypatch.setenv("ATCROSTER_ENVIRONMENT", "production")
    monkeypatch.setenv("ATCROSTER_COMMIT_SHA", "abc123")
    record = logging.LogRecord(
        "test", logging.WARNING, __file__, 1, "worker_failed", (), None
    )
    record.structured_fields = {
        "request_id": "correlation-1",
        "retry_count": 5,
        "password": "excluded",
    }
    payload = json.loads(JsonFormatter().format(record))
    assert payload["service"] == "atcroster-web"
    assert payload["version"] == "abc123"
    assert payload["environment"] == "production"
    assert payload["severity"] == "warning"
    assert payload["request_id"] == "correlation-1"
    assert "password" not in payload


def test_metrics_registry_renders_prometheus_text_without_personal_labels():
    registry = MetricsRegistry()
    registry.add("http_requests_total", route="health_ready", status="200")
    rendered = registry.render()
    assert rendered.startswith("atcroster_http_requests_total{")
    assert 'route="health_ready"' in rendered
    assert 'status="200"' in rendered


def test_internal_metrics_and_diagnostics_require_bearer_token():
    client = app.app.test_client()
    token = "internal-monitoring-test-token-32-characters"
    original = app.app.config.get("ATCROSTER_INTERNAL_METRICS_TOKEN")
    app.app.config["ATCROSTER_INTERNAL_METRICS_TOKEN"] = token
    try:
        assert client.get("/internal/metrics").status_code == 403
        headers = {"Authorization": f"Bearer {token}"}
        metrics = client.get("/internal/metrics", headers=headers)
        assert metrics.status_code == 200
        assert metrics.content_type.startswith("text/plain")
        assert b"atcroster_http_requests_total" in metrics.data
        diagnostics = client.get("/internal/health", headers=headers)
        assert diagnostics.status_code == 200
        assert diagnostics.get_json()["database"] == "reachable"
    finally:
        app.app.config["ATCROSTER_INTERNAL_METRICS_TOKEN"] = original
