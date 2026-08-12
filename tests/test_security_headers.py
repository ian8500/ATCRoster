from __future__ import annotations

from flask import Flask, g
from flask_login import LoginManager

from atcroster.security.headers import (
    SecurityHeaderDependencies,
    content_security_policy,
    register_security_headers,
)
from production_operations import MetricsRegistry


def test_content_security_policy_preserves_nonce_sources_and_directives():
    policy = content_security_policy("nonce-value", production=False)
    assert policy == (
        "default-src 'self'; base-uri 'self'; form-action 'self'; "
        "frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "style-src 'self' 'nonce-nonce-value' https://fonts.googleapis.com "
        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "style-src-attr 'none'; script-src 'self' 'nonce-nonce-value' "
        "https://cdn.jsdelivr.net; connect-src 'self'; worker-src 'self'; "
        "manifest-src 'self'"
    )
    assert "'unsafe-inline'" not in policy


def test_production_policy_only_adds_upgrade_insecure_requests():
    development = content_security_policy("same", production=False)
    production = content_security_policy("same", production=True)
    assert production == development + "; upgrade-insecure-requests"


def test_registered_headers_preserve_nonce_hsts_and_metrics_completion():
    application = Flask(__name__)
    application.secret_key = "direct-header-test"
    login_manager = LoginManager(application)

    @login_manager.user_loader
    def load_user(_user_id):
        return None
    completed: list[dict[str, object]] = []

    @application.before_request
    def request_context():
        g.csp_nonce = "direct-test"
        g.request_id = "request-test"
        g.metrics_started_at = 4.0

    @application.get("/probe")
    def probe():
        return "ok"

    def finish(_metrics, started_at, **facts):
        completed.append({"started_at": started_at, **facts})
        return 0.125

    register_security_headers(
        application,
        SecurityHeaderDependencies(
            deployment_environment="production",
            metrics=object(),
            finish_request=finish,
        ),
    )

    response = application.test_client().get("/probe")
    assert response.headers["X-Request-ID"] == "request-test"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    assert "script-src 'self' 'nonce-direct-test'" in response.headers[
        "Content-Security-Policy"
    ]
    assert response.headers["Strict-Transport-Security"] == (
        "max-age=31536000; includeSubDomains"
    )
    assert completed == [
        {
            "started_at": 4.0,
            "route": "probe",
            "method": "GET",
            "status": 200,
        }
    ]


def test_roster_request_records_timing_and_slow_request_metric():
    application = Flask(__name__)
    application.secret_key = "roster-timing-test"
    login_manager = LoginManager(application)

    @login_manager.user_loader
    def load_user(_user_id):
        return None
    registry = MetricsRegistry()

    @application.before_request
    def request_context():
        g.metrics_started_at = 1.0

    @application.get("/roster")
    def roster_month():
        return "roster"

    register_security_headers(application, SecurityHeaderDependencies(
        deployment_environment="test", metrics=registry,
        finish_request=lambda *_args, **_kwargs: 2.5,
        slow_roster_seconds=2.0,
    ))
    response = application.test_client().get("/roster")
    rendered = registry.render()
    assert response.headers["Server-Timing"] == "roster;dur=2500.00"
    assert "atcroster_roster_page_requests_total 1" in rendered
    assert "atcroster_roster_page_slow_requests_total 1" in rendered


def test_versioned_static_assets_are_immutable_but_dynamic_responses_are_not():
    application = Flask(__name__, static_folder="../static")
    application.secret_key = "static-cache-test"
    register_security_headers(application, SecurityHeaderDependencies(
        deployment_environment="test", metrics=object(),
        finish_request=lambda *_args, **_kwargs: 0.0,
    ))

    response = application.test_client().get("/static/styles.css?v=123")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
