from __future__ import annotations

from flask import Flask, g
from flask_login import LoginManager

from atcroster.security.headers import (
    SecurityHeaderDependencies,
    content_security_policy,
    register_security_headers,
)


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
