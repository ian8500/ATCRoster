from __future__ import annotations

from flask import Flask
from jinja2 import DictLoader

from atcroster.public import public_blueprint
from atcroster.public.blueprint import legal_context


def public_application(tmp_path) -> Flask:
    static_folder = tmp_path / "static"
    static_folder.mkdir()
    (static_folder / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
    application = Flask(__name__, static_folder=str(static_folder))
    application.jinja_loader = DictLoader(
        {
            "privacy.html": "privacy|{{ legal_entity }}|{{ privacy_email }}",
            "cookies.html": "cookies|{{ policy_date }}",
            "terms.html": "terms|{{ legal_address }}",
            "subprocessors.html": "subprocessors|{{ company_number }}",
        }
    )
    application.register_blueprint(public_blueprint)
    return application


def test_public_blueprint_preserves_legacy_routes_and_endpoints(tmp_path):
    application = public_application(tmp_path)
    routes = {
        rule.rule: (rule.endpoint, sorted(rule.methods - {"HEAD", "OPTIONS"}))
        for rule in application.url_map.iter_rules()
        if rule.rule != "/static/<path:filename>"
    }
    assert routes == {
        "/favicon.ico": ("favicon", ["GET"]),
        "/privacy": ("privacy_notice", ["GET"]),
        "/cookies": ("cookie_notice", ["GET"]),
        "/terms": ("terms_of_service", ["GET"]),
        "/subprocessors": ("subprocessor_notice", ["GET"]),
    }


def test_public_pages_render_and_favicon_is_served(tmp_path, monkeypatch):
    monkeypatch.setenv("ATCROSTER_LEGAL_ENTITY", "Example Aviation")
    monkeypatch.setenv("ATCROSTER_PRIVACY_EMAIL", "privacy@example.test")
    application = public_application(tmp_path)
    client = application.test_client()

    assert b"privacy|Example Aviation|privacy@example.test" in client.get(
        "/privacy"
    ).data
    assert client.get("/cookies").status_code == 200
    assert client.get("/terms").status_code == 200
    assert client.get("/subprocessors").status_code == 200
    favicon_response = client.get("/favicon.ico")
    assert favicon_response.status_code == 200
    assert favicon_response.mimetype == "image/svg+xml"


def test_legal_context_preserves_support_email_fallback(monkeypatch):
    monkeypatch.delenv("ATCROSTER_PRIVACY_EMAIL", raising=False)
    monkeypatch.setenv("ATCROSTER_SUPPORT_EMAIL", "support@example.test")
    assert legal_context()["privacy_email"] == "support@example.test"
