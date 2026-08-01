from __future__ import annotations

from flask import Flask

import app as application_module
from atcroster.security.csrf import UNSAFE_METHODS, register_csrf_protection


def csrf_application() -> Flask:
    application = Flask(__name__)
    application.secret_key = "isolated-csrf-tests"
    register_csrf_protection(application)

    @application.get("/token")
    def token():
        return application.jinja_env.globals["csrf_token"]()

    @application.post("/form")
    def form_mutation():
        return "accepted"

    @application.patch("/header")
    def header_mutation():
        return "accepted"

    return application


def test_token_is_stable_within_session_and_form_token_is_accepted():
    client = csrf_application().test_client()
    token = client.get("/token").get_data(as_text=True)
    assert token
    assert client.get("/token").get_data(as_text=True) == token
    assert client.post("/form", data={"_csrf_token": token}).status_code == 200


def test_header_token_is_accepted_and_missing_or_invalid_tokens_are_rejected():
    client = csrf_application().test_client()
    token = client.get("/token").get_data(as_text=True)
    assert client.patch(
        "/header", headers={"X-CSRF-Token": token}
    ).status_code == 200
    assert client.post("/form").status_code == 400
    assert client.post("/form", data={"_csrf_token": "wrong"}).status_code == 400


def test_every_registered_unsafe_browser_route_uses_the_global_hook():
    unsafe_endpoints = {
        rule.endpoint
        for rule in application_module.app.url_map.iter_rules()
        if set(rule.methods or ()) & UNSAFE_METHODS
    }
    hooks = application_module.app.before_request_funcs.get(None, ())

    assert unsafe_endpoints
    assert application_module._enforce_csrf in hooks
    assert not hasattr(application_module._enforce_csrf, "exempt_endpoints")
