from __future__ import annotations

from copy import deepcopy

import pytest

from atcroster import create_app, get_runtime_settings


def production_config() -> dict[str, object]:
    return {
        "ATCROSTER_ENVIRONMENT": "production",
        "ATCROSTER_TRUSTED_PROXY_HOPS": 1,
        "ATCROSTER_FIELD_ENCRYPTION_KEYS": (
            "v1:QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE="
        ),
        "SECRET_KEY": "production-test-secret-with-32-characters",
        "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://example.invalid/control",
        "SESSION_COOKIE_SECURE": True,
        "TRUSTED_HOSTS": ["example.invalid"],
        "REDIS_URL": "redis://example.invalid/0",
        "ATCROSTER_INTERNAL_METRICS_TOKEN": "metrics-test-token-with-32-characters",
    }


def test_create_app_returns_isolated_configured_applications():
    first = create_app(
        {"TESTING": True, "SECRET_KEY": "first"},
        validate_external_services=False,
    )
    second = create_app(
        {"TESTING": True, "SECRET_KEY": "second"},
        validate_external_services=False,
    )

    assert first is not second
    assert first.config["SECRET_KEY"] == "first"
    assert second.config["SECRET_KEY"] == "second"
    assert get_runtime_settings(first).deployment_environment == "development"
    assert get_runtime_settings(second).deployment_environment == "development"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"TRUSTED_HOSTS": []}, "ATCROSTER_TRUSTED_HOSTS"),
        ({"ATCROSTER_TRUSTED_PROXY_HOPS": 4}, "between 0 and 3"),
        ({"SECRET_KEY": "short"}, "FLASK_SECRET_KEY"),
        ({"SQLALCHEMY_DATABASE_URI": "sqlite:///unsafe.db"}, "PostgreSQL"),
        ({"SESSION_COOKIE_SECURE": False}, "SECURE_COOKIES"),
        (
            {
                "ATCROSTER_FIELD_ENCRYPTION_KEYS": "",
                "ATCROSTER_FIELD_ENCRYPTION_KEY": "",
            },
            "FIELD_ENCRYPTION_KEYS",
        ),
        ({"REDIS_URL": ""}, "REDIS_URL"),
        ({"ATCROSTER_INTERNAL_METRICS_TOKEN": "short"}, "METRICS_TOKEN"),
    ],
)
def test_production_configuration_fails_closed(update, message):
    config = deepcopy(production_config())
    config.update(update)

    with pytest.raises(RuntimeError, match=message):
        create_app(config, validate_external_services=False)


def test_proxy_hops_must_be_an_integer():
    config = production_config()
    config["ATCROSTER_TRUSTED_PROXY_HOPS"] = "not-an-integer"

    with pytest.raises(RuntimeError, match="must be an integer"):
        create_app(config, validate_external_services=False)


def test_production_configuration_uses_host_allowlist_and_host_cookie():
    app = create_app(production_config(), validate_external_services=False)

    assert app.config["TRUSTED_HOSTS"] == ["example.invalid"]
    assert app.config["SESSION_COOKIE_NAME"] == "__Host-atcroster"
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert get_runtime_settings(app).trusted_proxy_hops == 1
