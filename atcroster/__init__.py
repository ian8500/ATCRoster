"""ATCRoster application construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from atcroster.config import (
    RuntimeSettings,
    environment_snapshot,
    load_flask_config,
    runtime_settings,
)


def create_app(
    config_object: Mapping[str, Any] | object | None = None,
    *,
    validate_external_services: bool = True,
) -> Flask:
    """Create a configured Flask application without importing legacy routes."""

    repository = Path(__file__).resolve().parents[1]
    app = Flask(
        "atcroster",
        instance_path=str(repository / "instance"),
        template_folder=str(repository / "templates"),
        static_folder=str(repository / "static"),
    )
    environ = environment_snapshot()
    app.config.from_mapping(load_flask_config(environ, app.instance_path))
    explicit_keys: set[str] = set()
    if config_object is not None:
        if isinstance(config_object, Mapping):
            explicit_keys = set(config_object)
            app.config.from_mapping(config_object)
        else:
            app.config.from_object(config_object)
    if (
        str(app.config.get("ATCROSTER_ENVIRONMENT", "")).lower() == "production"
        and "SESSION_COOKIE_NAME" not in explicit_keys
    ):
        app.config["SESSION_COOKIE_NAME"] = "__Host-atcroster"

    settings = runtime_settings(app.config, environ)
    app.extensions["atcroster_runtime_settings"] = settings
    if settings.trusted_proxy_hops:
        app.wsgi_app = ProxyFix(  # type: ignore[method-assign]
            app.wsgi_app,
            x_for=settings.trusted_proxy_hops,
            x_proto=settings.trusted_proxy_hops,
            x_host=settings.trusted_proxy_hops,
        )

    if validate_external_services and settings.deployment_environment == "production":
        _validate_external_services(app)
    return app


def _validate_external_services(app: Flask) -> None:
    """Validate required production integrations without opening databases."""

    from briefing_storage import configured_briefing_storage
    from platform_provisioning import validate_token_encryption_config

    configured_briefing_storage(app.instance_path)
    validate_token_encryption_config()


def get_runtime_settings(app: Flask) -> RuntimeSettings:
    return app.extensions["atcroster_runtime_settings"]
