"""Explicit application configuration loading and production validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
import base64
import hashlib
import os
from pathlib import Path
from typing import Any


TRUE_VALUES = frozenset({"1", "true", "yes"})


@dataclass(frozen=True)
class RuntimeSettings:
    """Security-sensitive settings used outside Flask's config mapping."""

    deployment_environment: str
    field_encryption_key: str
    field_encryption_keys: str
    trusted_proxy_hops: int


def load_flask_config(
    environ: Mapping[str, str],
    instance_path: str,
) -> dict[str, Any]:
    """Build Flask configuration without connecting to external services."""

    deployment_environment = environ.get("ATCROSTER_ENVIRONMENT", "development").lower()
    database_url = environ.get(
        "DATABASE_URL",
        f"sqlite:///{Path(instance_path) / 'roster.db'}",
    )
    engine_options: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_size": 5,
        "max_overflow": 5,
        "pool_timeout": int(environ.get("ATCROSTER_DB_POOL_TIMEOUT_SECONDS", "10")),
    }
    if database_url.startswith("postgresql"):
        engine_options["connect_args"] = {
            "connect_timeout": int(
                environ.get("ATCROSTER_DB_CONNECT_TIMEOUT_SECONDS", "5")
            ),
            "options": (
                "-c statement_timeout="
                + str(int(environ.get("ATCROSTER_DB_STATEMENT_TIMEOUT_MS", "15000")))
            ),
        }
    return {
        "ATCROSTER_ENVIRONMENT": deployment_environment,
        "SECRET_KEY": environ.get("FLASK_SECRET_KEY", "fallback-change-me"),
        "SQLALCHEMY_DATABASE_URI": database_url,
        "SQLALCHEMY_ENGINE_OPTIONS": engine_options,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "MAX_CONTENT_LENGTH": int(
            environ.get("ATCROSTER_MAX_REQUEST_BYTES", str(2 * 1024 * 1024))
        ),
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_NAME": environ.get(
            "ATCROSTER_SESSION_COOKIE_NAME",
            (
                "__Host-atcroster"
                if deployment_environment == "production"
                else "atcroster_session"
            ),
        ),
        "SESSION_COOKIE_SECURE": environ.get("ATCROSTER_SECURE_COOKIES", "").lower()
        in TRUE_VALUES,
        "PERMANENT_SESSION_LIFETIME": timedelta(
            minutes=int(environ.get("ATCROSTER_SESSION_ABSOLUTE_MINUTES", "720"))
        ),
        "PREFERRED_URL_SCHEME": "https",
        "TRUSTED_HOSTS": (
            [
                host.strip()
                for host in environ.get("ATCROSTER_TRUSTED_HOSTS", "").split(",")
                if host.strip()
            ]
            or None
        ),
    }


def runtime_settings(
    config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> RuntimeSettings:
    """Resolve and validate non-Flask runtime settings."""

    deployment_environment = str(
        config.get("ATCROSTER_ENVIRONMENT", "development")
    ).lower()
    try:
        trusted_proxy_hops = int(
            config.get(
                "ATCROSTER_TRUSTED_PROXY_HOPS",
                environ.get("ATCROSTER_TRUSTED_PROXY_HOPS", "0"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ATCROSTER_TRUSTED_PROXY_HOPS must be an integer.") from exc
    if not 0 <= trusted_proxy_hops <= 3:
        raise RuntimeError("ATCROSTER_TRUSTED_PROXY_HOPS must be between 0 and 3.")

    field_encryption_key = str(
        config.get(
            "ATCROSTER_FIELD_ENCRYPTION_KEY",
            environ.get("ATCROSTER_FIELD_ENCRYPTION_KEY", ""),
        )
    )
    field_encryption_keys = str(
        config.get(
            "ATCROSTER_FIELD_ENCRYPTION_KEYS",
            environ.get("ATCROSTER_FIELD_ENCRYPTION_KEYS", ""),
        )
    )
    if deployment_environment == "production":
        _validate_production_config(
            config=config,
            environ=environ,
            field_encryption_key=field_encryption_key,
            field_encryption_keys=field_encryption_keys,
        )
        if not field_encryption_keys:
            field_encryption_keys = f"legacy:{field_encryption_key}"
    else:
        field_encryption_key = base64.urlsafe_b64encode(
            hashlib.sha256(str(config["SECRET_KEY"]).encode()).digest()
        ).decode()
        field_encryption_keys = f"dev:{field_encryption_key}"

    return RuntimeSettings(
        deployment_environment=deployment_environment,
        field_encryption_key=field_encryption_key,
        field_encryption_keys=field_encryption_keys,
        trusted_proxy_hops=trusted_proxy_hops,
    )


def _validate_production_config(
    *,
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    field_encryption_key: str,
    field_encryption_keys: str,
) -> None:
    if not config.get("TRUSTED_HOSTS"):
        raise RuntimeError("Production requires ATCROSTER_TRUSTED_HOSTS.")
    if (
        "ATCROSTER_TRUSTED_PROXY_HOPS" not in environ
        and "ATCROSTER_TRUSTED_PROXY_HOPS" not in config
    ):
        raise RuntimeError(
            "Production requires an explicit ATCROSTER_TRUSTED_PROXY_HOPS value."
        )
    secret_key = str(config.get("SECRET_KEY", ""))
    if secret_key == "fallback-change-me" or len(secret_key) < 32:
        raise RuntimeError(
            "Production requires FLASK_SECRET_KEY with at least 32 characters."
        )
    if str(config.get("SQLALCHEMY_DATABASE_URI", "")).startswith("sqlite"):
        raise RuntimeError("Production requires PostgreSQL; SQLite is not supported.")
    if not config.get("SESSION_COOKIE_SECURE"):
        raise RuntimeError("Production requires ATCROSTER_SECURE_COOKIES=true.")
    if not field_encryption_key and not field_encryption_keys:
        raise RuntimeError("Production requires ATCROSTER_FIELD_ENCRYPTION_KEYS.")
    if not environ.get("REDIS_URL") and not config.get("REDIS_URL"):
        raise RuntimeError("Production requires REDIS_URL.")


def environment_snapshot() -> dict[str, str]:
    """Return a copy so factory tests cannot mutate process configuration."""

    return dict(os.environ)
