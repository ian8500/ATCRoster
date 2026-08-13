"""Request-scoped authentication security runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from flask import g, request, session

from .events import record_security_event
from .mfa import decrypt_secret, matching_totp_step, totp_qr_data_uri
from .platform_login import pending_platform_login
from .rate_limits import consume_rate_limit, privacy_rate_limit_key, reset_rate_limit
from .sessions import credential_for_auth_stamp


@dataclass(frozen=True)
class AuthRuntimeDependencies:
    app: Any
    db: Any
    limiter: Any
    metrics: Any
    privacy_key: Callable[..., str]
    limiter_unavailable: type[Exception]
    structured_event: Callable[..., Any]
    PlatformIdentity: Any
    PlatformMfaCredential: Any
    MfaCredential: Any
    RecoveryRequest: Any
    decrypt_field: Callable[[str], str]
    now: Callable[[], Any]
    active_recovery_from_digest: Callable[..., Any]


class AuthRuntime:
    """Own rate limiting, security telemetry, and MFA request helpers."""

    def __init__(self, dependencies: AuthRuntimeDependencies):
        self.dependencies = dependencies

    def _rate_key(self, scope: str, subject: object) -> str:
        deps = self.dependencies
        return privacy_rate_limit_key(
            str(deps.app.config["SECRET_KEY"]),
            scope,
            request.remote_addr or "unknown",
            subject,
            deps.privacy_key,
        )

    def login_rate_key(self, username: str) -> str:
        return self._rate_key("login", username.lower())

    def consume_rate_limit(
        self,
        scope: str,
        subject: object,
        limit: int = 10,
        window: timedelta = timedelta(minutes=15),
        fail_closed: bool = True,
    ) -> bool:
        deps = self.dependencies
        return consume_rate_limit(
            limiter=deps.limiter,
            key=self._rate_key(scope, subject),
            limit=limit,
            window=window,
            unavailable=deps.limiter_unavailable,
            security_event=self.security_event,
            scope=scope,
            fail_closed=fail_closed,
        )

    def reset_rate_limit(self, scope: str, subject: object) -> None:
        deps = self.dependencies
        reset_rate_limit(
            limiter=deps.limiter,
            key=self._rate_key(scope, subject),
            unavailable=deps.limiter_unavailable,
            security_event=self.security_event,
            scope=scope,
        )

    def security_event(self, event: str, **safe_fields: Any) -> None:
        deps = self.dependencies
        record_security_event(
            metrics=deps.metrics,
            logger=deps.app.logger,
            request_id=getattr(g, "request_id", ""),
            structured_event=deps.structured_event,
            event=event,
            **safe_fields,
        )

    def credential_for_auth_stamp(self, user: Any):
        deps = self.dependencies
        return credential_for_auth_stamp(
            user, deps.PlatformMfaCredential, deps.MfaCredential
        )

    def active_recovery(self, field_name: str, raw_token: str, expected_state: str):
        deps = self.dependencies
        return deps.active_recovery_from_digest(
            deps.RecoveryRequest,
            field_name,
            raw_token,
            expected_state,
            deps.now,
        )

    def decrypt_mfa_secret(self, credential: Any) -> str:
        return decrypt_secret(credential, self.dependencies.decrypt_field)

    def matching_totp_step(self, secret: str, code: str) -> int | None:
        return matching_totp_step(secret, code, self.dependencies.now)

    def pending_platform_login(self):
        deps = self.dependencies
        return pending_platform_login(
            session,
            db=deps.db,
            PlatformIdentity=deps.PlatformIdentity,
        )

    @staticmethod
    def totp_qr_data_uri(provisioning_uri: str) -> str:
        return totp_qr_data_uri(provisioning_uri)
