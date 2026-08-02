"""Authenticated browser-session lifecycle and revocation checks."""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from flask import Response, flash, redirect, session, url_for
from flask_login import logout_user


@dataclass(frozen=True)
class SessionLifecycleDependencies:
    now: Callable[[], datetime]
    credential_for_user: Callable[[Any], Any | None]
    security_event: Callable[..., None]


class SessionLifecycle:
    """Bind sessions to mutable identity state and enforce time limits."""

    def __init__(self, dependencies: SessionLifecycleDependencies) -> None:
        self._dependencies = dependencies

    def auth_stamp(self, user: Any) -> str:
        parts = [
            str(getattr(user, "password_hash", "")),
            str(getattr(user, "role", "")),
            str(getattr(user, "membership_status", "")),
        ]
        credential = self._dependencies.credential_for_user(user)
        enrolled_at = getattr(credential, "enrolled_at", None)
        if enrolled_at and enrolled_at.tzinfo is not None:
            enrolled_at = enrolled_at.astimezone(timezone.utc).replace(tzinfo=None)
        parts.extend(
            [
                str(bool(credential and credential.enabled)),
                str(bool(getattr(credential, "reset_required", False))),
                hashlib.sha256(
                    str(getattr(credential, "encrypted_secret", "")).encode()
                ).hexdigest(),
                enrolled_at.isoformat() if enrolled_at else "",
            ]
        )
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def initialize(self, user: Any, *, platform_mfa: bool = False) -> None:
        """Regenerate authenticated state after the final authentication factor."""
        now = self._dependencies.now()
        session.permanent = True
        session["_session_nonce"] = secrets.token_urlsafe(24)
        session["_session_started_at"] = now.isoformat()
        session["_last_seen_epoch"] = int(now.timestamp())
        session["_auth_stamp"] = self.auth_stamp(user)
        if platform_mfa:
            session["_platform_mfa_verified"] = True

    def enforce_request(self, user: Any) -> Response | None:
        """Clear rejected principals and revoke expired or stale sessions."""
        if session.get("_user_id") and not user.is_authenticated:
            session.clear()
            return None
        if not user.is_authenticated:
            return None

        now = self._dependencies.now()
        now_epoch = int(now.timestamp())
        idle_limit = int(os.environ.get("ATCROSTER_SESSION_IDLE_MINUTES", "30")) * 60
        absolute_limit = (
            int(os.environ.get("ATCROSTER_SESSION_ABSOLUTE_MINUTES", "720")) * 60
        )
        last_seen = int(session.get("_last_seen_epoch") or now_epoch)
        started_raw = session.get("_session_started_at")
        try:
            started_epoch = int(datetime.fromisoformat(str(started_raw)).timestamp())
        except (TypeError, ValueError):
            started_epoch = now_epoch
            session["_session_started_at"] = now.isoformat()
        expiry_reason = (
            "absolute"
            if now_epoch - started_epoch > absolute_limit
            else "idle"
            if now_epoch - last_seen > idle_limit
            else ""
        )
        if expiry_reason:
            self._dependencies.security_event(
                "session_expired",
                reason=expiry_reason,
                principal=self._principal_digest(user),
            )
            logout_user()
            session.clear()
            flash("Your secure session has expired. Sign in again.", "error")
            return redirect(url_for("login"))

        expected_stamp = session.get("_auth_stamp")
        current_stamp = self.auth_stamp(user)
        if expected_stamp and not secrets.compare_digest(
            str(expected_stamp), current_stamp
        ):
            self._dependencies.security_event(
                "session_forced_invalidation",
                principal=self._principal_digest(user),
            )
            logout_user()
            session.clear()
            flash(
                "Your account security or permissions changed. Sign in again.",
                "error",
            )
            return redirect(url_for("login"))
        session["_auth_stamp"] = current_stamp
        session["_last_seen_epoch"] = now_epoch
        return None

    @staticmethod
    def _principal_digest(user: Any) -> str:
        return hashlib.sha256(str(user.get_id()).encode()).hexdigest()[:16]
