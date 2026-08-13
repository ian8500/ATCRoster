"""Default-deny CSRF protection for browser mutation routes."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from flask import Flask, abort, request, session

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def csrf_token() -> str:
    """Return the session token, creating it with cryptographic randomness."""
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def validate_csrf() -> None:
    """Validate form or header input using a constant-time comparison."""
    supplied = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
    expected = session.get("_csrf_token")
    if (
        not expected
        or not supplied
        or not secrets.compare_digest(str(expected), str(supplied))
    ):
        abort(400, "Invalid or missing CSRF token.")


def register_csrf_protection(
    app: Flask,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """Register the Jinja helper and global unsafe-method enforcement hook."""
    app.jinja_env.globals["csrf_token"] = csrf_token

    def enforce_csrf() -> None:
        if request.method in UNSAFE_METHODS:
            validate_csrf()

    enforce_csrf.__name__ = "_enforce_csrf"
    app.before_request(enforce_csrf)
    return validate_csrf, enforce_csrf
