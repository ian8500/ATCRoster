"""Reusable Flask access-control decorators."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import abort
from flask_login import current_user


def create_admin_required(
    is_admin_user: Callable[[Any], bool],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create the legacy airport-admin route guard."""

    def admin_required(view: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any):
            if not current_user.is_authenticated or not is_admin_user(current_user):
                abort(403)
            return view(*args, **kwargs)

        return wrapper

    return admin_required


def create_roster_edit_required(
    current_user: Any, can_edit_roster: Callable[[Any], bool]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create the established roster-edit guard for legacy route consumers."""
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not current_user.is_authenticated or not can_edit_roster(current_user):
                return ("Forbidden", 403)
            return function(*args, **kwargs)

        return wrapper

    return decorator
