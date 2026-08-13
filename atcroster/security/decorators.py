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
