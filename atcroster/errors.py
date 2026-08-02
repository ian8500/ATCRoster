"""Application error handlers with explicit logging and security dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flask import Flask, Response, g, render_template, request, url_for
from flask_login import current_user
from werkzeug.exceptions import SecurityError


@dataclass(frozen=True)
class ErrorHandlerDependencies:
    security_event: Callable[..., None]


def module_error_navigation() -> dict[str, str]:
    """Keep module errors inside the module instead of linking to Roster."""
    if request.path.startswith("/briefing"):
        return {
            "home_url": url_for("briefing.home"),
            "home_label": "Return to briefing",
        }
    if request.path.startswith("/training"):
        return {
            "home_url": url_for("training_home"),
            "home_label": "Return to training",
        }
    if request.path.startswith("/competency"):
        return {
            "home_url": url_for("competency_home"),
            "home_label": "Return to competency",
        }
    return {}


def register_error_handlers(
    app: Flask,
    dependencies: ErrorHandlerDependencies,
) -> dict[int, Callable[..., Any]]:
    """Register the stable 400, 403, 404 and 500 response contracts."""

    def internal_error(error):
        app.logger.error(
            "unhandled_request_error request_id=%s path=%s",
            getattr(g, "request_id", ""),
            request.path,
            exc_info=error,
        )
        return (
            render_template(
                "error.html",
                request_id=getattr(g, "request_id", ""),
                **module_error_navigation(),
            ),
            500,
        )

    def bad_request(error):
        if isinstance(error, SecurityError):
            return Response(
                "Bad Request: untrusted host.",
                status=400,
                content_type="text/plain; charset=utf-8",
            )
        description = getattr(error, "description", "") or ""
        if "CSRF" in description:
            dependencies.security_event(
                "csrf_rejected", route=request.endpoint or "unmatched"
            )
            message = (
                "This page or form has expired. Reload the page and try the action "
                "once more."
            )
        elif description and not description.startswith(
            "The browser (or proxy) sent a request"
        ):
            message = description
        else:
            message = (
                "The request was not valid. Check the entered values and try again."
            )
        return (
            render_template(
                "error.html",
                status_code=400,
                error_title="We could not validate that request",
                error_message=message,
                request_id=getattr(g, "request_id", ""),
                **module_error_navigation(),
            ),
            400,
        )

    def forbidden(_error):
        dependencies.security_event(
            "forbidden_role_action",
            route=request.endpoint or "unmatched",
            unit_id=getattr(current_user, "unit_id", None),
            actor_id=getattr(current_user, "id", None),
        )
        is_platform_admin = (
            getattr(current_user, "is_authenticated", False)
            and getattr(current_user, "role", "") == "superadmin"
        )
        module_context = module_error_navigation()
        return (
            render_template(
                "error.html",
                status_code=403,
                error_title="You do not have access to this area",
                error_message=(
                    "Platform administrators cannot access airport personnel or "
                    "operational roster data. Return to Platform Administration."
                    if is_platform_admin
                    else (
                        "Your account role does not permit this action. Ask your Unit "
                        "Administrator for access."
                    )
                ),
                home_url=(
                    url_for("platform_admin")
                    if is_platform_admin
                    else module_context.get("home_url", url_for("index"))
                ),
                home_label=(
                    "Return to Platform Administration"
                    if is_platform_admin
                    else module_context.get("home_label", "Return to roster")
                ),
                request_id=getattr(g, "request_id", ""),
            ),
            403,
        )

    def not_found(_error):
        return (
            render_template(
                "error.html",
                status_code=404,
                error_title="That page or record was not found",
                error_message=(
                    "It may have moved, been removed, or belong to a different airport."
                ),
                request_id=getattr(g, "request_id", ""),
                **module_error_navigation(),
            ),
            404,
        )

    handlers = {
        400: bad_request,
        403: forbidden,
        404: not_found,
        500: internal_error,
    }
    names = {
        400: "_bad_request",
        403: "_forbidden",
        404: "_not_found",
        500: "_internal_error",
    }
    for status_code, handler in handlers.items():
        handler.__name__ = names[status_code]
        app.register_error_handler(status_code, handler)
    return handlers
