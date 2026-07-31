"""Route ownership for absence and shift-request workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flask import Blueprint


@dataclass(frozen=True)
class AbsenceRequestDependencies:
    leave: Callable
    requests_page: Callable
    admin_request_respond: Callable


def create_absence_requests_blueprint(
    dependencies: AbsenceRequestDependencies,
) -> Blueprint:
    blueprint = Blueprint("absence_requests", __name__)

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/leave", "leave", dependencies.leave, ["GET", "POST"]),
            ("/requests", "requests_page", dependencies.requests_page, ["GET", "POST"]),
            (
                "/admin/requests/<int:rid>/respond",
                "admin_request_respond",
                dependencies.admin_request_respond,
                ["POST"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
