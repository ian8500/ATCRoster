"""Route ownership for operations assurance and planning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flask import Blueprint


@dataclass(frozen=True)
class OperationsDependencies:
    operations_assurance: Callable
    coverage_heatmap: Callable
    scenarios_page: Callable


def create_operations_blueprint(dependencies: OperationsDependencies) -> Blueprint:
    blueprint = Blueprint("operations", __name__)

    @blueprint.record_once
    def register_routes(state):
        routes = (
            (
                "/operations/<ym>",
                "operations_assurance",
                dependencies.operations_assurance,
                ["GET", "POST"],
            ),
            (
                "/planning/coverage/<ym>",
                "coverage_heatmap",
                dependencies.coverage_heatmap,
                ["GET"],
            ),
            (
                "/planning/scenarios",
                "scenarios_page",
                dependencies.scenarios_page,
                ["GET", "POST"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
