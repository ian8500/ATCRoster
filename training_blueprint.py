"""Route ownership for training and competency workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flask import Blueprint


@dataclass(frozen=True)
class TrainingDependencies:
    training_home: Callable
    training_profile: Callable
    competency_home: Callable
    competency_profile: Callable
    training_admin: Callable
    training_analytics: Callable


def create_training_blueprint(dependencies: TrainingDependencies) -> Blueprint:
    blueprint = Blueprint("training", __name__)

    @blueprint.record_once
    def register_routes(state):
        routes = (
            ("/training/", "training_home", dependencies.training_home, ["GET"]),
            (
                "/training/<int:sid>",
                "training_profile",
                dependencies.training_profile,
                ["GET", "POST"],
            ),
            ("/competency/", "competency_home", dependencies.competency_home, ["GET"]),
            (
                "/competency/<int:sid>",
                "competency_profile",
                dependencies.competency_profile,
                ["GET", "POST"],
            ),
            (
                "/training/admin",
                "training_admin",
                dependencies.training_admin,
                ["GET", "POST"],
            ),
            (
                "/training/analytics",
                "training_analytics",
                dependencies.training_analytics,
                ["GET"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule, endpoint=endpoint, view_func=view_func, methods=methods
            )

    return blueprint
