"""Roster route ownership extracted from the legacy application module.

The current handlers are injected while their remaining data-access concerns
are separated incrementally. Historical global endpoint names are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flask import Blueprint


@dataclass(frozen=True)
class RosterViews:
    roster_month_publish: Callable
    roster_month_unpublish: Callable
    roster_month: Callable
    assign_cell: Callable
    roster_export_csv: Callable
    roster_print_view: Callable


def create_roster_blueprint(views: RosterViews) -> Blueprint:
    blueprint = Blueprint("roster", __name__)

    @blueprint.record_once
    def register_routes(state):
        routes = (
            (
                "/roster/<ym>/publish",
                "roster_month_publish",
                views.roster_month_publish,
                ["POST"],
            ),
            (
                "/roster/<ym>/unpublish",
                "roster_month_unpublish",
                views.roster_month_unpublish,
                ["POST"],
            ),
            ("/roster/<ym>", "roster_month", views.roster_month, ["GET"]),
            (
                "/assign/<int:staff_id>/<ym>/<day>",
                "assign_cell",
                views.assign_cell,
                ["POST"],
            ),
            (
                "/roster/<ym>/export",
                "roster_export_csv",
                views.roster_export_csv,
                ["GET"],
            ),
            (
                "/roster/<ym>/print",
                "roster_print_view",
                views.roster_print_view,
                ["GET"],
            ),
        )
        for rule, endpoint, view_func, methods in routes:
            state.app.add_url_rule(
                rule,
                endpoint=endpoint,
                view_func=view_func,
                methods=methods,
            )

    return blueprint
