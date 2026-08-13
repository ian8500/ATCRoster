"""Deterministic roster maintenance CLI commands."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import click
from flask.cli import with_appcontext


@dataclass(frozen=True)
class RosterCliDependencies:
    db: Any
    Unit: Any
    RosterImpactEventType: Any
    add_months: Callable[[int, int, int], tuple[int, int]]
    roster_period_service: Any
    roster_impact_service: Callable[[], Any]


def create_roster_cli(dependencies: RosterCliDependencies):
    """Build the roster CLI group for composition-root registration."""

    @click.group("roster")
    def roster_cli():
        """Deterministic roster maintenance commands."""

    @roster_cli.command("ensure-future-periods")
    @with_appcontext
    @click.option("--months-ahead", type=click.IntRange(min=0, max=60), default=None)
    @click.option(
        "--unit-code", default=None, help="Limit maintenance to one airport code."
    )
    def ensure_future_roster_periods(months_ahead, unit_code):
        configured = int(
            months_ahead
            if months_ahead is not None
            else os.environ.get("ROSTER_GENERATION_MONTHS_AHEAD", "18")
        )
        query = dependencies.Unit.query.filter(
            dependencies.Unit.status == "active", dependencies.Unit.code != "CTRL"
        )
        if unit_code:
            query = query.filter(
                dependencies.db.func.upper(dependencies.Unit.code)
                == unit_code.strip().upper()
            )
        created_periods = generated_periods = 0
        for unit in query.order_by(dependencies.Unit.id).all():
            reference = datetime.now(ZoneInfo(unit.timezone or "Europe/London")).date()
            for offset in range(configured + 1):
                year, month = dependencies.add_months(
                    reference.year, reference.month, offset
                )
                period, created = dependencies.roster_period_service.ensure_period(
                    unit, year, month, reference_date=reference
                )
                dependencies.db.session.flush()
                created_periods += int(created)
                if created and period.status == "FUTURE_AUTOMATIC":
                    start = date(year, month, 1)
                    next_year, next_month = dependencies.add_months(year, month, 1)
                    dependencies.roster_impact_service().handle_roster_impact_event(
                        unit.id,
                        dependencies.RosterImpactEventType.FUTURE_PERIOD_CREATED,
                        start,
                        effective_to=date(next_year, next_month, 1) - timedelta(days=1),
                        rebuild_baseline=True,
                        reason=f"Automatic future roster period {year:04d}-{month:02d} created.",
                        reference_date=reference,
                    )
                    generated_periods += 1
                dependencies.db.session.commit()
        click.echo(
            f"Roster horizon ready: {created_periods} period(s) created, {generated_periods} future period(s) populated."
        )

    return roster_cli
