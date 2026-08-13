"""Roster-period lifecycle and derived protection status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from roster_horizon import get_unit_automatic_recalculation_start


ROSTER_PERIOD_STATUSES = frozenset({
    "CURRENT", "PROTECTED", "FUTURE_AUTOMATIC", "HISTORICAL", "CLOSED",
})


@dataclass(frozen=True)
class RosterPeriodDependencies:
    db: Any
    RosterPeriod: Any
    utcnow: Any


def create_roster_period_dependencies(
    *, db: Any, saas_models: Any, **services: Any
) -> RosterPeriodDependencies:
    """Bind roster-period records at the roster lifecycle boundary."""
    return RosterPeriodDependencies(
        db=db,
        RosterPeriod=saas_models.RosterPeriod,
        **services,
    )


class RosterPeriodService:
    def __init__(self, dependencies: RosterPeriodDependencies) -> None:
        self.dependencies = dependencies

    @staticmethod
    def status_for(unit: Any, year: int, month: int, reference_date: date) -> str:
        period_start = date(year, month, 1)
        current_start = reference_date.replace(day=1)
        if period_start < current_start:
            return "HISTORICAL"
        if period_start == current_start:
            return "CURRENT"
        boundary = get_unit_automatic_recalculation_start(unit, reference_date)
        return "PROTECTED" if period_start < boundary else "FUTURE_AUTOMATIC"

    def ensure_period(
        self, unit: Any, year: int, month: int, *, reference_date: date,
        generation_method: str = "AUTOMATIC",
        generation_version: str = "deterministic-baseline-v1",
        generated_by_user_id: int | None = None,
    ) -> tuple[Any, bool]:
        dep = self.dependencies
        row = dep.RosterPeriod.query.filter_by(
            unit_id=unit.id, year=year, month=month
        ).first()
        created = row is None
        if row is None:
            row = dep.RosterPeriod(
                unit_id=unit.id, year=year, month=month,
                status=self.status_for(unit, year, month, reference_date),
                generated_at=dep.utcnow(), generated_by_user_id=generated_by_user_id,
                generation_method=generation_method,
                generation_version=generation_version,
            )
            dep.db.session.add(row)
        elif row.status != "CLOSED":
            row.status = self.status_for(unit, year, month, reference_date)
        return row, created
