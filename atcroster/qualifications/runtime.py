"""Bound qualification history and operational-assurance services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .assurance import has_valid_endorsement, monthly_position_assurance
from .history import (
    qualification_snapshot,
    record_qualification_history,
    sync_legacy_roster_profile,
)


@dataclass(frozen=True)
class QualificationRuntimeDependencies:
    db: Any
    PersonQualificationHistory: Any
    PositionEndorsement: Any
    Assignment: Any
    OperationalPosition: Any
    PositionRequirement: Any
    current_user: Callable[[], Any]
    month_range: Callable[..., tuple[Any, list[Any]]]


class QualificationRuntime:
    """Own qualification history and position-assurance model interactions."""

    def __init__(self, dependencies: QualificationRuntimeDependencies):
        self.dependencies = dependencies

    @staticmethod
    def snapshot(record: Any) -> dict[str, Any]:
        return qualification_snapshot(record)

    def record_history(self, record: Any, action: str) -> None:
        deps = self.dependencies
        return record_qualification_history(
            record,
            action,
            db=deps.db,
            PersonQualificationHistory=deps.PersonQualificationHistory,
            actor_id=deps.current_user().id,
        )

    @staticmethod
    def sync_roster_profile(
        person: Any, qualification_type: Any, expires_on: Any
    ) -> None:
        return sync_legacy_roster_profile(person, qualification_type, expires_on)

    def valid_endorsement(self, person_id: int, position_id: int, on_day: Any) -> bool:
        return has_valid_endorsement(
            person_id,
            position_id,
            on_day,
            PositionEndorsement=self.dependencies.PositionEndorsement,
        )

    def position_assurance(self, year: int, month: int) -> list[dict[str, Any]]:
        deps = self.dependencies
        return monthly_position_assurance(
            year,
            month,
            Assignment=deps.Assignment,
            OperationalPosition=deps.OperationalPosition,
            PositionRequirement=deps.PositionRequirement,
            month_range=deps.month_range,
            valid_endorsement=self.valid_endorsement,
        )
