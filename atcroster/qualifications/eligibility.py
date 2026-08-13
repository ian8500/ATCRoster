"""Bound operational-capability and qualification eligibility services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from operational_capability import (
    OperationalCapabilityDependencies,
    OperationalCapabilityService,
)

from .status import staff_has_qualification, staff_is_countable


@dataclass(frozen=True)
class EligibilityDependencies:
    db: Any
    Staff: Any
    QualificationType: Any
    PersonQualification: Any
    authenticated_unit_id: Callable[[], int]
    today: Callable[[], date] = date.today


def create_eligibility_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> EligibilityDependencies:
    """Bind eligibility records within the qualifications domain."""
    return EligibilityDependencies(
        db=db,
        Staff=operational_models.Staff,
        QualificationType=saas_models.QualificationType,
        PersonQualification=saas_models.PersonQualification,
        **services,
    )


class EligibilityService:
    """Own dated operational capability and tenant-scoped eligibility checks."""

    def __init__(self, dependencies: EligibilityDependencies) -> None:
        self.dependencies = dependencies
        self._capabilities = OperationalCapabilityService(
            OperationalCapabilityDependencies(
                db=dependencies.db,
                Staff=dependencies.Staff,
                QualificationType=dependencies.QualificationType,
                PersonQualification=dependencies.PersonQualification,
            )
        )

    def operational_capability(self, staff_id: int, on_date: date) -> Any:
        return self._capabilities.get_staff_operational_capability(staff_id, on_date)

    def capability_service(self) -> OperationalCapabilityService:
        """Return the bound service for the legacy compatibility export."""
        return self._capabilities

    def capability_matrix(
        self, staff: list[Any], days: list[date]
    ) -> dict[tuple[int, date], Any]:
        return self._capabilities.get_capability_matrix(staff, days)

    def is_countable(self, staff: Any, on_date: date) -> bool:
        return staff_is_countable(
            staff, on_date, capability_for=self.operational_capability
        )

    def has_qualification(
        self, staff: Any, qualification_code: str, duty_date: date
    ) -> bool:
        deps = self.dependencies
        return staff_has_qualification(
            staff,
            qualification_code,
            duty_date,
            QualificationType=deps.QualificationType,
            PersonQualification=deps.PersonQualification,
            authenticated_unit_id=deps.authenticated_unit_id,
        )

    def has_shift_qualification(
        self, staff: Any, shift: Any, duty_date: date | None = None
    ) -> bool:
        return self.has_qualification(
            staff,
            shift.required_qualification,
            duty_date or self.dependencies.today(),
        )
