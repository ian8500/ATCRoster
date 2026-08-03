"""Effective-dated operational contribution independent of roster placement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from collections import defaultdict
from typing import Any


INDEPENDENT_UE_CODES = frozenset({"ADI", "APS", "UE"})


@dataclass(frozen=True)
class OperationalCapability:
    in_unit: bool
    roster_active: bool
    medically_valid: bool
    independent_competencies: frozenset[str]
    supervised_competencies: frozenset[str]
    counts_as_operational: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OperationalCapabilityDependencies:
    db: Any
    Staff: Any
    QualificationType: Any
    PersonQualification: Any


class OperationalCapabilityService:
    def __init__(self, dependencies: OperationalCapabilityDependencies) -> None:
        self.dependencies = dependencies

    def get_staff_operational_capability(
        self, staff_id: int, on_date: date
    ) -> OperationalCapability:
        dep = self.dependencies
        person = dep.db.session.get(dep.Staff, int(staff_id))
        if not person:
            return OperationalCapability(
                False, False, False, frozenset(), frozenset(), False,
                ("Staff member does not exist.",),
            )
        qualifications = dep.db.session.query(
            dep.PersonQualification, dep.QualificationType
        ).join(
            dep.QualificationType,
            dep.QualificationType.id
            == dep.PersonQualification.qualification_type_id,
        ).filter(
            dep.PersonQualification.unit_id == person.unit_id,
            dep.PersonQualification.person_id == person.id,
        ).all()
        return self._evaluate(person, on_date, qualifications)

    def get_capability_matrix(
        self, people: list[Any], dates: list[date]
    ) -> dict[tuple[int, date], OperationalCapability]:
        """Evaluate a roster with one qualification query, avoiding cell N+1s."""
        persisted = [person for person in people if person.id is not None]
        if not persisted or not dates:
            return {}
        person_ids = [person.id for person in persisted]
        rows = self.dependencies.db.session.query(
            self.dependencies.PersonQualification,
            self.dependencies.QualificationType,
        ).join(
            self.dependencies.QualificationType,
            self.dependencies.QualificationType.id
            == self.dependencies.PersonQualification.qualification_type_id,
        ).filter(
            self.dependencies.PersonQualification.person_id.in_(person_ids),
        ).all()
        by_person: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
        for record, qualification_type in rows:
            by_person[record.person_id].append((record, qualification_type))
        return {
            (person.id, on_date): self._evaluate(
                person, on_date, by_person.get(person.id, ())
            )
            for person in persisted
            for on_date in dates
        }

    def _evaluate(
        self, person: Any, on_date: date, qualifications: Any
    ) -> OperationalCapability:
        reasons: list[str] = []
        in_unit = self._in_unit(person, on_date)
        if not in_unit:
            reasons.append("Outside effective unit or employment dates.")
        roster_active = bool(
            person.is_operational
            and person.membership_status in {"active", "no_login"}
        )
        if not roster_active:
            reasons.append("Operational roster status is inactive.")

        valid_codes: set[str] = set()
        supervised: set[str] = set()
        for record, qtype in qualifications:
            code = (qtype.code or "").upper()
            if self._qualification_valid(record, on_date):
                valid_codes.add(code)
            elif code in INDEPENDENT_UE_CODES and record.status == "suspended":
                supervised.add(code)

        medical_valid = "MEDICAL" in valid_codes
        if not qualifications and person.medical_expiry:
            medical_valid = person.medical_expiry >= on_date
        if not medical_valid:
            reasons.append("Medical is not valid on this date.")
        independent = valid_codes & INDEPENDENT_UE_CODES
        if not independent:
            # Transitional compatibility for airports not yet migrated to
            # authoritative qualification records.
            legacy = {
                "ADI": person.tower_ue_expiry,
                "APS": person.radar_ue_expiry,
            }
            independent = {
                code for code, expiry in legacy.items()
                if expiry is not None and expiry >= on_date
            }
        if not independent:
            reasons.append("No independent unit endorsement is valid.")
        counts = bool(in_unit and roster_active and medical_valid and independent)
        return OperationalCapability(
            in_unit=in_unit,
            roster_active=roster_active,
            medically_valid=medical_valid,
            independent_competencies=frozenset(independent),
            supervised_competencies=frozenset(supervised),
            counts_as_operational=counts,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _in_unit(person: Any, on_date: date) -> bool:
        starts = (
            getattr(person, "employment_start_date", None),
            getattr(person, "unit_join_date", None),
            getattr(person, "roster_start_date", None),
        )
        ends = (
            getattr(person, "employment_end_date", None),
            getattr(person, "final_unit_date", None),
            getattr(person, "final_operational_duty_date", None),
        )
        return not any(value and on_date < value for value in starts) and not any(
            value and on_date > value for value in ends
        )

    @staticmethod
    def _qualification_valid(record: Any, on_date: date) -> bool:
        if record.status != "valid":
            return False
        if record.valid_from and on_date < record.valid_from:
            return False
        valid_to = getattr(record, "valid_to", None) or record.expires_on
        if valid_to and on_date > valid_to:
            return False
        suspended_from = getattr(record, "suspended_from", None)
        suspended_to = getattr(record, "suspended_to", None)
        if suspended_from and on_date >= suspended_from and (
            suspended_to is None or on_date <= suspended_to
        ):
            return False
        return True
