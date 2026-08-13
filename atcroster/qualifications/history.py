"""Qualification history, legacy profile synchronization, and roster impacts."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable


def qualification_snapshot(record: Any) -> dict[str, Any]:
    return {
        "person_id": record.person_id,
        "qualification_type_id": record.qualification_type_id,
        "issued_on": record.issued_on,
        "valid_from": record.valid_from,
        "expires_on": record.expires_on,
        "status": record.status,
    }


def record_qualification_history(
    record: Any,
    action: str,
    *,
    db: Any,
    PersonQualificationHistory: Any,
    actor_id: int,
) -> None:
    db.session.add(
        PersonQualificationHistory(
            unit_id=record.unit_id,
            person_qualification_id=record.id,
            actor_id=actor_id,
            action=action,
            snapshot_json=json.dumps(
                qualification_snapshot(record), default=str, sort_keys=True
            ),
        )
    )


def sync_legacy_roster_profile(
    person: Any, qualification_type: Any, expires_on: date | None
) -> None:
    field = {
        "MEDICAL": "medical_expiry",
        "ADI": "tower_ue_expiry",
        "APS": "radar_ue_expiry",
        "MET": "met_ue_expiry",
    }.get(qualification_type.code)
    if field:
        setattr(person, field, expires_on)


def classify_qualification_impact(
    code: str,
    old_status: str | None,
    old_valid_from: date | None,
    old_expires_on: date | None,
    new_status: str | None,
    new_valid_from: date | None,
    new_expires_on: date | None,
    *,
    impact_types: Any,
    today: date,
) -> tuple[Any | None, date]:
    """Classify a qualification transition and its true effective date."""
    del old_valid_from
    code = (code or "").strip().upper()
    old_valid = old_status == "valid"
    new_valid = new_status == "valid"
    effective = new_valid_from or new_expires_on or old_expires_on or today
    if code == "MEDICAL":
        if new_valid and not old_valid:
            return impact_types.MEDICAL_RESTORED, effective
        if old_valid and not new_valid:
            return impact_types.MEDICAL_EXPIRED, effective
        if new_valid and new_expires_on != old_expires_on:
            return impact_types.MEDICAL_RESTORED, effective
    if code == "OJTI" and new_valid and not old_valid:
        return impact_types.OJTI_ACHIEVED, effective
    if code in {"ASSESSOR", "ASSR"} and new_valid and not old_valid:
        return impact_types.ASSESSOR_ACHIEVED, effective
    if code in {"ADI", "APS", "MET", "UE"}:
        if old_valid and new_status == "suspended":
            return impact_types.UE_SUSPENDED, effective
        if old_valid and not new_valid:
            return impact_types.UE_EXPIRED, effective
        if new_valid and not old_valid:
            return (
                impact_types.UE_RESTORED
                if old_status == "suspended"
                else impact_types.FIRST_UE_ACHIEVED
            ), effective
        if new_valid and new_expires_on != old_expires_on:
            return impact_types.ADDITIONAL_UE_ACHIEVED, effective
    return None, effective


def has_other_valid_ue(
    unit_id: int,
    person_id: int,
    excluded_type_id: int,
    on_date: date,
    *,
    db: Any,
    PersonQualification: Any,
    QualificationType: Any,
) -> bool:
    return (
        db.session.query(PersonQualification.id)
        .join(
            QualificationType,
            QualificationType.id == PersonQualification.qualification_type_id,
        )
        .filter(
            PersonQualification.unit_id == unit_id,
            PersonQualification.person_id == person_id,
            PersonQualification.qualification_type_id != excluded_type_id,
            PersonQualification.status == "valid",
            QualificationType.code.in_(("ADI", "APS", "MET", "UE")),
            db.or_(
                PersonQualification.valid_from.is_(None),
                PersonQualification.valid_from <= on_date,
            ),
            db.or_(
                PersonQualification.expires_on.is_(None),
                PersonQualification.expires_on >= on_date,
            ),
        )
        .first()
        is not None
    )


def record_roster_impact_for_qualification(
    person: Any,
    qualification_type: Any,
    old_status: str | None,
    old_valid_from: date | None,
    old_expires_on: date | None,
    record: Any,
    *,
    impact_types: Any,
    today: date,
    has_other_ue: Callable[[int, int, int, date], bool],
    record_roster_impact: Callable[..., Any],
    reason: str,
):
    impact_type, impact_date = classify_qualification_impact(
        qualification_type.code,
        old_status,
        old_valid_from,
        old_expires_on,
        record.status if record else None,
        record.valid_from if record else None,
        record.expires_on if record else None,
        impact_types=impact_types,
        today=today,
    )
    if impact_type == impact_types.FIRST_UE_ACHIEVED and has_other_ue(
        person.unit_id, person.id, qualification_type.id, impact_date
    ):
        impact_type = impact_types.ADDITIONAL_UE_ACHIEVED
    if impact_type:
        record_roster_impact(
            impact_type,
            impact_date,
            staff_ids=[person.id],
            rebuild_baseline=False,
            reason=reason,
        )
    return impact_type
