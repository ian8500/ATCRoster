"""Authoritative qualification status checks."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def staff_has_qualification(
    staff: Any,
    qualification_code: str,
    duty_date: date,
    *,
    QualificationType: Any,
    PersonQualification: Any,
    authenticated_unit_id: Callable[[], int],
) -> bool:
    """Evaluate tenant-scoped competence at the requested duty date."""
    code = (qualification_code or "").strip().upper()
    if not code:
        return True
    unit_id = int(getattr(staff, "unit_id", 0) or 0)
    try:
        context_unit_id = authenticated_unit_id()
    except RuntimeError:
        return False
    if not unit_id or unit_id != context_unit_id:
        return False
    qualification_type = QualificationType.query.filter_by(
        unit_id=unit_id, code=code, is_active=True
    ).first()
    if not qualification_type:
        return False
    record = PersonQualification.query.filter_by(
        unit_id=unit_id,
        person_id=staff.id,
        qualification_type_id=qualification_type.id,
    ).first()
    if not record or record.status != "valid":
        return False
    if record.valid_from and record.valid_from > duty_date:
        return False
    if qualification_type.expiry_required:
        return bool(record.expires_on and record.expires_on >= duty_date)
    return not record.expires_on or record.expires_on >= duty_date


def staff_is_countable(
    staff: Any,
    duty_date: date,
    *,
    capability_for: Callable[[int, date], Any],
) -> bool:
    """Apply dated capability, retaining unsaved-profile preview compatibility."""
    if staff.id is None:
        medical_valid = bool(staff.medical_expiry and staff.medical_expiry >= duty_date)
        independent_endorsement = any(
            expiry and expiry >= duty_date
            for expiry in (staff.tower_ue_expiry, staff.radar_ue_expiry)
        )
        return bool(medical_valid and independent_endorsement)
    return capability_for(staff.id, duty_date).counts_as_operational
