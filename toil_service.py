"""Transactional, idempotent TOIL ledger operations."""

from __future__ import annotations

import secrets
from typing import Any, Callable


class ToilValidationError(ValueError):
    pass


def apply_toil_transaction(
    db: Any,
    Staff: Any,
    ToilTransaction: Any,
    *,
    unit_id: int,
    person_id: int,
    delta_half_days: int,
    reason: str,
    actor_id: int,
    utcnow: Callable[[], Any],
    transaction_key: str | None = None,
    source_type: str = "manual",
    source_id: int | None = None,
) -> Any:
    if not isinstance(delta_half_days, int) or delta_half_days == 0:
        raise ToilValidationError("TOIL adjustment must be a non-zero half-day value.")
    if abs(delta_half_days) > 400:
        raise ToilValidationError("TOIL adjustment exceeds the supported limit.")
    safe_reason = (reason or "").strip()[:500]
    if not safe_reason:
        raise ToilValidationError("A TOIL adjustment reason is required.")
    key = (transaction_key or secrets.token_hex(24)).strip()[:64]
    if not key:
        raise ToilValidationError("A TOIL transaction key is required.")
    person = (
        Staff.query.filter_by(id=person_id, unit_id=unit_id).with_for_update().first()
    )
    if person is None:
        raise ToilValidationError("Unknown staff member for this airport.")
    existing = ToilTransaction.query.filter_by(
        unit_id=unit_id, transaction_key=key
    ).first()
    if existing:
        return existing
    new_balance = int(person.toil_half_days or 0) + delta_half_days
    person.toil_half_days = new_balance
    row = ToilTransaction(
        unit_id=unit_id,
        person_id=person_id,
        delta_half_days=delta_half_days,
        balance_after_half_days=new_balance,
        reason=safe_reason,
        source_type=(source_type or "manual").strip()[:40],
        source_id=source_id,
        actor_id=actor_id,
        transaction_key=key,
        occurred_at=utcnow(),
    )
    db.session.add(row)
    db.session.flush()
    return row
