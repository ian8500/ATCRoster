"""Transactional locking for mutable roster months."""

from __future__ import annotations

from typing import Any, Callable


def lock_roster_month(
    db: Any, Requirement: Any, unit_id: int, year: int, month: int,
    ensure_month_requirement: Callable[[int, int], Any],
) -> Any:
    """Lock (or create then lock) the requirement row for a roster month."""
    requirement = Requirement.query.filter_by(
        unit_id=unit_id, year=year, month=month,
    ).with_for_update().first()
    if requirement is None:
        ensure_month_requirement(year, month)
        db.session.flush()
        requirement = Requirement.query.filter_by(
            unit_id=unit_id, year=year, month=month,
        ).with_for_update().one()
    return requirement
