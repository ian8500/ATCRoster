"""Roster-impact horizon and cache invalidation support."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def generated_horizon_end(
    unit_id: int,
    effective_from: date,
    *,
    db: Any,
    Assignment: Any,
) -> date | None:
    return (
        db.session.query(db.func.max(Assignment.day))
        .filter(
            Assignment.unit_id == unit_id,
            Assignment.day >= effective_from,
        )
        .scalar()
    )


def invalidate_impact_months(
    unit_id: int,
    effective_from: date,
    effective_to: date,
    *,
    cache: Any,
    cached_loader: Callable[..., Any],
    add_months: Callable[[int, int, int], tuple[int, int]],
) -> None:
    """Invalidate every monthly roster cache entry touched by an impact."""
    cursor = effective_from.replace(day=1)
    final = effective_to.replace(day=1)
    while cursor <= final:
        if cache:
            try:
                cache.delete_memoized(
                    cached_loader, int(unit_id), cursor.year, cursor.month
                )
            except Exception:
                pass
        next_year, next_month = add_months(cursor.year, cursor.month, 1)
        cursor = date(next_year, next_month, 1)
