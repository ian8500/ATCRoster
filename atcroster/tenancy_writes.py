"""Tenant stamping, audit immutability, and cache invalidation tracking."""

from __future__ import annotations

from typing import Any, Callable


def enforce_operational_writes(
    session: Any,
    *,
    operational_models: tuple[type, ...],
    append_only_models: tuple[type, ...],
    SmsAudit: type,
    inspect_record: Callable[[Any], Any],
    authenticated_unit_id: Callable[[], int],
) -> None:
    """Stamp tenant ownership and reject cross-tenant or audit mutations."""
    touched_units = session.info.setdefault("roster_cache_touched_units", set())
    for record in session.new | session.dirty | session.deleted:
        if isinstance(record, operational_models):
            unit_id = getattr(record, "unit_id", None)
            if unit_id:
                touched_units.add(int(unit_id))
    for record in session.dirty:
        if isinstance(record, append_only_models) and session.is_modified(
            record, include_collections=False
        ):
            if isinstance(record, SmsAudit):
                changed = {
                    attribute.key
                    for attribute in inspect_record(record).attrs
                    if attribute.history.has_changes()
                }
                if changed == {"delivery_status"}:
                    continue
            raise PermissionError("Audit evidence is append-only")
    for record in session.deleted:
        if isinstance(record, append_only_models):
            raise PermissionError("Audit evidence is append-only")
    try:
        unit_id = authenticated_unit_id()
    except RuntimeError:
        return
    for record in session.new:
        if isinstance(record, operational_models):
            supplied = getattr(record, "unit_id", None)
            if supplied not in (None, unit_id):
                raise PermissionError("Cross-unit writes are forbidden")
            record.unit_id = unit_id


def invalidate_touched_units(
    session: Any, invalidate_unit: Callable[[int], None]
) -> None:
    for unit_id in session.info.pop("roster_cache_touched_units", set()):
        invalidate_unit(unit_id)


def discard_touched_units(session: Any) -> None:
    session.info.pop("roster_cache_touched_units", None)
