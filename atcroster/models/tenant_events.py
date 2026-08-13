"""SQLAlchemy tenant and roster-cache session event registration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sqlalchemy import event, inspect
from sqlalchemy.orm import with_loader_criteria


def register_tenant_session_events(
    session_class: Any,
    *,
    operational_models: Iterable[type],
    append_only_models: Iterable[type],
    SmsAudit: type,
    authenticated_unit_id: Callable[[], int],
    enforce_operational_writes: Callable[..., Any],
    invalidate_touched_units: Callable[..., Any],
    discard_touched_units: Callable[..., Any],
    invalidate_unit: Callable[[int], Any],
) -> None:
    """Apply tenant isolation and cache consistency to the canonical session."""
    scoped_models = tuple(operational_models)
    audit_models = tuple(append_only_models)

    @event.listens_for(session_class, "do_orm_execute")
    def scope_operational_selects(execute_state):
        if not execute_state.is_select or execute_state.execution_options.get(
            "skip_tenant_scope"
        ):
            return
        try:
            unit_id = authenticated_unit_id()
        except RuntimeError:
            return
        statement = execute_state.statement
        for model in scoped_models:
            statement = statement.options(
                with_loader_criteria(
                    model,
                    lambda cls: cls.unit_id == unit_id,
                    include_aliases=True,
                    track_closure_variables=True,
                )
            )
        execute_state.statement = statement

    @event.listens_for(session_class, "before_flush")
    def stamp_operational_writes(session_obj, _flush_context, _instances):
        return enforce_operational_writes(
            session_obj,
            operational_models=scoped_models,
            append_only_models=audit_models,
            SmsAudit=SmsAudit,
            inspect_record=inspect,
            authenticated_unit_id=authenticated_unit_id,
        )

    @event.listens_for(session_class, "after_commit")
    def invalidate_roster_cache_after_commit(session_obj):
        return invalidate_touched_units(session_obj, invalidate_unit)

    @event.listens_for(session_class, "after_rollback")
    def discard_roster_cache_invalidation_after_rollback(session_obj):
        return discard_touched_units(session_obj)
