"""Append-only operational change-log creation."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


class ChangeAuditService:
    """Bind append-only change audit persistence to the application models."""

    def __init__(
        self,
        *,
        db: Any,
        ChangeLog: Any,
        current_user: Callable[[], Any],
        now: Callable[[], Any],
    ) -> None:
        self.db = db
        self.ChangeLog = ChangeLog
        self.current_user = current_user
        self.now = now

    def record(
        self,
        entity_type: str,
        entity_id: int,
        field: str,
        old: Any,
        new: Any,
        note: str = "",
        context_day: date | None = None,
    ) -> None:
        record_change(
            db=self.db,
            ChangeLog=self.ChangeLog,
            user=self.current_user(),
            now=self.now,
            entity_type=entity_type,
            entity_id=entity_id,
            field=field,
            old=old,
            new=new,
            note=note,
            context_day=context_day,
        )


def context_month_for_date(value: date | None) -> str | None:
    return None if value is None else f"{value.year:04d}-{value.month:02d}"


def record_change(
    *, db: Any, ChangeLog: Any, user: Any, now: Callable[[], Any],
    entity_type: str, entity_id: int, field: str, old: Any, new: Any,
    note: str = "", context_day: date | None = None,
) -> None:
    """Persist one change-log entry, rolling back a failed append safely."""
    try:
        db.session.add(ChangeLog(
            when=now(), who_user_id=getattr(user, "id", None),
            entity_type=entity_type, entity_id=entity_id, field=field,
            old_value=str(old) if old is not None else None,
            new_value=str(new) if new is not None else None,
            context_month=context_month_for_date(context_day), note=note or "",
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()
