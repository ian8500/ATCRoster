"""Persistence helpers for the shift-request workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable


def clamp_request_navigation(
    year: int, month: int, minimum_month: date
) -> tuple[str | None, str]:
    """Return safe adjacent month links for request administration."""
    previous_year, previous_month = (year - 1, 12) if month == 1 else (year, month - 1)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    previous_allowed = date(previous_year, previous_month, 1) >= minimum_month.replace(day=1)
    return (
        f"{previous_year}-{previous_month:02d}" if previous_allowed else None,
        f"{next_year}-{next_month:02d}",
    )


def load_unit_request_rules(
    unit_id: int | None,
    *,
    db: Any,
    Unit: Any,
    current_unit_id: Callable[[], int],
    normalise_rules: Callable[[Any, Any], tuple[int, int]],
) -> tuple[int, int]:
    """Load and normalize one airport's request horizon and lock day."""
    unit = db.session.get(Unit, unit_id or current_unit_id())
    return normalise_rules(
        getattr(unit, "request_months_ahead", 3),
        getattr(unit, "request_lock_day", 20),
    )


def add_request_audit(
    request_record: Any,
    actor_id: int,
    transition: str,
    old_value: object,
    new_value: object,
    reason: str = "",
    *,
    db: Any,
    RequestAudit: Any,
) -> None:
    """Append a tenant-scoped request transition audit record."""
    db.session.add(
        RequestAudit(
            unit_id=request_record.unit_id,
            request_id=request_record.id,
            actor_id=actor_id,
            transition=transition,
            old_value=json.dumps(old_value, default=str, sort_keys=True),
            new_value=json.dumps(new_value, default=str, sort_keys=True),
            reason=(reason or "")[:500],
        )
    )


def add_requester_notification(
    request_record: Any, *, db: Any, Notification: Any
) -> None:
    """Queue the established user-facing notification for a request state."""
    if request_record.status not in {
        "pending",
        "approved",
        "rejected",
        "fulfilled",
    }:
        return
    if request_record.status == "fulfilled":
        outcome = "was approved and added to the roster"
    elif request_record.status == "rejected":
        outcome = "was refused"
    else:
        outcome = f"is now {request_record.status}"
    comment = (request_record.admin_response or "").strip()
    comment_text = f" Manager comment: {comment}" if comment else ""
    db.session.add(
        Notification(
            unit_id=request_record.unit_id,
            recipient_id=request_record.staff_id,
            kind=f"shift_request_{request_record.status}",
            message=(
                f"Your {request_record.code} shift request for "
                f"{request_record.day.strftime('%d %B %Y')} "
                f"{outcome}.{comment_text}"
            ),
        )
    )


@dataclass(frozen=True)
class RequestWorkflowDependencies:
    db: Any
    Unit: Any
    RequestAudit: Any
    Notification: Any
    current_unit_id: Callable[[], int]
    normalise_rules: Callable[[Any, Any], tuple[int, int]]
    lock_date: Callable[[int, int, int], Any]
    month_is_locked: Callable[[int, int, int, Any], bool]
    add_months: Callable[[Any, int], Any]
    date_bounds: Callable[[Any, int], tuple[Any, Any]]
    safe_admin_month: Callable[[str | None, Any], str]


class RequestWorkflowService:
    """Own request-window policy, audit, and requester notifications."""

    def __init__(self, dependencies: RequestWorkflowDependencies):
        self.dependencies = dependencies

    def unit_rules(self, unit_id: int | None = None) -> tuple[int, int]:
        deps = self.dependencies
        return load_unit_request_rules(
            unit_id,
            db=deps.db,
            Unit=deps.Unit,
            current_unit_id=deps.current_unit_id,
            normalise_rules=deps.normalise_rules,
        )

    def lock_date_for_month(self, year: int, month: int, unit_id: int | None = None):
        _, lock_day = self.unit_rules(unit_id)
        return self.dependencies.lock_date(year, month, lock_day)

    def is_month_locked(
        self,
        year: int,
        month: int,
        today: Any = None,
        unit_id: int | None = None,
    ) -> bool:
        _, lock_day = self.unit_rules(unit_id)
        return self.dependencies.month_is_locked(year, month, lock_day, today)

    def add_months(self, first: Any, count: int):
        return self.dependencies.add_months(first, count)

    def request_date_bounds(self, today: Any, unit_id: int) -> tuple[Any, Any]:
        months, _ = self.unit_rules(unit_id)
        return self.dependencies.date_bounds(today, months)

    def add_audit(
        self,
        request_record: Any,
        actor_id: int,
        transition: str,
        old_value: object,
        new_value: object,
        reason: str = "",
    ) -> None:
        deps = self.dependencies
        return add_request_audit(
            request_record,
            actor_id,
            transition,
            old_value,
            new_value,
            reason,
            db=deps.db,
            RequestAudit=deps.RequestAudit,
        )

    def notify_requester(self, request_record: Any) -> None:
        deps = self.dependencies
        return add_requester_notification(
            request_record,
            db=deps.db,
            Notification=deps.Notification,
        )

    def safe_admin_month(self, raw_value: str | None, fallback: Any) -> str:
        return self.dependencies.safe_admin_month(raw_value, fallback)
