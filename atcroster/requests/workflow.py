"""Persistence helpers for the shift-request workflow."""

from __future__ import annotations

import json
from typing import Any, Callable


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
