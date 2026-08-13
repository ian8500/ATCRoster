"""Persistence of successful SMS delivery audit records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .sms import normalise_sms_number


@dataclass(frozen=True)
class SmsAuditService:
    db: Any
    SmsAudit: Any
    current_unit_id: Callable[[], int]
    current_user: Callable[[], Any]

    def record(
        self, *, sender_number: str, recipient_number: str, recipient_label: str,
        body: str, message_type: str, provider_message_id: str,
        delivery_status: str = "submitted",
    ) -> None:
        actor = self.current_user()
        self.db.session.add(self.SmsAudit(
            unit_id=self.current_unit_id(),
            sent_by_staff_id=actor.id,
            sent_by_name=actor.name,
            sender_number=normalise_sms_number(sender_number),
            recipient_number=normalise_sms_number(recipient_number),
            recipient_label=(recipient_label or recipient_number)[:120],
            message_type=(message_type or "unit")[:30],
            message_content=body,
            provider_message_id=(provider_message_id or "")[:64],
            provider="clicksend",
            delivery_status=delivery_status[:30],
        ))
        self.db.session.commit()
