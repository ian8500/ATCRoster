"""Overtime SMS recipient orchestration."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional


class OvertimeSmsService:
    def __init__(self, configuration: Any, audit: Any, send: Callable[..., tuple[bool, str]]):
        self.configuration = configuration
        self.audit = audit
        self.send = send

    def notify(self, staff_list: list[Any], message: str) -> tuple[int, list[tuple[Optional[Any], str]]]:
        sender_options = self.configuration.sender_options()
        from_number = self.configuration.default_number("sms_default_sender", sender_options)
        if not (self.configuration.service_configured() and from_number):
            return 0, [(None, "SMS sending is not configured.")]
        sent = 0
        failures: list[tuple[Optional[Any], str]] = []
        for staff in staff_list:
            if not (staff and staff.phone_number):
                failures.append((staff, "No phone number on file."))
                continue
            ok, detail = self.send(staff.phone_number, message, from_number)
            if ok:
                self.audit.record(
                    sender_number=from_number, recipient_number=staff.phone_number,
                    recipient_label=staff.name, body=message, message_type="overtime",
                    provider_message_id=detail,
                )
                sent += 1
            else:
                failures.append((staff, detail))
        return sent, failures


def default_overtime_sms_body(chosen_date: date | None, shift_code: str | None) -> str:
    if not (chosen_date and shift_code):
        return ""
    return f"Overtime available on {chosen_date.isoformat()} for {shift_code} shift. Please reply if interested."
