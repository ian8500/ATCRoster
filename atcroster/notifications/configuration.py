"""Unit-scoped SMS configuration selection without database ownership."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .sms import messagemedia_credentials, normalise_sms_number, normalise_uk_mobile


def validate_sms_settings(
    senders: list[dict[str, str]], destinations: list[dict[str, str]],
    sender_errors: list[str], destination_errors: list[str],
    default_sender: str, default_destination: str,
) -> str | None:
    """Return the user-safe validation error for unit SMS configuration."""
    if sender_errors or destination_errors:
        invalid = ", ".join(
            [f"sender {item}" for item in sender_errors]
            + [f"destination {item}" for item in destination_errors]
        )
        return f"Use international numbers such as +447700900123. Check {invalid}."
    if default_sender and default_sender not in {item["number"] for item in senders}:
        return "The default sender must be in the sender list."
    if default_destination and default_destination not in {item["number"] for item in destinations}:
        return "The default operational number must be in its list."
    return None


@dataclass(frozen=True)
class SmsConfigurationService:
    settings_snapshot: Callable[[int], dict[str, str]]
    current_unit_id: Callable[[], int]

    def _unit_id(self, unit_id: int | None) -> int:
        return int(unit_id or self.current_unit_id() or 1)

    def number_options(self, key: str, unit_id: int | None = None) -> list[dict[str, str]]:
        raw = self.settings_snapshot(self._unit_id(unit_id)).get(key, "[]")
        try:
            values = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in values if isinstance(values, list) else []:
            if not isinstance(item, dict):
                continue
            number = normalise_sms_number(item.get("number"))
            if number and number not in seen:
                seen.add(number)
                result.append({"number": number, "label": str(item.get("label") or number).strip()[:80] or number})
        return result

    def sender_options(self, unit_id: int | None = None) -> list[dict[str, str]]:
        configured = self.number_options("sms_sender_numbers", unit_id)
        fallback = normalise_uk_mobile(messagemedia_credentials()[2])
        if fallback and fallback not in {item["number"] for item in configured}:
            configured.append({"number": fallback, "label": "Unit fallback sender"})
        return configured

    def operational_options(self, unit_id: int | None = None) -> list[dict[str, str]]:
        return self.number_options("sms_operational_numbers", unit_id)

    def default_number(self, setting_key: str, options: list[dict[str, str]], unit_id: int | None = None) -> str:
        configured = normalise_sms_number(
            self.settings_snapshot(self._unit_id(unit_id)).get(setting_key)
        )
        allowed = {item["number"] for item in options}
        return configured if configured in allowed else (options[0]["number"] if options else "")

    def service_configured(self) -> bool:
        key, secret, fallback = messagemedia_credentials()
        return bool(key and secret and normalise_sms_number(fallback))
