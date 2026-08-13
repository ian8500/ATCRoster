"""Small, provider-neutral SMS boundary and ClickSend implementation."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import request as urllib_request


@dataclass(frozen=True)
class SmsResult:
    accepted: bool
    provider: str = "clicksend"
    provider_message_id: str = ""
    status: str = "failed"
    error_code: str = ""
    error_message: str = ""

    def legacy(self) -> tuple[bool, str]:
        """Compatibility adapter for the existing workflow boundary."""
        return self.accepted, self.provider_message_id or self.error_message


def clicksend_credentials() -> tuple[str, str, str]:
    return (
        os.getenv("CLICK_SEND_USERNAME", "").strip(),
        os.getenv("CLICK_SEND_API_KEY", ""),
        os.getenv("CLICK_SEND_DEFAULT_SENDER", "").strip(),
    )


def normalise_sms_number(value: str | None) -> str:
    """Return E.164; accept UK domestic mobile input but not malformed values."""
    candidate = re.sub(r"[\s().-]+", "", value or "")
    if candidate.startswith("0044"):
        candidate = "+44" + candidate[4:]
    elif candidate.startswith("07") and len(candidate) == 11:
        candidate = "+44" + candidate[1:]
    return candidate if re.fullmatch(r"\+[1-9]\d{7,14}", candidate) else ""


def normalise_uk_mobile(value: str | None) -> str:
    candidate = normalise_sms_number(value)
    return candidate if re.fullmatch(r"\+447\d{9}", candidate) else ""


def parse_sms_number_lines(raw: str) -> tuple[list[dict[str, str]], list[str]]:
    result: list[dict[str, str]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate((raw or "").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        label, separator, number_value = line.partition("|")
        if not separator:
            number_value, label = label, ""
        number = normalise_sms_number(number_value)
        if not number:
            errors.append(f"line {line_number}")
        elif number not in seen:
            seen.add(number)
            result.append({"number": number, "label": label.strip()[:80] or number})
    return result, errors


class ClickSendSmsProvider:
    """ClickSend REST v3 sender. It deliberately performs no automatic retry."""

    endpoint = "https://rest.clicksend.com/v3/sms/send"

    def send(self, *, to: str, body: str, sender: str | None = None) -> SmsResult:
        username, api_key, default_sender = clicksend_credentials()
        recipient = normalise_sms_number(to)
        resolved_sender = normalise_uk_mobile(sender or default_sender)
        if not (username and api_key and resolved_sender):
            return SmsResult(False, error_code="not_configured", error_message="ClickSend credentials or verified sender are not configured.")
        if not recipient:
            return SmsResult(False, error_code="invalid_recipient", error_message="Missing or invalid destination number.")
        if not body.strip():
            return SmsResult(False, error_code="invalid_message", error_message="SMS message cannot be empty.")
        payload = json.dumps({"messages": [{"to": recipient, "body": body, "from": resolved_sender}]}).encode("utf-8")
        req = urllib_request.Request(self.endpoint, data=payload, method="POST")
        token = base64.b64encode(f"{username}:{api_key}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        try:
            with urllib_request.urlopen(req, timeout=10) as response:  # nosec B310 -- fixed ClickSend HTTPS origin
                return self._parse_response(response.read())
        except urllib_error.HTTPError as exc:
            return self._http_error(exc.code, self._safe_error_body(exc))
        except urllib_error.URLError:
            return SmsResult(False, error_code="connection_error", error_message="Could not connect to ClickSend.")
        except TimeoutError:
            return SmsResult(False, error_code="timeout", error_message="ClickSend timed out; the message was not retried.")
        except Exception:
            return SmsResult(False, error_code="provider_error", error_message="ClickSend returned an unexpected error.")

    @staticmethod
    def _safe_error_body(exc: urllib_error.HTTPError) -> str:
        try:
            data = json.loads(exc.read().decode("utf-8") or "{}")
            return str(data.get("response_msg") or data.get("message") or "")[:240]
        except Exception:
            return ""

    @staticmethod
    def _http_error(code: int, detail: str) -> SmsResult:
        messages = {401: "ClickSend credentials were rejected.", 403: "ClickSend denied this SMS request.", 429: "ClickSend is rate limiting SMS requests."}
        if code >= 500:
            message = "ClickSend is temporarily unavailable; the message was not retried."
        else:
            message = messages.get(code, detail or "ClickSend rejected the SMS request.")
        return SmsResult(False, error_code=f"http_{code}", error_message=message)

    @staticmethod
    def _parse_response(raw: bytes) -> SmsResult:
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            item = (payload.get("data") or {}).get("messages", [{}])[0]
            status = str(item.get("status") or payload.get("response_code") or "submitted").lower()
            message_id = str(item.get("message_id") or item.get("id") or "")
            if str(payload.get("response_code", "SUCCESS")).upper() != "SUCCESS" or status in {"failed", "error", "rejected"}:
                return SmsResult(False, provider_message_id=message_id, status=status, error_code=str(item.get("status_code") or "rejected"), error_message=str(item.get("status_text") or payload.get("response_msg") or "ClickSend rejected the SMS."))
            return SmsResult(True, provider_message_id=message_id, status=status or "submitted")
        except (ValueError, TypeError, KeyError, IndexError):
            return SmsResult(False, error_code="malformed_response", error_message="ClickSend returned an invalid response.")


def send_via_clicksend(to_number: str, body: str, from_number: str | None = None) -> tuple[bool, str]:
    return ClickSendSmsProvider().send(to=to_number, body=body, sender=from_number).legacy()
