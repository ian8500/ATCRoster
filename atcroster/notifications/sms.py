"""Provider-neutral SMS normalization and ClickSend delivery helpers."""

from __future__ import annotations

import base64
import json
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request


def clicksend_credentials() -> tuple[str, str, str]:
    """Return ClickSend's API username, API key, and optional sender fallback."""
    return (
        os.getenv("CLICK_SEND_USERNAME", "") or os.getenv("CLICKSEND_USERNAME", ""),
        os.getenv("CLICK_SEND_API_KEY", "") or os.getenv("CLICKSEND_API_KEY", ""),
        os.getenv("CLICK_SEND_DEFAULT_SENDER", "")
        or os.getenv("CLICKSEND_FALLBACK_SENDER", ""),
    )


def normalise_sms_number(value: str | None) -> str:
    """Return an E.164 number while accepting harmless display punctuation."""
    candidate = re.sub(r"[\s().-]+", "", value or "")
    return candidate if re.fullmatch(r"\+[1-9]\d{7,14}", candidate) else ""


def normalise_uk_mobile(value: str | None) -> str:
    """Normalize a UK mobile sender to E.164."""
    candidate = re.sub(r"[\s().-]+", "", value or "")
    if candidate.startswith("0044"):
        candidate = "+44" + candidate[4:]
    elif candidate.startswith("44") and len(candidate) == 12:
        candidate = "+" + candidate
    elif candidate.startswith("07") and len(candidate) == 11:
        candidate = "+44" + candidate[1:]
    return candidate if re.fullmatch(r"\+447\d{9}", candidate) else ""


def parse_sms_number_lines(raw: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse one ``label | +number`` or plain E.164 number per line."""
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


def send_via_clicksend(
    to_number: str, body: str, from_number: str | None = None,
) -> tuple[bool, str]:
    """Send through ClickSend's fixed HTTPS endpoint using HTTP Basic auth."""
    username, api_key, fallback = clicksend_credentials()
    sender = normalise_sms_number(from_number or fallback)
    recipient = normalise_sms_number(to_number)
    if not (username and api_key and sender):
        return False, "ClickSend credentials or a sender number are not configured."
    if not recipient:
        return False, "Missing or invalid destination number."
    payload = json.dumps({"messages": [{
        "body": body, "from": sender, "to": recipient,
    }]}).encode("utf-8")
    request = urllib_request.Request(
        "https://rest.clicksend.com/v3/sms/send", data=payload, method="POST",
    )
    token = base64.b64encode(f"{username}:{api_key}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {token}")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib_request.urlopen(request, timeout=10) as response:  # nosec B310 -- fixed provider origin
            if not 200 <= response.status < 300:
                return False, f"ClickSend returned HTTP {response.status}."
            raw = response.read(64 * 1024).decode("utf-8")
            parsed = json.loads(raw or "{}")
            if str(parsed.get("response_code") or "").upper() != "SUCCESS":
                return False, str(parsed.get("response_msg") or "ClickSend rejected the message.")[:300]
            messages = ((parsed.get("data") or {}).get("messages") or [])
            message = messages[0] if messages else {}
            return True, str(message.get("message_id") or "submitted")
    except urllib_error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8")[:300]
        except Exception:
            detail = str(error)
        return False, f"{error.code}: {detail}"
    except urllib_error.URLError as error:
        return False, str(getattr(error, "reason", error))
    except Exception as error:
        return False, str(error)


# Compatibility import surface for integrations that imported the historical helper.
messagemedia_credentials = clicksend_credentials
send_via_messagemedia = send_via_clicksend
