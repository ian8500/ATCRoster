"""Provider-neutral SMS normalization and MessageMedia delivery helpers."""

from __future__ import annotations

import base64
import json
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request


def messagemedia_credentials() -> tuple[str, str, str]:
    return (
        os.getenv("MESSAGEMEDIA_API_KEY", ""),
        os.getenv("MESSAGEMEDIA_API_SECRET", ""),
        os.getenv("MESSAGEMEDIA_FALLBACK_SENDER", ""),
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


def send_via_messagemedia(
    to_number: str, body: str, from_number: str | None = None,
) -> tuple[bool, str]:
    """Send through Sinch MessageMedia's fixed documented HTTPS endpoint."""
    api_key, api_secret, fallback = messagemedia_credentials()
    sender = normalise_sms_number(from_number or fallback)
    recipient = normalise_sms_number(to_number)
    if not (api_key and api_secret and sender):
        return False, "Sinch MessageMedia credentials or fallback sender are not configured."
    if not recipient:
        return False, "Missing or invalid destination number."
    payload = json.dumps({"messages": [{
        "content": body, "source_number": sender,
        "destination_number": recipient, "delivery_report": True,
    }]}).encode("utf-8")
    request = urllib_request.Request(
        "https://api.messagemedia.com/v1/messages", data=payload, method="POST",
    )
    token = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {token}")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib_request.urlopen(request, timeout=10) as response:  # nosec B310 -- fixed provider origin
            parsed = json.loads(response.read().decode("utf-8") or "{}")
            message = (parsed.get("messages") or [{}])[0]
            return True, str(message.get("message_id") or message.get("id") or "submitted")
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
