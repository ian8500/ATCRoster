"""Plain-text account email delivery and address validation helpers."""

from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage
from typing import Any


def email_service_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM_ADDRESS"))


def valid_email(value: str) -> str:
    candidate = (value or "").strip().casefold()
    if len(candidate) > 254 or candidate.count("@") != 1:
        return ""
    local, domain = candidate.split("@")
    if not (local and len(local) <= 64 and re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+", local)):
        return ""
    labels = domain.split(".")
    if len(labels) < 2 or any(
        not label or len(label) > 63 or not re.fullmatch(r"[a-z0-9-]+", label)
        or label.startswith("-") or label.endswith("-")
        for label in labels
    ):
        return ""
    return candidate


def send_account_email(to_address: str, subject: str, body: str, logger: Any) -> bool:
    """Send plain text without exposing message contents in failure logs."""
    if not to_address or not email_service_configured():
        return False
    message = EmailMessage()
    message["To"] = to_address
    message["From"] = os.environ["SMTP_FROM_ADDRESS"]
    message["Subject"] = subject[:160]
    message.set_content(body)
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    username, password = os.getenv("SMTP_USERNAME", ""), os.getenv("SMTP_PASSWORD", "")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    try:
        with smtplib.SMTP(host, port, timeout=10) as connection:
            if use_tls:
                connection.starttls()
            if username:
                connection.login(username, password)
            connection.send_message(message)
        return True
    except Exception:
        logger.exception("account_email_delivery_failed")
        return False
