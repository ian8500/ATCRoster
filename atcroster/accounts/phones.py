"""Workforce contact-number normalization."""

from __future__ import annotations

import re


def normalise_phone_number(value: str | None) -> str:
    """Keep only digits and a leading plus, converting international 00."""
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9+]+", "", value.strip())
    if cleaned.startswith("00") and not cleaned.startswith("000"):
        return "+" + cleaned[2:]
    return cleaned
