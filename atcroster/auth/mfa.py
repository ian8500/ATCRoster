"""TOTP primitives shared by platform and airport-user MFA workflows."""

from __future__ import annotations

import base64
import io
import secrets
from datetime import timedelta
from typing import Any, Callable

import pyotp


def decrypt_secret(credential: Any, decrypt: Callable[[str], str]) -> str:
    """Decrypt a credential secret while preserving a safe operational error."""
    try:
        return decrypt(credential.encrypted_secret)
    except ValueError as exc:
        raise RuntimeError("MFA credential cannot be decrypted.") from exc


def matching_totp_step(
    secret: str, code: str, now: Callable[[], Any],
) -> int | None:
    """Accept the current adjacent TOTP windows and return the consumed step."""
    totp = pyotp.TOTP(secret)
    current_time = now()
    for offset in (-1, 0, 1):
        candidate_time = current_time + timedelta(seconds=offset * 30)
        if secrets.compare_digest(totp.at(candidate_time), code):
            return int(candidate_time.timestamp() // 30)
    return None


def totp_qr_data_uri(provisioning_uri: str) -> str:
    """Render a QR SVG locally so MFA secrets never leave the application."""
    import qrcode
    import qrcode.image.svg

    qr_buffer = io.BytesIO()
    qrcode.make(
        provisioning_uri,
        image_factory=qrcode.image.svg.SvgPathImage,
        box_size=8,
        border=4,
    ).save(qr_buffer)
    return "data:image/svg+xml;base64," + base64.b64encode(qr_buffer.getvalue()).decode("ascii")
