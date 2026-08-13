"""Authentication-session credential lookup."""

from __future__ import annotations

from typing import Any


def credential_for_auth_stamp(
    user: Any, PlatformMfaCredential: Any, MfaCredential: Any,
) -> Any:
    """Return the authoritative MFA credential for a logged-in principal."""
    if getattr(user, "role", "") == "superadmin":
        return PlatformMfaCredential.query.filter_by(identity_id=user.id).first()
    return MfaCredential.query.filter_by(person_id=user.id).first()
