"""Platform-identity MFA login completion helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Callable


def pending_platform_login(
    session: Any, *, db: Any, PlatformIdentity: Any
) -> tuple[Any | None, Any | None]:
    identity_id = int(session.get("_platform_mfa_identity_id") or 0)
    user_id = int(session.get("_platform_mfa_user_id") or 0)
    if not identity_id or user_id != identity_id:
        return None, None
    identity = db.session.get(PlatformIdentity, identity_id)
    if not identity or identity.role != "superadmin":
        return None, None
    return identity, identity


def complete_platform_login(
    identity: Any,
    user: Any,
    *,
    recovery_used: bool,
    session: Any,
    db: Any,
    login_user: Callable[[Any], None],
    initialize_session: Callable[..., None],
    now: Callable[[], Any],
    security_event: Callable[..., None],
    login_redirect: Callable[..., str],
    redirect: Callable[[str], Any],
):
    """Establish an authenticated platform session after successful MFA."""
    next_url = session.get("_platform_mfa_next", "")
    session.clear()
    login_user(user)
    initialize_session(user, platform_mfa=True)
    identity.last_active_at = now()
    security_event(
        "platform_recovery_code_used" if recovery_used else "platform_mfa_verified",
        "success",
        identity.id,
        hashlib.sha256(identity.username.lower().encode()).hexdigest()[:16],
    )
    db.session.commit()
    return redirect(
        login_redirect(
            next_url,
            default_endpoint="platform_admin",
            user_id=user.id,
        )
    )
