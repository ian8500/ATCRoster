"""Recovery and support email-recipient resolution."""

from __future__ import annotations

from typing import Any, Callable


def platform_support_emails(
    PlatformIdentity: Any, configured_support_email: str, valid_email: Callable[[str], str],
) -> list[str]:
    configured = valid_email(configured_support_email)
    rows = PlatformIdentity.query.filter(
        PlatformIdentity.public_id.like("platform-%"), PlatformIdentity.email != "",
    ).all()
    return list(dict.fromkeys(
        address for address in [configured, *(row.email for row in rows)] if address
    ))


def unit_admin_emails(
    db: Any, PlatformIdentity: Any, UnitMembership: Any, unit_id: int,
) -> list[str]:
    rows = db.session.query(PlatformIdentity).join(
        UnitMembership, UnitMembership.identity_id == PlatformIdentity.id,
    ).filter(
        UnitMembership.unit_id == unit_id,
        UnitMembership.status == "active",
        UnitMembership.role == "UnitAdmin",
        PlatformIdentity.email != "",
    ).all()
    return list(dict.fromkeys(row.email for row in rows if row.email))
