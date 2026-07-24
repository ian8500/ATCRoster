"""Transactional membership-limit enforcement."""
from sqlalchemy import select, func


COUNTED_MEMBERSHIP_STATES = ("active",)


def activate_membership(db, Unit, UnitMembership, membership_id: int):
    """Activate while holding a unit row lock to prevent concurrent overrun."""
    membership = db.session.get(UnitMembership, membership_id)
    if not membership:
        raise LookupError("Membership not found")
    unit = db.session.execute(
        select(Unit).where(Unit.id == membership.unit_id).with_for_update()
    ).scalar_one()
    active = db.session.execute(
        select(func.count(UnitMembership.id)).where(
            UnitMembership.unit_id == unit.id,
            UnitMembership.status.in_(COUNTED_MEMBERSHIP_STATES),
        )
    ).scalar_one()
    if membership.status != "active" and active >= unit.active_user_limit:
        raise ValueError("Active account limit reached; deactivate an account first")
    membership.status = "active"
    return membership
