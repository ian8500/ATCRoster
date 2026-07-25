"""Transactional membership-limit enforcement."""
from sqlalchemy import func, select

COUNTED_MEMBERSHIP_STATES = ("active",)


def lock_unit_capacity(db, Unit, UnitMembership, unit_id: int):
    """Lock a unit and return its active count for invite/activation flows."""
    unit = db.session.execute(
        select(Unit).where(Unit.id == int(unit_id)).with_for_update()
    ).scalar_one()
    active = db.session.execute(
        select(func.count(UnitMembership.id)).where(
            UnitMembership.unit_id == unit.id,
            UnitMembership.status.in_(COUNTED_MEMBERSHIP_STATES),
        )
    ).scalar_one()
    if active >= unit.active_user_limit:
        raise ValueError(
            "Active account limit reached; deactivate an account first"
        )
    return unit, active


def activate_membership(db, Unit, UnitMembership, membership_id: int):
    """Activate while holding a unit row lock to prevent concurrent overrun."""
    membership = db.session.get(UnitMembership, membership_id)
    if not membership:
        raise LookupError("Membership not found")
    if membership.status != "active":
        lock_unit_capacity(
            db, Unit, UnitMembership, membership.unit_id
        )
    membership.status = "active"
    return membership
