"""Roster staffing-requirement persistence."""

from __future__ import annotations

from typing import Any


def ensure_month_requirement(db: Any, Requirement: Any, year: int, month: int, default: tuple[int, ...] = (4, 4, 4, 2)) -> Any:
    """Load or create the monthly staffing requirement row."""
    requirement = Requirement.query.filter_by(year=year, month=month).first()
    if requirement is not None:
        return requirement
    if len(default) == 3:
        morning, afternoon, night = default
        day = 0
    else:
        morning, day, afternoon, night = default
    requirement = Requirement(
        year=year, month=month, req_m=morning, req_d=day, req_a=afternoon, req_n=night,
        req_sat_m=morning, req_sat_d=day, req_sat_a=afternoon, req_sat_n=night,
        req_sun_m=morning, req_sun_d=day, req_sun_a=afternoon, req_sun_n=night,
    )
    db.session.add(requirement)
    db.session.commit()
    return requirement
