"""Operational-position endorsement assurance reporting."""

from __future__ import annotations

from typing import Any, Callable


def has_valid_endorsement(
    person_id: int,
    position_id: int,
    on_day: Any,
    *,
    PositionEndorsement: Any,
) -> bool:
    record = PositionEndorsement.query.filter_by(
        person_id=person_id,
        position_id=position_id,
        status="valid",
    ).first()
    return bool(
        record
        and record.valid_from <= on_day
        and (record.valid_until is None or record.valid_until >= on_day)
    )


def monthly_position_assurance(
    year: int,
    month: int,
    *,
    Assignment: Any,
    OperationalPosition: Any,
    PositionRequirement: Any,
    month_range: Callable[..., tuple[Any, list[Any]]],
    valid_endorsement: Callable[[int, int, Any], bool],
) -> list[dict[str, Any]]:
    """Compare position requirements with endorsed rostered staff."""
    _, days = month_range(year, month)
    requirements = (
        PositionRequirement.query.filter(
            PositionRequirement.day >= days[0],
            PositionRequirement.day <= days[-1],
        )
        .order_by(PositionRequirement.day, PositionRequirement.shift_code)
        .all()
    )
    positions = {row.id: row for row in OperationalPosition.query.all()}
    rows = []
    for requirement in requirements:
        assignments = Assignment.query.filter_by(
            day=requirement.day, code=requirement.shift_code
        ).all()
        eligible_count = sum(
            1
            for assignment in assignments
            if valid_endorsement(
                assignment.staff_id,
                requirement.position_id,
                requirement.day,
            )
        )
        target = requirement.required_count + requirement.contingency_count
        rows.append(
            {
                "requirement": requirement,
                "position": positions.get(requirement.position_id),
                "eligible": eligible_count,
                "target": target,
                "shortfall": max(0, target - eligible_count),
            }
        )
    return rows
