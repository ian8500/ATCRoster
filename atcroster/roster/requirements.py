"""Roster staffing-requirement persistence."""

from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Any, Callable


def ensure_month_requirement(
    db: Any,
    Requirement: Any,
    year: int,
    month: int,
    default: tuple[int, ...] = (4, 4, 4, 2),
) -> Any:
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
        year=year,
        month=month,
        req_m=morning,
        req_d=day,
        req_a=afternoon,
        req_n=night,
        req_sat_m=morning,
        req_sat_d=day,
        req_sat_a=afternoon,
        req_sat_n=night,
        req_sun_m=morning,
        req_sun_d=day,
        req_sun_a=afternoon,
        req_sun_n=night,
    )
    db.session.add(requirement)
    db.session.commit()
    return requirement


def requirements_for_day(
    requirement: Any,
    day: date,
    special: Any,
    daily_requirements: Callable[[Any, date, Any], dict[str, int]],
) -> dict[str, int]:
    """Resolve standard and special staffing requirements for a calendar day."""
    return daily_requirements(requirement, day, special)


def save_monthly_requirements(
    form: Any,
    *,
    db: Any,
    Requirement: Any,
    impact_type: Any,
    record_roster_impact: Callable[..., None],
) -> None:
    """Persist the monthly staffing grid and its bounded impact event."""
    periods = form.getlist("ym")
    fields = [
        f"req_{prefix}{code}"
        for prefix in ("", "sat_", "sun_")
        for code in ("m", "d", "a", "n")
    ]
    values = {field: form.getlist(field) for field in fields}
    for index, period in enumerate(periods):
        year, month = [int(value) for value in period.split("-")]
        row = Requirement.query.filter_by(year=year, month=month).first()
        if not row:
            row = Requirement(year=year, month=month)
            db.session.add(row)
        for field in fields:
            try:
                value = int(values[field][index] or 0)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid staffing value for {field}.") from exc
            setattr(row, field, max(0, value))
    if periods:
        ordered = sorted(
            (int(value.split("-")[0]), int(value.split("-")[1])) for value in periods
        )
        start_year, start_month = ordered[0]
        end_year, end_month = ordered[-1]
        record_roster_impact(
            impact_type,
            date(start_year, start_month, 1),
            effective_to=date(end_year, end_month, monthrange(end_year, end_month)[1]),
            rebuild_baseline=False,
            reason="Monthly staffing requirements changed.",
        )
    db.session.commit()


def save_special_requirement(
    form: Any,
    *,
    db: Any,
    SpecialRequirement: Any,
    impact_type: Any,
    record_roster_impact: Callable[..., None],
) -> str:
    """Create or replace one date-specific staffing requirement."""
    try:
        selected_day = date.fromisoformat(form.get("special_day") or "")
    except ValueError as exc:
        raise ValueError("Choose a valid date for the special requirement.") from exc
    label = (form.get("special_label") or "").strip()[:80]
    if not label:
        raise ValueError("Describe the reason, for example Christmas Day.")
    row = SpecialRequirement.query.filter_by(day=selected_day).first()
    if not row:
        row = SpecialRequirement(day=selected_day)
        db.session.add(row)
    row.label = label
    for code in ("m", "d", "a", "n"):
        try:
            value = int(form.get(f"special_req_{code}") or 0)
        except ValueError as exc:
            raise ValueError("Special staffing values must be numbers.") from exc
        setattr(row, f"req_{code}", max(0, value))
    record_roster_impact(
        impact_type,
        selected_day,
        effective_to=selected_day,
        rebuild_baseline=False,
        reason=f"Special staffing requirement changed: {label}.",
    )
    db.session.commit()
    return f"Special requirements saved for {selected_day.strftime('%d %B %Y')}."


def delete_special_requirement(
    record_id: int,
    *,
    db: Any,
    SpecialRequirement: Any,
    impact_type: Any,
    record_roster_impact: Callable[..., None],
) -> None:
    row = SpecialRequirement.query.filter_by(id=record_id).first_or_404()
    removed_day = row.day
    db.session.delete(row)
    record_roster_impact(
        impact_type,
        removed_day,
        effective_to=removed_day,
        rebuild_baseline=False,
        reason="Special staffing requirement removed.",
    )
    db.session.commit()
