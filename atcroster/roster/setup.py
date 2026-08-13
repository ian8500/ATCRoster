"""Unit-level roster pattern configuration."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def update_unit_roster_setup(
    values: Mapping[str, str],
    *,
    db: Any,
    validate_pattern: Callable[[str | None], list[str]],
    parse_date: Callable[[str | None], Any],
    save_setting: Callable[[str, str], None],
    record_roster_impact: Callable[..., None],
    impact_type: Any,
) -> tuple[str, str]:
    """Validate and atomically persist the unit's base roster pattern."""
    pattern = validate_pattern(values.get("base_pattern_csv"))
    anchor = parse_date(values.get("base_pattern_anchor"))
    if not pattern or not anchor:
        return "Choose at least one valid base-pattern duty and a start date.", "error"

    active_nights = [str(day) for day in range(7) if values.get(f"night_day_{day}")]
    save_setting("base_pattern_csv", ",".join(pattern))
    save_setting("base_pattern_anchor", anchor.isoformat())
    save_setting("night_active_weekdays", ",".join(active_nights))
    record_roster_impact(
        impact_type,
        anchor,
        rebuild_baseline=True,
        reason="Unit base roster pattern changed.",
    )
    db.session.commit()
    return "Unit roster setup saved.", "ok"
