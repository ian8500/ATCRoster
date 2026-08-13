"""Watch definition administration and roster-impact handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class WatchConfigurationDependencies:
    db: Any
    Watch: Any
    Staff: Any
    StaffWatchHistory: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    validate_pattern: Callable[[str | None], list[str]]
    parse_date: Callable[[str | None], Any]
    record_roster_impact: Callable[..., None]


def update_watch_configuration(
    action: str,
    values: Mapping[str, str],
    dependencies: WatchConfigurationDependencies,
) -> tuple[str, str]:
    """Create, update, or delete a unit watch definition."""
    unit_id = dependencies.current_unit_id()
    if action == "watch_new":
        name = (values.get("name") or "").strip()
        pattern = dependencies.validate_pattern(values.get("pattern_csv"))
        anchor = dependencies.parse_date(values.get("pattern_anchor"))
        if not name:
            return "Enter a watch name.", "error"
        if dependencies.Watch.query.filter_by(unit_id=unit_id, name=name).first():
            return "That watch name already exists.", "error"
        max_order = (
            dependencies.db.session.query(
                dependencies.db.func.max(dependencies.Watch.order_index)
            )
            .filter(dependencies.Watch.unit_id == unit_id)
            .scalar()
            or 0
        )
        watch = dependencies.Watch(
            unit_id=unit_id,
            name=name[:32],
            order_index=max_order + 1,
            pattern_csv=",".join(pattern),
            pattern_anchor=anchor,
        )
        dependencies.db.session.add(watch)
        dependencies.db.session.flush()
        dependencies.record_roster_impact(
            dependencies.RosterImpactEventType.WATCH_CREATION,
            anchor or date.today(),
            watch_ids=[watch.id],
            rebuild_baseline=False,
            reason=f"Watch {name[:32]} created.",
        )
        dependencies.db.session.commit()
        return f"{name} created.", "ok"

    watch = dependencies.Watch.query.filter_by(
        id=int(values.get("watch_id") or 0), unit_id=unit_id
    ).first_or_404()
    if action == "watch_edit":
        name = (values.get("name") or "").strip()
        pattern = dependencies.validate_pattern(values.get("pattern_csv"))
        anchor = dependencies.parse_date(values.get("pattern_anchor"))
        duplicate = dependencies.Watch.query.filter(
            dependencies.Watch.unit_id == unit_id,
            dependencies.Watch.name == name,
            dependencies.Watch.id != watch.id,
        ).first()
        if not name:
            return "Enter a watch name.", "error"
        if duplicate:
            return "That watch name already exists.", "error"
        old_pattern, old_anchor = watch.pattern_csv, watch.pattern_anchor
        watch.name = name[:32]
        watch.pattern_csv = ",".join(pattern)
        watch.pattern_anchor = anchor
        if (watch.pattern_csv, watch.pattern_anchor) != (old_pattern, old_anchor):
            dependencies.record_roster_impact(
                dependencies.RosterImpactEventType.WATCH_PATTERN_CHANGE,
                anchor or date.today(),
                watch_ids=[watch.id],
                rebuild_baseline=True,
                reason=f"Watch {watch.name} pattern changed.",
            )
        dependencies.db.session.commit()
        return f"{watch.name} updated.", "ok"

    in_use = (
        dependencies.Staff.query.filter_by(unit_id=unit_id, watch_id=watch.id).first()
        or dependencies.StaffWatchHistory.query.filter_by(
            unit_id=unit_id, watch_id=watch.id
        ).first()
    )
    if in_use:
        return (
            "Move staff and remove scheduled moves before deleting this watch.",
            "error",
        )
    name = watch.name
    dependencies.record_roster_impact(
        dependencies.RosterImpactEventType.WATCH_DEACTIVATION,
        date.today(),
        rebuild_baseline=False,
        reason=f"Unused watch {name} removed.",
    )
    dependencies.db.session.delete(watch)
    dependencies.db.session.commit()
    return f"{name} deleted.", "ok"
