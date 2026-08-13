"""Roster reference-data initialization."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping


def bootstrap_reference_data(
    *,
    db: Any,
    Unit: Any,
    AnnotationType: Any,
    RosterSetting: Any,
    annotation_defaults: Iterable[Mapping[str, Any]],
    roster_defaults: Mapping[str, Any],
    normalise_codes: Callable[[Any], Any],
    refresh_annotation_cache: Callable[[], None],
    refresh_roster_settings_cache: Callable[[], None],
) -> None:
    """Create and populate canonical roster reference records idempotently."""
    Unit.__table__.create(bind=db.engine, checkfirst=True)
    AnnotationType.__table__.create(bind=db.engine, checkfirst=True)
    RosterSetting.__table__.create(bind=db.engine, checkfirst=True)
    if Unit.query.count() == 0:
        db.session.add(
            Unit(id=1, code="FIRST", name="First airport unit", status="active")
        )
        db.session.flush()

    if AnnotationType.query.count() == 0:
        for index, config in enumerate(annotation_defaults):
            db.session.add(
                AnnotationType(
                    code=config.get("code", "").upper(),
                    label=config.get("label") or config.get("code", ""),
                    category=config.get("category", "Other"),
                    allow_suffix=bool(config.get("allow_suffix", False)),
                    suffixes="".join(
                        sorted(
                            {
                                character
                                for character in (config.get("suffixes") or "").upper()
                            }
                        )
                    ),
                    toil_half_days=int(config.get("toil_half_days", 0) or 0),
                    tags=config.get("tags", ""),
                    is_active=bool(config.get("is_active", True)),
                    sort_order=config.get("sort_order", index * 10),
                )
            )
        db.session.commit()

    for unit in Unit.query.filter(Unit.status != "platform_control").all():
        if not AnnotationType.query.filter_by(unit_id=unit.id, code="INFO").first():
            db.session.add(
                AnnotationType(
                    unit_id=unit.id,
                    code="INFO",
                    label="Information",
                    category="Information",
                    colour="#6c757d",
                    description=(
                        "Additional roster information. Excluded from reports."
                    ),
                    allow_suffix=False,
                    suffixes="",
                    toil_half_days=0,
                    tags="info,report_exclude",
                    note_required=False,
                    admin_only=False,
                    is_active=True,
                    sort_order=0,
                )
            )

    for key, values in roster_defaults.items():
        if not RosterSetting.query.filter_by(unit_id=1, key=key).first():
            db.session.add(
                RosterSetting(
                    unit_id=1,
                    key=key,
                    value=json.dumps(normalise_codes(values)),
                )
            )
    db.session.commit()
    refresh_annotation_cache()
    refresh_roster_settings_cache()
