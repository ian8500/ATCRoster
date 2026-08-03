"""Transactional orchestration for changes which affect a generated roster."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Iterable

from roster_horizon import get_unit_automatic_recalculation_start


class RosterImpactEventType(str, Enum):
    UNIT_JOINER = "UNIT_JOINER"
    UNIT_LEAVER = "UNIT_LEAVER"
    RETURN_TO_UNIT = "RETURN_TO_UNIT"
    TEMPORARY_DETACHMENT_START = "TEMPORARY_DETACHMENT_START"
    TEMPORARY_DETACHMENT_END = "TEMPORARY_DETACHMENT_END"
    WATCH_TRANSFER = "WATCH_TRANSFER"
    WORK_PATTERN_CHANGE = "WORK_PATTERN_CHANGE"
    PART_TIME_CHANGE = "PART_TIME_CHANGE"
    FULL_TIME_CHANGE = "FULL_TIME_CHANGE"
    PATTERN_ANCHOR_CHANGE = "PATTERN_ANCHOR_CHANGE"
    OPERATIONAL_ROSTER_ACTIVATION = "OPERATIONAL_ROSTER_ACTIVATION"
    OPERATIONAL_ROSTER_DEACTIVATION = "OPERATIONAL_ROSTER_DEACTIVATION"
    FIRST_UE_ACHIEVED = "FIRST_UE_ACHIEVED"
    ADDITIONAL_UE_ACHIEVED = "ADDITIONAL_UE_ACHIEVED"
    UE_EXPIRED = "UE_EXPIRED"
    UE_SUSPENDED = "UE_SUSPENDED"
    UE_RESTORED = "UE_RESTORED"
    MEDICAL_EXPIRED = "MEDICAL_EXPIRED"
    MEDICAL_RESTORED = "MEDICAL_RESTORED"
    OJTI_ACHIEVED = "OJTI_ACHIEVED"
    ASSESSOR_ACHIEVED = "ASSESSOR_ACHIEVED"
    COMPETENCY_RESTRICTION_START = "COMPETENCY_RESTRICTION_START"
    COMPETENCY_RESTRICTION_END = "COMPETENCY_RESTRICTION_END"
    WATCH_PATTERN_CHANGE = "WATCH_PATTERN_CHANGE"
    SHIFT_DEFINITION_CHANGE = "SHIFT_DEFINITION_CHANGE"
    STAFFING_REQUIREMENT_CHANGE = "STAFFING_REQUIREMENT_CHANGE"
    WATCH_CREATION = "WATCH_CREATION"
    WATCH_DEACTIVATION = "WATCH_DEACTIVATION"
    FUTURE_PERIOD_CREATED = "FUTURE_PERIOD_CREATED"
    MANUAL_RECALCULATION = "MANUAL_RECALCULATION"


@dataclass(frozen=True)
class RosterImpactDependencies:
    db: Any
    Unit: Any
    RosterImpactEvent: Any
    RosterImpactException: Any
    population_service: Any
    generated_horizon_end: Callable[[int, date], date | None]
    utcnow: Callable[[], Any]
    recalculate_coverage: Callable[[int, date, date, tuple[int, ...], tuple[int, ...]], None] | None = None


@dataclass(frozen=True)
class RosterImpactResult:
    event_id: int
    protected_from: date | None
    protected_to: date | None
    automatic_from: date | None
    automatic_to: date | None
    exception_count: int
    coverage_recalculated: bool
    population_result: Any = None


class RosterImpactService:
    def __init__(self, dependencies: RosterImpactDependencies) -> None:
        self.dependencies = dependencies

    def handle_roster_impact_event(
        self,
        unit_id: int,
        event_type: RosterImpactEventType | str,
        effective_from: date,
        effective_to: date | None = None,
        *,
        staff_ids: Iterable[int] | None = None,
        watch_ids: Iterable[int] | None = None,
        rebuild_baseline: bool = False,
        recalculate_coverage: bool = True,
        preserve_overrides: bool = True,
        reason: str | None = None,
        triggered_by_user_id: int | None = None,
        reference_date: date | None = None,
    ) -> RosterImpactResult:
        if not preserve_overrides:
            raise ValueError("Roster-impact processing must preserve editor overrides.")
        if effective_to is not None and effective_to < effective_from:
            raise ValueError("Roster-impact event end date cannot precede its start date.")
        try:
            kind = event_type if isinstance(event_type, RosterImpactEventType) else RosterImpactEventType(str(event_type))
        except ValueError as exc:
            raise ValueError("Unknown roster-impact event type.") from exc

        dep = self.dependencies
        unit = dep.db.session.get(dep.Unit, int(unit_id))
        if unit is None:
            raise ValueError("The selected airport does not exist.")
        staff = tuple(sorted({int(value) for value in staff_ids or ()}))
        watches = tuple(sorted({int(value) for value in watch_ids or ()}))
        horizon = effective_to or dep.generated_horizon_end(unit.id, effective_from)
        if horizon is not None and horizon < effective_from:
            horizon = None
        boundary = get_unit_automatic_recalculation_start(unit, reference_date)
        protected_from = protected_to = automatic_from = automatic_to = None
        if horizon is not None:
            if effective_from < boundary:
                protected_from = effective_from
                protected_to = min(horizon, boundary - timedelta(days=1))
            if horizon >= boundary:
                automatic_from = max(effective_from, boundary)
                automatic_to = horizon

        with dep.db.session.begin_nested():
            event = dep.RosterImpactEvent(
                unit_id=unit.id, event_type=kind.value,
                effective_from=effective_from, effective_to=horizon,
                staff_ids_json=json.dumps(staff), watch_ids_json=json.dumps(watches),
                rebuild_baseline=bool(rebuild_baseline),
                recalculate_coverage=bool(recalculate_coverage),
                preserve_overrides=True, reason=(reason or "")[:500],
                triggered_by_user_id=triggered_by_user_id, status="PROCESSING",
                protected_from=protected_from, protected_to=protected_to,
                automatic_from=automatic_from, automatic_to=automatic_to,
                created_at=dep.utcnow(),
            )
            dep.db.session.add(event)
            dep.db.session.flush()

            exception_count = 0
            if rebuild_baseline and protected_from is not None:
                scopes = [(value, None) for value in staff] or [(None, value) for value in watches] or [(None, None)]
                for staff_id, watch_id in scopes:
                    dep.db.session.add(dep.RosterImpactException(
                        unit_id=unit.id, event_id=event.id, staff_id=staff_id,
                        watch_id=watch_id, effective_from=protected_from,
                        effective_to=protected_to, exception_type="PROTECTED_ROSTER_IMPACT",
                        severity="WARNING",
                        description=(reason or f"{kind.value} requires review inside the protected roster period.")[:1000],
                        status="OPEN", created_at=dep.utcnow(),
                    ))
                    exception_count += 1

            population_result = None
            if rebuild_baseline and automatic_from is not None:
                population_result = dep.population_service.populate_or_recalculate_baseline(
                    unit.id, automatic_from, automatic_to, staff_ids=staff,
                    watch_ids=watches, mode="event", reason=reason,
                    triggered_by_user_id=triggered_by_user_id,
                    reference_date=reference_date, generation_event_id=event.id,
                )

            coverage_done = False
            if recalculate_coverage and horizon is not None and dep.recalculate_coverage:
                dep.recalculate_coverage(unit.id, effective_from, horizon, staff, watches)
                coverage_done = True

            summary = {
                "exception_count": exception_count,
                "coverage_recalculated": coverage_done,
                "population": None if population_result is None else {
                    key: value for key, value in asdict(population_result).items()
                    if key != "changes"
                },
            }
            event.result_json = json.dumps(summary, default=str, sort_keys=True)
            event.status = "COMPLETED"
            event.completed_at = dep.utcnow()
            dep.db.session.flush()

        return RosterImpactResult(
            event_id=event.id, protected_from=protected_from,
            protected_to=protected_to, automatic_from=automatic_from,
            automatic_to=automatic_to, exception_count=exception_count,
            coverage_recalculated=coverage_done,
            population_result=population_result,
        )
