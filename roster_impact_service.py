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
    override_classifier: Any = None


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
        **options: Any,
    ) -> RosterImpactResult:
        """Apply an impact atomically and retain an honest failed audit row."""
        if options.get("preserve_overrides", True) is False:
            raise ValueError("Roster-impact processing must preserve editor overrides.")
        if effective_to is not None and effective_to < effective_from:
            raise ValueError("Roster-impact event end date cannot precede its start date.")
        try:
            kind = event_type if isinstance(event_type, RosterImpactEventType) else RosterImpactEventType(str(event_type))
        except ValueError as exc:
            raise ValueError("Unknown roster-impact event type.") from exc
        try:
            return self._handle_roster_impact_event(
                unit_id, kind, effective_from, effective_to, **options
            )
        except Exception as exc:
            # Discard both the workforce mutation and partial recalculation,
            # then write the failure in a clean transaction. This avoids the
            # dangerous state where only half of an impact is persisted.
            dep = self.dependencies
            dep.db.session.rollback()
            unit = dep.db.session.get(dep.Unit, int(unit_id))
            if unit is not None:
                failed_at = dep.utcnow()
                failed = dep.RosterImpactEvent(
                    unit_id=unit.id, event_type=kind.value,
                    effective_from=effective_from, effective_to=effective_to,
                    staff_ids_json=json.dumps(tuple(sorted({
                        int(value) for value in options.get("staff_ids") or ()
                    }))),
                    watch_ids_json=json.dumps(tuple(sorted({
                        int(value) for value in options.get("watch_ids") or ()
                    }))),
                    rebuild_baseline=bool(options.get("rebuild_baseline", False)),
                    recalculate_coverage=bool(options.get("recalculate_coverage", True)),
                    preserve_overrides=True,
                    reason=(options.get("reason") or "")[:500],
                    triggered_by_user_id=options.get("triggered_by_user_id"),
                    status="FAILED", started_at=failed_at, completed_at=failed_at,
                    error_message=str(exc)[:2000], created_at=failed_at,
                    affected_dates=(effective_to - effective_from).days + 1
                    if effective_to else 0,
                )
                dep.db.session.add(failed)
                dep.db.session.commit()
            raise

    def _handle_roster_impact_event(
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
        allow_protected_rebuild: bool = False,
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
            if allow_protected_rebuild:
                automatic_from = effective_from
                automatic_to = horizon

        with dep.db.session.begin_nested():
            event = dep.RosterImpactEvent(
                unit_id=unit.id, event_type=kind.value,
                effective_from=effective_from, effective_to=horizon,
                staff_ids_json=json.dumps(staff), watch_ids_json=json.dumps(watches),
                rebuild_baseline=bool(rebuild_baseline),
                recalculate_coverage=bool(recalculate_coverage),
                preserve_overrides=True, reason=(reason or "")[:500],
                triggered_by_user_id=triggered_by_user_id, status="RUNNING",
                protected_from=protected_from, protected_to=protected_to,
                automatic_from=automatic_from, automatic_to=automatic_to,
                created_at=dep.utcnow(), started_at=dep.utcnow(),
                affected_dates=(horizon - effective_from).days + 1 if horizon else 0,
            )
            dep.db.session.add(event)
            dep.db.session.flush()

            exception_count = 0
            if rebuild_baseline and protected_from is not None and not allow_protected_rebuild:
                exception_type, severity = _protected_exception_details(kind)
                scopes = [(value, None) for value in staff] or [(None, value) for value in watches] or [(None, None)]
                for staff_id, watch_id in scopes:
                    dep.db.session.add(dep.RosterImpactException(
                        unit_id=unit.id, event_id=event.id, staff_id=staff_id,
                        watch_id=watch_id, effective_from=protected_from,
                        effective_to=protected_to, exception_type=exception_type,
                        severity=severity,
                        description=(reason or f"{kind.value} requires review inside the protected roster period.")[:1000],
                        status="OPEN", created_at=dep.utcnow(),
                    ))
                    exception_count += 1

            population_result = None
            if rebuild_baseline and automatic_from is not None:
                population_result = dep.population_service.populate_or_recalculate_baseline(
                    unit.id, automatic_from, automatic_to, staff_ids=staff,
                    watch_ids=watches,
                    mode="manual" if allow_protected_rebuild else "event", reason=reason,
                    triggered_by_user_id=triggered_by_user_id,
                    reference_date=reference_date, generation_event_id=event.id,
                )

            classification_result = None
            if horizon is not None and dep.override_classifier is not None:
                classification_result = dep.override_classifier.classify_range(
                    unit.id, effective_from, horizon, staff_ids=staff,
                    preserve_redundant=bool(
                        getattr(unit, "preserve_redundant_overrides", True)
                    ),
                )
                for finding in classification_result.findings:
                    exception_type = {
                        "AFTER_UNIT_LEAVING_DATE": "OVERRIDE_AFTER_LEAVING_DATE",
                        "OUTSIDE_EMPLOYMENT": "INVALID_OPERATIONAL_CONTRIBUTION",
                        "CONFLICTS_WITH_HARD_RESTRICTION": "PATTERN_CHANGE_REQUIRES_REVIEW",
                        "REQUIRES_REVIEW": "PATTERN_CHANGE_REQUIRES_REVIEW",
                    }[finding.classification]
                    dep.db.session.add(dep.RosterImpactException(
                        unit_id=unit.id, event_id=event.id,
                        staff_id=finding.assignment.staff_id,
                        effective_from=finding.assignment.day,
                        effective_to=finding.assignment.day,
                        exception_type=exception_type,
                        severity="CRITICAL" if finding.classification in {
                            "AFTER_UNIT_LEAVING_DATE", "OUTSIDE_EMPLOYMENT"
                        } else "WARNING",
                        description=finding.description[:1000], status="OPEN",
                        created_at=dep.utcnow(),
                    ))
                    exception_count += 1

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
                "overrides": None if classification_result is None else {
                    "classified": classification_result.classified,
                    "redundant": classification_result.redundant,
                    "invalid": classification_result.invalid,
                },
            }
            event.result_json = json.dumps(summary, default=str, sort_keys=True)
            population_summary = summary.get("population") or {}
            event.assignments_created = int(population_summary.get("created") or 0)
            event.baselines_changed = int(population_summary.get("updated") or 0)
            event.overrides_retained = sum(
                1 for change in getattr(population_result, "changes", ())
                if change.effective_code != change.generated_code
            )
            event.exceptions_created = exception_count
            event.redundant_overrides_found = (
                classification_result.redundant if classification_result else 0
            )
            event.warnings_created = exception_count + int(
                population_summary.get("unresolved_flexible_dates") or 0
            )
            event.status = (
                "COMPLETED_WITH_WARNINGS" if event.warnings_created else "COMPLETED"
            )
            event.completed_at = dep.utcnow()
            dep.db.session.flush()

        return RosterImpactResult(
            event_id=event.id, protected_from=protected_from,
            protected_to=protected_to, automatic_from=automatic_from,
            automatic_to=automatic_to, exception_count=exception_count,
            coverage_recalculated=coverage_done,
            population_result=population_result,
        )


def _protected_exception_details(
    kind: RosterImpactEventType,
) -> tuple[str, str]:
    if kind == RosterImpactEventType.UNIT_JOINER:
        return "JOINER_REQUIRES_MANUAL_ROSTER_ENTRY", "WARNING"
    if kind == RosterImpactEventType.UNIT_LEAVER:
        return "LEAVER_HAS_PROTECTED_DUTIES", "CRITICAL"
    if kind == RosterImpactEventType.WATCH_TRANSFER:
        return "WATCH_MOVE_NOT_APPLIED_TO_PROTECTED_ROSTER", "WARNING"
    if kind in {
        RosterImpactEventType.PART_TIME_CHANGE,
        RosterImpactEventType.FULL_TIME_CHANGE,
    }:
        return "PART_TIME_CHANGE_REQUIRES_REVIEW", "WARNING"
    return "PATTERN_CHANGE_REQUIRES_REVIEW", "WARNING"
