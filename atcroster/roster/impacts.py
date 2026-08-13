"""Roster-impact horizon and cache invalidation support."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Any, Callable


def generated_horizon_end(
    unit_id: int,
    effective_from: date,
    *,
    db: Any,
    Assignment: Any,
) -> date | None:
    return (
        db.session.query(db.func.max(Assignment.day))
        .filter(
            Assignment.unit_id == unit_id,
            Assignment.day >= effective_from,
        )
        .scalar()
    )


def invalidate_impact_months(
    unit_id: int,
    effective_from: date,
    effective_to: date,
    *,
    cache: Any,
    cached_loader: Callable[..., Any],
    add_months: Callable[[int, int, int], tuple[int, int]],
) -> None:
    """Invalidate every monthly roster cache entry touched by an impact."""
    cursor = effective_from.replace(day=1)
    final = effective_to.replace(day=1)
    while cursor <= final:
        if cache:
            try:
                cache.delete_memoized(
                    cached_loader, int(unit_id), cursor.year, cursor.month
                )
            except Exception:
                pass
        next_year, next_month = add_months(cursor.year, cursor.month, 1)
        cursor = date(next_year, next_month, 1)


@dataclass(frozen=True)
class RosterImpactRuntimeDependencies:
    db: Any
    Unit: Any
    Assignment: Any
    RosterImpactEvent: Any
    RosterImpactException: Any
    RosterImpactEventType: Any
    PersonQualification: Any
    QualificationType: Any
    cache: Any
    cached_loader: Callable[..., Any]
    add_months: Callable[[int, int, int], tuple[int, int]]
    current_unit_id: Callable[[], int]
    current_user: Callable[[], Any]
    population_service: Callable[[], Any]
    override_classifier: Callable[[], Any]
    service_factory: Callable[[Any], Any]
    service_dependencies: Callable[..., Any]
    classify_qualification_impact: Callable[..., Any]
    has_other_valid_ue: Callable[..., bool]
    record_qualification_impact: Callable[..., Any]
    now: Callable[[], Any]


def create_roster_impact_runtime_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any,
    roster_impact_event_type: Any, **services: Any
) -> RosterImpactRuntimeDependencies:
    """Bind roster-impact records at the roster-domain boundary."""
    return RosterImpactRuntimeDependencies(
        db=db,
        Unit=operational_models.Unit,
        Assignment=operational_models.Assignment,
        RosterImpactEvent=saas_models.RosterImpactEvent,
        RosterImpactException=saas_models.RosterImpactException,
        RosterImpactEventType=roster_impact_event_type,
        PersonQualification=saas_models.PersonQualification,
        QualificationType=saas_models.QualificationType,
        **services,
    )


class RosterImpactRuntime:
    """Own roster-impact service construction, recording, and qualification mapping."""

    def __init__(self, dependencies: RosterImpactRuntimeDependencies):
        self.dependencies = dependencies

    def generated_horizon_end(self, unit_id: int, effective_from: date):
        deps = self.dependencies
        return generated_horizon_end(
            unit_id, effective_from, db=deps.db, Assignment=deps.Assignment
        )

    def invalidate_coverage(
        self,
        unit_id: int,
        effective_from: date,
        effective_to: date,
        _staff_ids: tuple[int, ...],
        _watch_ids: tuple[int, ...],
    ) -> None:
        deps = self.dependencies
        return invalidate_impact_months(
            unit_id,
            effective_from,
            effective_to,
            cache=deps.cache,
            cached_loader=deps.cached_loader,
            add_months=deps.add_months,
        )

    def service(self):
        deps = self.dependencies
        return deps.service_factory(
            deps.service_dependencies(
                db=deps.db,
                Unit=deps.Unit,
                RosterImpactEvent=deps.RosterImpactEvent,
                RosterImpactException=deps.RosterImpactException,
                population_service=deps.population_service(),
                generated_horizon_end=self.generated_horizon_end,
                recalculate_coverage=self.invalidate_coverage,
                override_classifier=deps.override_classifier(),
                utcnow=deps.now,
            )
        )

    def record(
        self,
        event_type: Any,
        effective_from: date,
        *,
        effective_to: date | None = None,
        staff_ids=(),
        watch_ids=(),
        rebuild_baseline: bool = False,
        recalculate_coverage: bool = True,
        reason: str = "",
    ):
        user = self.dependencies.current_user()
        actor_id = getattr(user, "person_id", None)
        if actor_id is None and getattr(user, "is_authenticated", False):
            actor_id = getattr(user, "id", None)
        return self.service().handle_roster_impact_event(
            self.dependencies.current_unit_id(),
            event_type,
            effective_from,
            effective_to,
            staff_ids=staff_ids,
            watch_ids=watch_ids,
            rebuild_baseline=rebuild_baseline,
            recalculate_coverage=recalculate_coverage,
            reason=reason,
            triggered_by_user_id=actor_id,
        )

    def qualification_impact_type(
        self,
        code: str,
        old_status: str | None,
        old_valid_from: date | None,
        old_expires_on: date | None,
        new_status: str | None,
        new_valid_from: date | None,
        new_expires_on: date | None,
    ):
        deps = self.dependencies
        return deps.classify_qualification_impact(
            code,
            old_status,
            old_valid_from,
            old_expires_on,
            new_status,
            new_valid_from,
            new_expires_on,
            impact_types=deps.RosterImpactEventType,
            today=date.today(),
        )

    def person_has_other_valid_ue(
        self, unit_id: int, person_id: int, excluded_type_id: int, on_date: date
    ) -> bool:
        deps = self.dependencies
        return deps.has_other_valid_ue(
            unit_id,
            person_id,
            excluded_type_id,
            on_date,
            db=deps.db,
            PersonQualification=deps.PersonQualification,
            QualificationType=deps.QualificationType,
        )

    def record_qualification(
        self,
        person: Any,
        qualification_type: Any,
        old_status: str | None,
        old_valid_from: date | None,
        old_expires_on: date | None,
        record: Any,
        *,
        reason: str = "Qualification changed.",
    ):
        deps = self.dependencies
        return deps.record_qualification_impact(
            person,
            qualification_type,
            old_status,
            old_valid_from,
            old_expires_on,
            record,
            impact_types=deps.RosterImpactEventType,
            today=date.today(),
            has_other_ue=self.person_has_other_valid_ue,
            record_roster_impact=self.record,
            reason=reason,
        )
