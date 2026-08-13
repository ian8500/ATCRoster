"""Composition of work-pattern, validation, and roster-proposal services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from override_classification_service import (
    OverrideClassificationDependencies,
    OverrideClassificationService,
)
from roster_proposal_service import RosterProposalDependencies, RosterProposalService
from roster_validation_service import (
    RosterValidationDependencies,
    RosterValidationService,
)
from work_pattern_admin_service import (
    WorkPatternAdminDependencies,
    WorkPatternAdminService,
)
from work_pattern_blueprint import (
    WorkPatternBlueprintDependencies,
    create_work_pattern_blueprint,
)
from work_pattern_migration_service import (
    WorkPatternMigrationDependencies,
    WorkPatternMigrationService,
)
from work_pattern_service import WorkPatternDependencies, WorkPatternService


@dataclass(frozen=True)
class PlanningDependencies:
    db: Any
    Staff: Any
    ShiftType: Any
    Leave: Any
    Assignment: Any
    Sickness: Any
    Requirement: Any
    SpecialRequirement: Any
    WorkPattern: Any
    WorkPatternDay: Any
    WorkPatternDayAllowedShift: Any
    StaffPatternAssignment: Any
    StaffRule: Any
    BankHoliday: Any
    RosterProposal: Any
    RosterProposalAssignment: Any
    ChangeLog: Any
    shift_group: Callable[[Any], str]
    requirements_for_day: Callable[..., Any]
    shift_group_for_day: Callable[..., str]
    shift_minutes: Callable[[Any], int]
    staff_is_countable_on: Callable[..., bool]
    staff_has_qualification: Callable[..., bool]
    would_trigger_fatigue: Callable[..., Any]
    compute_fairness_range: Callable[..., Any]
    now: Callable[[], Any]
    pattern_context: Callable[..., Any]
    is_admin_user: Callable[[Any], bool]
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    record_roster_impact: Callable[..., Any]


@dataclass(frozen=True)
class PlanningServices:
    patterns: Any
    validation: Any
    proposals: Any
    override_classification: Any


def create_planning_services(app: Any, deps: PlanningDependencies) -> PlanningServices:
    patterns = WorkPatternService(WorkPatternDependencies(
        Staff=deps.Staff, ShiftType=deps.ShiftType, Leave=deps.Leave,
        Assignment=deps.Assignment, WorkPattern=deps.WorkPattern,
        WorkPatternDay=deps.WorkPatternDay,
        WorkPatternDayAllowedShift=deps.WorkPatternDayAllowedShift,
        StaffPatternAssignment=deps.StaffPatternAssignment,
        StaffRule=deps.StaffRule, shift_group=deps.shift_group,
    ))
    admin = WorkPatternAdminService(WorkPatternAdminDependencies(
        db=deps.db, WorkPattern=deps.WorkPattern,
        WorkPatternDay=deps.WorkPatternDay,
        WorkPatternDayAllowedShift=deps.WorkPatternDayAllowedShift,
        ShiftType=deps.ShiftType, pattern_service=patterns,
    ))
    validation = RosterValidationService(RosterValidationDependencies(
        Staff=deps.Staff, ShiftType=deps.ShiftType, Assignment=deps.Assignment,
        StaffPatternAssignment=deps.StaffPatternAssignment,
        StaffRule=deps.StaffRule, work_pattern_service=patterns,
    ))
    proposals = RosterProposalService(RosterProposalDependencies(
        db=deps.db, Staff=deps.Staff, ShiftType=deps.ShiftType,
        Assignment=deps.Assignment, Sickness=deps.Sickness,
        Requirement=deps.Requirement, SpecialRequirement=deps.SpecialRequirement,
        RosterProposal=deps.RosterProposal,
        RosterProposalAssignment=deps.RosterProposalAssignment,
        ChangeLog=deps.ChangeLog, work_pattern_service=patterns,
        requirements_for_day=deps.requirements_for_day,
        shift_group_for_day=deps.shift_group_for_day,
        shift_minutes=deps.shift_minutes,
        staff_is_countable_on=deps.staff_is_countable_on,
        staff_has_qualification=deps.staff_has_qualification,
        would_trigger_fatigue=deps.would_trigger_fatigue,
        compute_fairness_range=deps.compute_fairness_range, utcnow=deps.now,
    ))
    override_classification = OverrideClassificationService(
        OverrideClassificationDependencies(
            Assignment=deps.Assignment, Staff=deps.Staff,
            ShiftType=deps.ShiftType, work_pattern_service=patterns,
        )
    )
    migration = WorkPatternMigrationService(WorkPatternMigrationDependencies(
        db=deps.db, Staff=deps.Staff, WorkPattern=deps.WorkPattern,
        WorkPatternDay=deps.WorkPatternDay, ShiftType=deps.ShiftType,
        StaffPatternAssignment=deps.StaffPatternAssignment,
        pattern_context=deps.pattern_context, pattern_service=patterns,
    ))
    app.register_blueprint(create_work_pattern_blueprint(
        WorkPatternBlueprintDependencies(
            db=deps.db, Staff=deps.Staff, ShiftType=deps.ShiftType,
            WorkPattern=deps.WorkPattern, WorkPatternDay=deps.WorkPatternDay,
            WorkPatternDayAllowedShift=deps.WorkPatternDayAllowedShift,
            StaffPatternAssignment=deps.StaffPatternAssignment,
            StaffRule=deps.StaffRule, BankHoliday=deps.BankHoliday,
            is_admin_user=deps.is_admin_user,
            current_unit_id=deps.current_unit_id,
            validate_csrf=deps.validate_csrf, pattern_service=patterns,
            admin_service=admin, migration_service=migration,
            record_roster_impact=deps.record_roster_impact,
        )
    ))
    return PlanningServices(patterns, validation, proposals, override_classification)
