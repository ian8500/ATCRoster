"""Operational monitoring, handover, and assurance route composition."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from handover_blueprint import HandoverDependencies, create_handover_blueprint
from live_position_blueprint import LivePositionDependencies, create_live_position_blueprint
from operations_blueprint import OperationsDependencies, create_operations_blueprint

@dataclass(frozen=True)
class OperationalRegistrationDependencies:
    db: Any
    Unit: Any
    Staff: Any
    Watch: Any
    ShiftType: Any
    Assignment: Any
    Requirement: Any
    SpecialRequirement: Any
    FeatureFlag: Any
    OperationalPosition: Any
    OperationalPositionTimeAllowance: Any
    OperationalPositionGroup: Any
    PositionCurrencyCategory: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    PositionParticipantRole: Any
    PositionSessionAudit: Any
    PositionEndorsement: Any
    PositionRequirement: Any
    HandoverField: Any
    HandoverRecord: Any
    HandoverOperationalState: Any
    HandoverEquipment: Any
    BreakPlan: Any
    AchievedDuty: Any
    FatigueReport: Any
    RosterRuleVersion: Any
    Scenario: Any
    now: Callable[[], Any]
    is_admin_user: Callable[[Any], bool]
    is_editor_user: Callable[[Any], bool]
    can_edit_roster: Callable[[Any], bool]
    live_position_enabled: Callable[[int], bool]
    competency_enabled: Callable[[int], bool]
    authenticated_database_route_optional: Callable[[], Any]
    authenticated_unit_context: Callable[..., Any]
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    requirements_for_day: Callable[..., Any]
    shift_group_for_day: Callable[..., Any]
    compliance_month: Callable[..., Any]
    log_change: Callable[..., Any]
    month_add: Callable[..., Any]
    position_assurance: Callable[..., Any]
    parse_year_month: Callable[..., Any]
    month_range: Callable[..., Any]
    staff_has_shift_qualification: Callable[..., bool]


def create_operational_registration_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> OperationalRegistrationDependencies:
    """Bind operational-route models at the live-position boundary."""
    return OperationalRegistrationDependencies(
        db=db,
        Unit=operational_models.Unit,
        Staff=operational_models.Staff,
        Watch=operational_models.Watch,
        ShiftType=operational_models.ShiftType,
        Assignment=operational_models.Assignment,
        Requirement=operational_models.Requirement,
        SpecialRequirement=operational_models.SpecialRequirement,
        FeatureFlag=saas_models.FeatureFlag,
        OperationalPosition=saas_models.OperationalPosition,
        OperationalPositionTimeAllowance=saas_models.OperationalPositionTimeAllowance,
        OperationalPositionGroup=saas_models.OperationalPositionGroup,
        PositionCurrencyCategory=saas_models.PositionCurrencyCategory,
        PositionStatusEvent=saas_models.PositionStatusEvent,
        PositionSession=saas_models.PositionSession,
        PositionSessionParticipant=saas_models.PositionSessionParticipant,
        PositionParticipantRole=saas_models.PositionParticipantRole,
        PositionSessionAudit=saas_models.PositionSessionAudit,
        PositionEndorsement=saas_models.PositionEndorsement,
        PositionRequirement=saas_models.PositionRequirement,
        HandoverField=operational_models.HandoverField,
        HandoverRecord=operational_models.HandoverRecord,
        HandoverOperationalState=operational_models.HandoverOperationalState,
        HandoverEquipment=operational_models.HandoverEquipment,
        BreakPlan=saas_models.BreakPlan,
        AchievedDuty=saas_models.AchievedDuty,
        FatigueReport=saas_models.FatigueReport,
        RosterRuleVersion=saas_models.RosterRuleVersion,
        Scenario=saas_models.Scenario,
        **services,
    )


def register_operational_blueprints(app: Any, d: OperationalRegistrationDependencies) -> None:
    app.register_blueprint(create_live_position_blueprint(LivePositionDependencies(
        db=d.db, Unit=d.Unit, OperationalPosition=d.OperationalPosition,
        OperationalPositionTimeAllowance=d.OperationalPositionTimeAllowance,
        OperationalPositionGroup=d.OperationalPositionGroup,
        PositionCurrencyCategory=d.PositionCurrencyCategory,
        PositionStatusEvent=d.PositionStatusEvent, PositionSession=d.PositionSession,
        PositionSessionParticipant=d.PositionSessionParticipant,
        PositionParticipantRole=d.PositionParticipantRole,
        PositionSessionAudit=d.PositionSessionAudit,
        PositionEndorsement=d.PositionEndorsement, Staff=d.Staff, Watch=d.Watch,
        utcnow=d.now, is_admin_user=d.is_admin_user,
        live_position_enabled=d.live_position_enabled,
        competency_enabled=d.competency_enabled,
        authenticated_database_route_optional=d.authenticated_database_route_optional,
        authenticated_unit_context=d.authenticated_unit_context,
    )))
    app.register_blueprint(create_handover_blueprint(HandoverDependencies(
        db=d.db, Unit=d.Unit, Staff=d.Staff, ShiftType=d.ShiftType,
        Assignment=d.Assignment, Requirement=d.Requirement,
        SpecialRequirement=d.SpecialRequirement, FeatureFlag=d.FeatureFlag,
        HandoverField=d.HandoverField, HandoverRecord=d.HandoverRecord,
        HandoverOperationalState=d.HandoverOperationalState,
        HandoverEquipment=d.HandoverEquipment,
        OperationalPosition=d.OperationalPosition, PositionSession=d.PositionSession,
        current_unit_id=d.current_unit_id, validate_csrf=d.validate_csrf,
        is_admin_user=d.is_admin_user, is_editor_user=d.is_editor_user,
        requirements_for_day=d.requirements_for_day,
        shift_group_for_day=d.shift_group_for_day, utcnow=d.now,
        live_position_enabled=d.live_position_enabled,
    )))
    app.register_blueprint(create_operations_blueprint(OperationsDependencies(
        db=d.db, OperationalPosition=d.OperationalPosition,
        PositionEndorsement=d.PositionEndorsement,
        PositionRequirement=d.PositionRequirement, Staff=d.Staff,
        ShiftType=d.ShiftType, BreakPlan=d.BreakPlan, Assignment=d.Assignment,
        AchievedDuty=d.AchievedDuty, FatigueReport=d.FatigueReport,
        RosterRuleVersion=d.RosterRuleVersion, is_admin_user=d.is_admin_user,
        compliance_month=d.compliance_month, validate_csrf=d.validate_csrf,
        current_unit_id=d.current_unit_id, utcnow=d.now, log_change=d.log_change,
        month_add=d.month_add, position_assurance=d.position_assurance,
        can_edit_roster=d.can_edit_roster, parse_year_month=d.parse_year_month,
        month_range=d.month_range, shift_counter_group_for_day=d.shift_group_for_day,
        staff_has_shift_qualification=d.staff_has_shift_qualification,
        Scenario=d.Scenario,
    )))
