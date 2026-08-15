"""Tenant routing registry for operational and append-only model classes."""

from __future__ import annotations

from typing import Any


def operational_models(operational: Any, saas: Any, briefing: Any) -> tuple[Any, ...]:
    """Return all records that must be airport-scoped by the session hooks."""
    return (
        operational.RosterSetting, operational.AnnotationType, operational.Watch,
        operational.Staff, operational.ShiftType, operational.Requirement,
        operational.SpecialRequirement, operational.SmsAudit, operational.Leave,
        operational.Sickness, operational.Assignment, operational.ShiftRequest,
        operational.RequestAudit, operational.Notification, operational.AnnotationAudit,
        operational.ChangeLog, operational.StaffWatchHistory,
        saas.QualificationType, saas.PersonQualification,
        saas.PersonQualificationHistory, saas.RosterPublication,
        saas.RosterAcknowledgement, saas.Scenario, saas.OperationalPosition,
        saas.LivePositionRecoveryPolicy,
        saas.PositionCurrencyCategory, saas.PositionParticipantRole,
        saas.PositionStatusEvent, saas.PositionSession,
        saas.PositionSessionParticipant, saas.ControllerKioskCredential,
        saas.PositionSessionAudit, saas.PositionEndorsement,
        saas.PositionRequirement, saas.BreakPlan, saas.AchievedDuty,
        saas.FatigueReport, saas.RosterRuleVersion, saas.RosterPeriod,
        saas.RosterImpactEvent, saas.RosterImpactException, saas.MfaCredential,
        briefing.BriefingMessageType, briefing.BriefingItem, briefing.BriefingDelivery,
        briefing.BriefingAudit, briefing.BriefingAssuranceRun,
        operational.HandoverField, operational.HandoverRecord,
        operational.HandoverOperationalState, operational.HandoverEquipment,
        operational.TrainingLevel, operational.TrainingObjective,
        operational.TrainingSession, operational.TrainingScore,
        saas.ToilTransaction, saas.WorkPattern, saas.WorkPatternDay,
        saas.WorkPatternDayAllowedShift, saas.StaffPatternAssignment,
        saas.StaffRule, saas.BankHoliday,
    )


def append_only_audit_models(saas: Any, operational: Any, briefing: Any) -> tuple[Any, ...]:
    """Return audit-evidence records that cannot be changed or deleted."""
    return (
        operational.SmsAudit,
        operational.RequestAudit,
        operational.AnnotationAudit,
        operational.ChangeLog,
        saas.SuperAdminAudit,
        saas.CentralSecurityAudit,
        saas.PositionSessionAudit,
        briefing.BriefingAudit,
        saas.ToilTransaction,
    )
