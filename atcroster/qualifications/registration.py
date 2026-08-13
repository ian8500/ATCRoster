"""Qualification and compliance blueprint composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fatigue_compliance import (
    FatigueComplianceDependencies,
    create_fatigue_compliance_blueprint,
)
from atcroster.live_position.currency import (
    OperationalCurrencyDependencies,
    create_operational_currency_blueprint,
)

from .blueprint import QualificationDependencies, create_qualification_blueprint


@dataclass(frozen=True)
class QualificationRegistrationDependencies:
    db: Any
    Unit: Any
    Staff: Any
    QualificationType: Any
    PersonQualification: Any
    PersonQualificationHistory: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    is_editor_user: Callable[[Any], bool]
    is_admin_user: Callable[[Any], bool]
    now: Callable[[], Any]
    qualification_impact_type: Callable[..., Any]
    person_has_other_valid_ue: Callable[..., bool]
    record_roster_impact: Callable[..., Any]
    validate_csrf: Callable[[], None]
    load_rule_config: Callable[..., Any]
    save_rule_config: Callable[..., Any]
    live_position_enabled: Callable[[int], bool]
    currency_requirement: Callable[..., dict[str, Any]]
    save_currency_requirement: Callable[[dict[str, Any]], None]
    currency_shortfalls: Callable[[int], dict[str, Any]]


def register_qualification_blueprints(
    app: Any, deps: QualificationRegistrationDependencies
) -> None:
    app.register_blueprint(create_qualification_blueprint(QualificationDependencies(
        db=deps.db, Staff=deps.Staff,
        QualificationType=deps.QualificationType,
        PersonQualification=deps.PersonQualification,
        PersonQualificationHistory=deps.PersonQualificationHistory,
        RosterImpactEventType=deps.RosterImpactEventType,
        current_unit_id=deps.current_unit_id,
        is_editor_user=deps.is_editor_user, is_admin_user=deps.is_admin_user,
        now=deps.now, qualification_impact_type=deps.qualification_impact_type,
        person_has_other_valid_ue=deps.person_has_other_valid_ue,
        record_roster_impact=deps.record_roster_impact,
    )))
    app.register_blueprint(create_fatigue_compliance_blueprint(
        FatigueComplianceDependencies(
            db=deps.db, Unit=deps.Unit, is_admin_user=deps.is_admin_user,
            current_unit_id=deps.current_unit_id,
            validate_csrf=deps.validate_csrf,
            load_rule_config=deps.load_rule_config,
            save_rule_config=deps.save_rule_config,
        )
    ))
    app.register_blueprint(create_operational_currency_blueprint(
        OperationalCurrencyDependencies(
            db=deps.db, current_unit_id=deps.current_unit_id,
            is_admin_user=deps.is_admin_user,
            live_position_enabled=deps.live_position_enabled,
            currency_requirement=deps.currency_requirement,
            save_currency_requirement=deps.save_currency_requirement,
            currency_shortfalls=deps.currency_shortfalls,
            validate_csrf=deps.validate_csrf,
        )
    ))
