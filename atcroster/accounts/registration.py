"""Account-domain blueprint composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .kiosk import KioskAccountDependencies, create_kiosk_account_blueprint
from .invitations import (
    InvitationAcceptanceDependencies,
    create_invitation_acceptance_blueprint,
)
from .password import PasswordDependencies, create_password_blueprint
from .profile import StaffProfileDependencies, create_staff_profile_blueprint
from .recovery_blueprint import (
    RecoveryRequestDependencies,
    create_recovery_request_blueprint,
)
from .unit_accounts import UnitAccountsDependencies, create_unit_accounts_blueprint


@dataclass(frozen=True)
class AccountRegistrationDependencies:
    db: Any
    Unit: Any
    Staff: Any
    PlatformIdentity: Any
    UnitMembership: Any
    SecureInvitation: Any
    RecoveryRequest: Any
    DatabaseRoutingMetadata: Any
    SmsSenderRegistration: Any
    MfaCredential: Any
    Assignment: Any
    Notification: Any
    deployment_environment: str
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    is_editor_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    consume_rate_limit: Callable[..., Any]
    valid_email: Callable[[str], str]
    normalized_login: Callable[[str], str]
    platform_support_emails: Callable[[], list[str]]
    unit_admin_emails: Callable[[int], list[str]]
    send_email: Callable[..., bool]
    now: Callable[[], Any]
    active_recovery: Callable[..., Any]
    bind_authenticated_unit: Callable[..., Any]
    generate_password_hash: Callable[..., str]
    tenant_get: Callable[..., Any]
    run_signup: Callable[..., Any]
    signup_error: type[Exception]
    normalise_uk_mobile: Callable[[str | None], str]
    normalise_phone: Callable[[str | None], str]
    qr_data_uri: Callable[..., str]
    absence_types: Callable[..., Any]
    month_range: Callable[..., Any]
    get_shift: Callable[..., Any]
    shift_duration_minutes: Callable[[Any], int]
    live_position_enabled: Callable[[int], bool]


def register_account_blueprints(app: Any, deps: AccountRegistrationDependencies) -> None:
    app.register_blueprint(create_recovery_request_blueprint(
        RecoveryRequestDependencies(
            db=deps.db, PlatformIdentity=deps.PlatformIdentity,
            UnitMembership=deps.UnitMembership, RecoveryRequest=deps.RecoveryRequest,
            Unit=deps.Unit, Staff=deps.Staff,
            DatabaseRoutingMetadata=deps.DatabaseRoutingMetadata,
            validate_csrf=deps.validate_csrf,
            consume_rate_limit=deps.consume_rate_limit,
            valid_email=deps.valid_email, normalized_login=deps.normalized_login,
            platform_support_emails=deps.platform_support_emails,
            unit_admin_emails=deps.unit_admin_emails, send_email=deps.send_email,
            now=deps.now, active_recovery=deps.active_recovery,
            is_admin_user=deps.is_admin_user,
            bind_authenticated_unit=deps.bind_authenticated_unit,
            generate_password_hash=deps.generate_password_hash,
        )
    ))
    app.register_blueprint(create_unit_accounts_blueprint(UnitAccountsDependencies(
        db=deps.db, Unit=deps.Unit, Staff=deps.Staff,
        PlatformIdentity=deps.PlatformIdentity, UnitMembership=deps.UnitMembership,
        SecureInvitation=deps.SecureInvitation,
        current_unit_id=deps.current_unit_id, is_admin_user=deps.is_admin_user,
        validate_csrf=deps.validate_csrf, normalized_login=deps.normalized_login,
        now=deps.now, tenant_get=deps.tenant_get,
    )))
    app.register_blueprint(create_invitation_acceptance_blueprint(
        InvitationAcceptanceDependencies(
            db=deps.db, SecureInvitation=deps.SecureInvitation, Unit=deps.Unit,
            DatabaseRoutingMetadata=deps.DatabaseRoutingMetadata, Staff=deps.Staff,
            deployment_environment=deps.deployment_environment,
            consume_rate_limit=deps.consume_rate_limit, now=deps.now,
            bind_authenticated_unit=deps.bind_authenticated_unit,
            validate_csrf=deps.validate_csrf, valid_email=deps.valid_email,
            run_signup=deps.run_signup, signup_error=deps.signup_error,
        )
    ))
    app.register_blueprint(create_staff_profile_blueprint(StaffProfileDependencies(
        db=deps.db, Staff=deps.Staff, UnitMembership=deps.UnitMembership,
        PlatformIdentity=deps.PlatformIdentity,
        SmsSenderRegistration=deps.SmsSenderRegistration,
        MfaCredential=deps.MfaCredential, Assignment=deps.Assignment,
        Notification=deps.Notification, current_unit_id=deps.current_unit_id,
        is_editor_user=deps.is_editor_user, validate_csrf=deps.validate_csrf,
        normalise_uk_mobile=deps.normalise_uk_mobile,
        valid_email=deps.valid_email, normalise_phone=deps.normalise_phone,
        now=deps.now, qr_data_uri=deps.qr_data_uri,
        absence_types=deps.absence_types, month_range=deps.month_range,
        get_shift=deps.get_shift, shift_duration_minutes=deps.shift_duration_minutes,
    )))
    app.register_blueprint(create_password_blueprint(PasswordDependencies(
        db=deps.db, Staff=deps.Staff, PlatformIdentity=deps.PlatformIdentity,
        tenant_get=deps.tenant_get, validate_csrf=deps.validate_csrf,
        generate_password_hash=deps.generate_password_hash,
    )))
    app.register_blueprint(create_kiosk_account_blueprint(KioskAccountDependencies(
        db=deps.db, Unit=deps.Unit, Staff=deps.Staff,
        UnitMembership=deps.UnitMembership, SecureInvitation=deps.SecureInvitation,
        current_unit_id=deps.current_unit_id,
        live_position_enabled=deps.live_position_enabled,
        tenant_get=deps.tenant_get, utcnow=deps.now,
        validate_csrf=deps.validate_csrf, is_admin_user=deps.is_admin_user,
    )))
