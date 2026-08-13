"""Authentication and MFA blueprint composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from auth_blueprint import AuthDependencies, create_auth_blueprint

from .mfa_blueprint import MfaRouteDependencies, create_mfa_blueprint


@dataclass(frozen=True)
class AuthRegistrationDependencies:
    db: Any
    PlatformIdentity: Any
    UnitMembership: Any
    DatabaseRoutingMetadata: Any
    Staff: Any
    Unit: Any
    PlatformMfaCredential: Any
    MfaCredential: Any
    deployment_environment: str
    validate_csrf: Callable[[], None]
    normalized_login: Callable[[str], str]
    login_rate_key: Callable[..., str]
    consume_rate_limit: Callable[..., Any]
    reset_rate_limit: Callable[..., Any]
    security_event: Callable[..., Any]
    central_security_event: Callable[..., Any]
    bind_authenticated_unit: Callable[..., Any]
    canonical_login_redirect: Callable[..., Any]
    airport_login_endpoint: Callable[..., str]
    initialize_authenticated_session: Callable[..., Any]
    record_successful_login: Callable[..., Any]
    decrypt_secret: Callable[..., Any]
    matching_totp_step: Callable[..., Any]
    encrypt_field: Callable[..., Any]
    now: Callable[[], Any]
    current_unit_id: Callable[[], int]
    current_auth_stamp: Callable[..., Any]
    totp_qr_data_uri: Callable[..., str]


def create_auth_registration_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> AuthRegistrationDependencies:
    """Bind authentication records at the authentication boundary."""
    return AuthRegistrationDependencies(
        db=db,
        PlatformIdentity=saas_models.PlatformIdentity,
        UnitMembership=saas_models.UnitMembership,
        DatabaseRoutingMetadata=saas_models.DatabaseRoutingMetadata,
        Staff=operational_models.Staff,
        Unit=operational_models.Unit,
        PlatformMfaCredential=saas_models.PlatformMfaCredential,
        MfaCredential=saas_models.MfaCredential,
        **services,
    )


def register_auth_blueprints(app: Any, deps: AuthRegistrationDependencies) -> None:
    app.register_blueprint(create_auth_blueprint(AuthDependencies(
        db=deps.db, PlatformIdentity=deps.PlatformIdentity,
        UnitMembership=deps.UnitMembership,
        DatabaseRoutingMetadata=deps.DatabaseRoutingMetadata,
        Staff=deps.Staff, Unit=deps.Unit,
        PlatformMfaCredential=deps.PlatformMfaCredential,
        MfaCredential=deps.MfaCredential, validate_csrf=deps.validate_csrf,
        normalized_login=deps.normalized_login, login_rate_key=deps.login_rate_key,
        consume_rate_limit=deps.consume_rate_limit,
        reset_rate_limit=deps.reset_rate_limit, security_event=deps.security_event,
        central_security_event=deps.central_security_event,
        bind_authenticated_unit=deps.bind_authenticated_unit,
        canonical_login_redirect=deps.canonical_login_redirect,
        airport_login_endpoint=deps.airport_login_endpoint,
        initialize_authenticated_session=deps.initialize_authenticated_session,
        record_successful_login=deps.record_successful_login,
    )))
    app.register_blueprint(create_mfa_blueprint(MfaRouteDependencies(
        db=deps.db, PlatformIdentity=deps.PlatformIdentity,
        PlatformMfaCredential=deps.PlatformMfaCredential, Staff=deps.Staff,
        MfaCredential=deps.MfaCredential,
        DatabaseRoutingMetadata=deps.DatabaseRoutingMetadata,
        deployment_environment=deps.deployment_environment,
        validate_csrf=deps.validate_csrf,
        consume_rate_limit=deps.consume_rate_limit,
        decrypt_secret=deps.decrypt_secret,
        matching_totp_step=deps.matching_totp_step,
        encrypt_field=deps.encrypt_field, now=deps.now,
        central_security_event=deps.central_security_event,
        bind_authenticated_unit=deps.bind_authenticated_unit,
        initialize_authenticated_session=deps.initialize_authenticated_session,
        security_event=deps.security_event,
        record_successful_login=deps.record_successful_login,
        canonical_login_redirect=deps.canonical_login_redirect,
        current_unit_id=deps.current_unit_id,
        current_auth_stamp=deps.current_auth_stamp,
        totp_qr_data_uri=deps.totp_qr_data_uri,
    )))
