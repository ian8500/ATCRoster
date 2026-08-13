"""Application assembly and legacy compatibility implementation."""

from functools import wraps
from collections import defaultdict, OrderedDict
from typing import Any, Optional, Tuple
from flask import render_template, request, redirect, url_for, flash, abort, session, g
import os
import re
import sys
from functools import lru_cache
from datetime import date, datetime, time, timedelta
import json
import hashlib

from flask_login import (
    LoginManager, login_user, logout_user,
    current_user, login_required
)
from werkzeug.security import generate_password_hash
from sqlalchemy import event, inspect as sa_inspect
from sqlalchemy.orm import Session as OrmSession, with_loader_criteria
from saas_models import register_saas_models
from fatigue_engine import (
    _analyze_segments,
    _custom_fatigue_flags,
    _is_early_start,
    _is_morning_duty,
    _is_night_duty,
    _is_working,
    _span,
)
from rate_limiting import (
    LimiterUnavailable, MemoryRateLimiter, RedisRateLimiter, privacy_key,
)
from fairness_service import FairnessAssignment, FairnessStaff, calculate_fairness
from reporting import (
    compute_annotation_metrics,
    current_leave_year_window,
    financial_year_start,
    group_consecutive_days,
    leave_summary_for_month,
)
from roster_logic import (
    add_months,
    daily_requirements,
    expand_pattern,
    iter_year_months,
    month_days,
    normalise_assignment_snapshot,
    parse_year_month,
    roster_lock_date,
    roster_month_is_locked,
    shift_minutes,
    validated_pattern,
)
from toil_service import apply_toil_transaction
from production_operations import (
    MetricsRegistry,
    begin_request,
    configure_production_logging,
    finish_request,
    register_operations_routes,
    structured_event,
)
from absence_requests import (
    add_months as add_request_months,
    group_sickness_instances,
    normalise_request_rules,
    request_date_bounds,
    request_lock_date,
    request_month_is_locked,
    safe_admin_month,
)
from fatigue_compliance import (
    FatigueComplianceDependencies,
    FatigueRuleConfigDependencies,
    FatigueRuleConfigService,
    compliance_month,
    create_fatigue_compliance_blueprint,
)
from work_pattern_service import WorkPatternDependencies, WorkPatternService
from roster_population_service import (
    DeterministicRosterPopulationService,
    PopulationDependencies,
)
from roster_impact_service import (
    RosterImpactDependencies,
    RosterImpactEventType,
    RosterImpactService,
)
from operational_capability import (
    OperationalCapabilityDependencies,
    OperationalCapabilityService,
)
from roster_month_cache import RosterMonthCache
from atcroster.roster.defaults import (
    DEFAULT_ABSENCE_TYPES,
    DEFAULT_ANNOTATION_TYPES,
    DEFAULT_BANNED_ROSTER_CODES,
    DEFAULT_EXCLUDE_FROM_COUNTERS,
    DEFAULT_NON_WORKING_CODES,
    DEFAULT_OPERATIONAL_CURRENCY_REQUIREMENT,
    DEFAULT_ROSTER_SETTINGS,
    DEFAULT_WORKING_CODES,
    MIN_MONTH,
    OPERATIONAL_CURRENCY_SETTING_KEY,
)
from atcroster.roster import (
    invalidate_month_for_day, is_month_locked as roster_period_is_locked,
    lock_date_for_month as roster_period_lock_date, memoize,
    month_add as roster_period_add, parse_annotation as parse_roster_annotation,
    month_has_data as roster_month_has_data,
    lock_roster_month as lock_roster_period,
    shift_groups_snapshot,
    expand as expand_roster_pattern, validate as validate_roster_pattern,
    parse_hhmm as parse_roster_hhmm, parse_iso_date as parse_roster_date,
    is_sunday as roster_date_is_sunday,
    parse_year_month as parse_roster_year_month,
    duration_minutes as roster_shift_duration_minutes,
    ensure_month_requirement as ensure_roster_month_requirement,
    requirements_for_day as resolve_roster_requirements_for_day,
    cell_is_protected,
    assignment_for_day,
    is_non_working as roster_code_is_non_working,
    normalize_code as normalize_roster_code,
    is_working_with_prefix as roster_code_is_working_with_prefix,
    set_assignment_code,
)
from atcroster.roster.shifts import save_counter_mapping
from atcroster.roster.requirements import (
    delete_special_requirement,
    save_monthly_requirements,
    save_special_requirement,
)
from atcroster.compression import register_response_compression
from access_policy import (
    has_permission,
    is_admin,
    is_editor,
    is_trainee,
    may_apply_annotations,
    may_edit_roster,
    may_manage_training,
    may_override_roster_conflicts,
    may_record_training,
    may_send_unit_messages,
    permissions_for,
)
from atcroster import create_app, get_runtime_settings
from atcroster.clock import utcnow
from atcroster.auth import (
    decrypt_secret,
    matching_totp_step,
    consume_rate_limit,
    canonical_login_redirect,
    airport_login_endpoint,
    credential_for_auth_stamp,
    privacy_rate_limit_key,
    record_security_event,
    reset_rate_limit,
    totp_qr_data_uri,
)
from atcroster.auth.mfa_blueprint import MfaRouteDependencies, create_mfa_blueprint
from atcroster.qualifications import (
    QualificationDependencies,
    create_qualification_blueprint,
    currency_window,
    classify_qualification_impact,
    has_other_valid_ue,
    has_valid_endorsement,
    load_currency_requirement,
    monthly_compliance_findings,
    monthly_position_assurance,
    minutes_between as calculate_minutes_between,
    operational_currency_shortfalls,
    qualification_snapshot,
    record_qualification_history as add_qualification_history,
    record_roster_impact_for_qualification,
    staff_has_qualification as qualification_status_for_staff,
    sync_legacy_roster_profile,
)
from atcroster.audit import context_month_for_date, record_central_security_event, record_change
from atcroster.workforce import effective_watch as resolve_effective_watch, has_leave_or_sickness, watch_id_for_staff_on as resolve_watch_id, watch_ids_for_staff_on as resolve_watch_ids
from atcroster.workforce.joiners import JoinerDependencies, create_joiner
from atcroster.fatigue import (
    assignment_is_fatigue_safe,
    configured_findings as configured_fatigue_findings,
    new_findings_for_proposed_assignment,
    segments_from_assignments,
)
from atcroster.errors import ErrorHandlerDependencies, register_error_handlers
from atcroster.extensions import create_tenant_database
from atcroster.public import public_blueprint
from atcroster.notifications import (
    NotificationDependencies,
    create_notification_blueprint,
    normalise_sms_number,
    normalise_uk_mobile,
    parse_sms_number_lines,
    send_via_messagemedia,
    email_service_configured,
    send_account_email,
    valid_email,
    SmsConfigurationService,
    SmsAuditService,
    OvertimeSmsService,
    default_overtime_sms_body,
    SmsAdministrationDependencies,
    create_sms_administration_blueprint,
    MessagingDependencies,
    create_messaging_blueprint,
)
from atcroster.notifications.configuration import save_sms_settings
from atcroster.roster.publication import (
    PublicationDependencies,
    create_publication_service,
)
from atcroster.roster.overtime import (
    OvertimeCandidateDependencies,
    OvertimeCandidateService,
    OvertimeDependencies,
    count_tagged_assignments,
    create_overtime_blueprint,
    had_sickness_within_48_hours,
    has_in_date_endorsement,
    worked_like_consecutive_days,
)
from atcroster.roster.setup import update_unit_roster_setup
from atcroster.roster.watch_configuration import (
    WatchConfigurationDependencies,
    update_watch_configuration,
)
from atcroster.roster.shift_configuration import (
    ShiftConfigurationDependencies,
    update_shift_definition,
)
from atcroster.roster.bootstrap import (
    ensure_shift as ensure_bootstrap_shift,
    ensure_watch as ensure_bootstrap_watch,
    seed_legacy_operational_data,
)
from atcroster.roster.reference_data import (
    bootstrap_reference_data as bootstrap_roster_reference_data,
)
from atcroster.roster.fairness import (
    FairnessDependencies,
    FairnessReportService,
)
from atcroster.roster.month_view import (
    MonthRosterLoadDependencies,
    load_month_roster,
)
from atcroster.roster.assignments import (
    AssignmentRefreshDependencies,
    allocate_day_shift_shortfall,
    generate_assignment_range,
    generate_month_assignments,
    refresh_pattern_day,
    set_absence_override,
    set_generated_assignment,
)
from atcroster.cli import CliDependencies, create_cli_commands
from atcroster.cli_roster import RosterCliDependencies, create_roster_cli
from atcroster.modules import ModuleDependencies, create_module_blueprint
from atcroster.calendar_feed import CalendarFeedDependencies, create_calendar_feed_blueprint
from atcroster.administration import (
    AdministrationDependencies,
    ToilAdministrationDependencies,
    create_administration_blueprint,
    create_toil_administration_blueprint,
    seed_toil_balances,
    annotation_accrual_half_days,
    apply_annotation_toil_delta,
    accrued_and_used_half_days,
)
from atcroster.administration.actions import (
    AdminActionDependencies,
    dispatch_admin_action,
)
from atcroster.administration.onboarding import (
    OnboardingDependencies,
    create_onboarding_blueprint,
)
from atcroster.administration.reference import (
    ReferenceDataDependencies,
    create_reference_data_blueprint,
)
from atcroster.administration.lifecycle import (
    StaffLifecycleDependencies,
    create_staff_lifecycle_blueprint,
)
from atcroster.administration.watch_moves import (
    WatchMoveDependencies,
    create_watch_move_blueprint,
)
from atcroster.administration.absence_types import update_absence_types
from atcroster.administration.context import (
    AdminContextDependencies,
    build_admin_context,
)
from atcroster.administration.staff_edit import (
    StaffEditDependencies,
    create_staff_edit_blueprint,
)
from atcroster.home import HomeDependencies, create_home_blueprint
from atcroster.navigation import (
    NavigationContextDependencies,
    build_navigation_context,
)
from atcroster.accounts import (
    KioskAccountDependencies,
    PasswordDependencies,
    create_kiosk_account_blueprint,
    create_password_blueprint,
    active_recovery_from_digest,
    platform_support_emails,
    normalise_phone_number,
    record_successful_login,
    unit_admin_emails,
    RecoveryRequestDependencies,
    create_recovery_request_blueprint,
)
from atcroster.accounts.unit_accounts import (
    UnitAccountsDependencies,
    create_unit_accounts_blueprint,
)
from atcroster.accounts.invitations import (
    InvitationAcceptanceDependencies,
    create_invitation_acceptance_blueprint,
)
from atcroster.accounts.profile import (
    StaffProfileDependencies,
    create_staff_profile_blueprint,
)
from atcroster.accounts.signup import (
    SignupSagaDependencies,
    SignupWorkflowError,
    normalized_login,
    run_invitation_signup,
)
from atcroster.admin_utilities import AdminUtilityDependencies, create_admin_utility_blueprint
from atcroster.platform import (
    WorkerHealthDependencies,
    add_assignment_annotation,
    add_columns_if_missing,
    add_performance_indexes,
    add_role_and_calendar_token,
    add_unique_assignment_key,
    add_invitation_target,
    add_toil_and_leave_fields,
    add_watch_pattern_configuration,
    create_worker_health_blueprint,
    upgrade_tenant_foundation,
)
from atcroster.platform.admin import (
    PlatformAdminDependencies,
    create_platform_admin_blueprint,
)
from atcroster.live_position import (
    OperationalCurrencyDependencies,
    create_operational_currency_blueprint,
)
from atcroster.security.csrf import register_csrf_protection
from atcroster.security.encryption import FieldEncryptionService
from atcroster.security.headers import (
    SecurityHeaderDependencies,
    register_security_headers,
)
from atcroster.security.sessions import (
    SessionLifecycle,
    SessionLifecycleDependencies,
)
from atcroster.tenancy_hooks import TenantHookDependencies, register_tenant_hooks
from atcroster.briefing_bootstrap import load_briefing_module
from migrations.fresh_schema import CONTROL_TABLES
from auth_blueprint import AuthDependencies, create_auth_blueprint
from absence_requests_blueprint import (
    AbsenceRequestDependencies,
    create_absence_requests_blueprint,
)
from reports_blueprint import ReportsDependencies, create_reports_blueprint
from roster_blueprint import RosterDependencies, create_roster_blueprint
from training_blueprint import TrainingDependencies, create_training_blueprint
from operations_blueprint import OperationsDependencies, create_operations_blueprint
from live_position_blueprint import LivePositionDependencies, create_live_position_blueprint
from handover_blueprint import HandoverDependencies, create_handover_blueprint
from work_pattern_admin_service import WorkPatternAdminDependencies, WorkPatternAdminService
from work_pattern_blueprint import WorkPatternBlueprintDependencies, create_work_pattern_blueprint
from roster_validation_service import RosterValidationDependencies, RosterValidationService
from roster_proposal_service import RosterProposalDependencies, RosterProposalService
from work_pattern_migration_service import WorkPatternMigrationDependencies, WorkPatternMigrationService
from override_classification_service import OverrideClassificationDependencies, OverrideClassificationService
from roster_period_service import RosterPeriodDependencies, RosterPeriodService
from tenancy import (
    authenticated_database_route_optional,
    authenticated_unit_id,
    authenticated_unit_context,
    bind_authenticated_unit,
    bind_platform_control,
    clear_request_context,
    operational_unit_context,
    reset_authenticated_unit,
    reset_platform_control,
)

try:
    from flask_caching import Cache
except Exception:
    Cache = None

# -------------------- App setup --------------------
app = create_app()
_runtime_settings = get_runtime_settings(app)
configure_production_logging(app, _runtime_settings.deployment_environment)
_operational_metrics = MetricsRegistry()

# Writable local instance folder for development and tests.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = app.instance_path
os.makedirs(INSTANCE_DIR, exist_ok=True)

DEPLOYMENT_ENV = _runtime_settings.deployment_environment
FIELD_ENCRYPTION_KEY = _runtime_settings.field_encryption_key
FIELD_ENCRYPTION_KEYS = _runtime_settings.field_encryption_keys

if DEPLOYMENT_ENV == "production":
    import redis
    _rate_limiter = RedisRateLimiter(
        redis.from_url(
            os.environ["REDIS_URL"], socket_connect_timeout=2,
            socket_timeout=2, decode_responses=True,
        ),
        prefix=f"atcroster:{DEPLOYMENT_ENV}:limit",
    )
    _rate_limiter.verify()
else:
    _rate_limiter = MemoryRateLimiter()


# Constructing the service validates configured material during startup rather
# than at first MFA use. Aliases preserve callers during incremental extraction.
_field_encryption = FieldEncryptionService(FIELD_ENCRYPTION_KEYS)
_field_ciphers = _field_encryption.ciphers
_encrypt_field = _field_encryption.encrypt
_decrypt_field = _field_encryption.decrypt

# Jinja helper
app.jinja_env.globals['now'] = lambda: datetime.now()


@lru_cache(maxsize=256)
def _asset_version(filename: str) -> Optional[int]:
    """Return a process-stable static version without a stat per template use."""
    static_folder = app.static_folder
    if not static_folder:
        return None
    try:
        return int(os.path.getmtime(os.path.join(static_folder, filename)))
    except (OSError, TypeError, ValueError):
        return None


def _asset_url(filename: str, **extra: object) -> str:
    """Return a cache-busting static asset URL using the file mtime."""

    version = _asset_version(filename)

    if version is not None:
        return url_for("static", filename=filename, v=version, **extra)

    return url_for("static", filename=filename, **extra)


app.jinja_env.globals["asset_url"] = _asset_url


REQUEST_STATUSES = frozenset({"pending", "approved", "rejected", "fulfilled", "cancelled"})
REQUEST_TRANSITIONS = {
    "pending": frozenset({"approved", "rejected", "cancelled"}),
    "approved": frozenset({"rejected", "cancelled"}),
    "rejected": frozenset(),
    "cancelled": frozenset(),
    "fulfilled": frozenset(),
}
PLATFORM_FEATURE_FLAGS = frozenset({
    "advanced_coverage", "scenario_planning", "calendar_exports",
    "fatigue_reporting", "custom_branding", "briefing_module",
    "training_module", "competency_module", "live_position_monitoring",
    "handover_module",
})

# The platform feature registry also contains supporting capabilities. Keep
# Super Admin's product controls focused on the modules an airport can launch.
PLATFORM_MODULE_FLAGS = frozenset({
    "briefing_module", "training_module", "competency_module",
    "live_position_monitoring", "handover_module",
})


def _current_unit_id() -> int:
    """Derive tenancy from the authenticated membership, never request data."""
    return int(getattr(current_user, "unit_id", 0) or 0)


@app.before_request
def _start_request_tenant_boundary():
    clear_request_context()
    g.metrics_started_at = begin_request(_operational_metrics)


_validate_csrf, _enforce_csrf = register_csrf_protection(app)


app.register_blueprint(public_blueprint)


_error_handlers = register_error_handlers(
    app,
    ErrorHandlerDependencies(
        security_event=lambda event, **safe_fields: _security_event(
            event, **safe_fields
        )
    ),
)
_bad_request = _error_handlers[400]
_forbidden = _error_handlers[403]
_not_found = _error_handlers[404]
_internal_error = _error_handlers[500]

# Database & login
db = create_tenant_database(app, DEPLOYMENT_ENV)
login_manager = LoginManager(app)
login_manager.login_view = "login"


_bind_tenant_context, _reset_tenant_context = register_tenant_hooks(
    app,
    TenantHookDependencies(
        deployment_environment=DEPLOYMENT_ENV,
        current_user=lambda: current_user,
        enforce_session=lambda user: _session_lifecycle.enforce_request(user),
        routing_for_unit=lambda unit_id: db.session.get(
            DatabaseRoutingMetadata, unit_id
        ),
        clear_context=clear_request_context,
        bind_authenticated_unit=bind_authenticated_unit,
        reset_authenticated_unit=reset_authenticated_unit,
        bind_platform_control=bind_platform_control,
        reset_platform_control=reset_platform_control,
    ),
)


@app.before_request
def _enforce_principal_boundaries():
    if (
        current_user.is_authenticated
        and getattr(current_user, "role", "") == "superadmin"
    ):
        if not session.get("_platform_mfa_verified"):
            logout_user()
            session.clear()
            return redirect(url_for("login"))
        allowed_platform_endpoints = {
            "platform_admin", "logout", "password_change",
            "platform_worker_health",
            "internal_metrics", "internal_health",
            "static", "favicon", "health_live", "health_ready",
        }
        if request.endpoint == "index":
            return redirect(url_for("platform_admin"))
        if request.endpoint not in allowed_platform_endpoints:
            abort(403)
    if (
        current_user.is_authenticated
        and getattr(current_user, "role", "") == "position_monitor"
    ):
        allowed_kiosk_endpoints = {
            "live_position.kiosk_hmi", "live_position.live_state",
            "live_position.controllers", "live_position.open_position",
            "live_position.live_events",
            "live_position.close_position",
            "live_position.logon", "live_position.logoff",
            "live_position.handover", "live_position.add_participant",
            "live_position.remove_participant",
            "logout", "static", "favicon", "health_live", "health_ready",
        }
        if request.endpoint not in allowed_kiosk_endpoints:
            if request.method == "GET":
                return redirect(url_for("live_position.kiosk_hmi"))
            abort(403)
    if (
        current_user.is_authenticated
        and getattr(current_user, "role", "") != "superadmin"
        and getattr(current_user, "role", "") != "position_monitor"
        and (
            DEPLOYMENT_ENV == "production"
            or UnitMembership.query.filter_by(
                person_id=current_user.id,
                unit_id=getattr(current_user, "unit_id", 0),
                role="UnitAdmin",
                status="active",
            ).first() is not None
        )
        and request.endpoint not in {
            "mfa_setup", "logout", "static", "favicon",
            "health_live", "health_ready",
        }
    ):
        credential = MfaCredential.query.filter_by(
            person_id=current_user.id, enabled=True
        ).first()
        if not credential:
            return redirect(url_for("mfa_setup"))


_security_headers = register_security_headers(
    app,
    SecurityHeaderDependencies(
        deployment_environment=DEPLOYMENT_ENV,
        metrics=_operational_metrics,
        finish_request=finish_request,
        slow_roster_seconds=float(os.environ.get(
            "ATCROSTER_SLOW_ROSTER_SECONDS", "2.0"
        )),
    ),
)
register_response_compression(app)


def _canonical_login_redirect(
    target: str | None,
    *,
    default_endpoint: str = "index",
    user_id: int | None = None,
) -> str:
    return canonical_login_redirect(target, url_for=url_for, default_endpoint=default_endpoint, user_id=user_id)


def _airport_login_endpoint(user) -> str:
    return airport_login_endpoint(user)


# ----- Lightweight caching -----
_cache = None
if Cache is not None:
    try:
        _cache = Cache(config={
            "CACHE_TYPE": "simple",            # in-memory
            "CACHE_DEFAULT_TIMEOUT": 120
        })
        _cache.init_app(app)
    except Exception:
        _cache = None


def _memoize(seconds=60):
    return memoize(_cache, seconds)


def _invalidate_month_cache_for_day(d: date):
    invalidate_month_for_day(
        _cache, _load_month_roster_fast, _current_unit_id(), d,
    )


def _messagemedia_credentials() -> tuple[str, str, str]:
    from atcroster.notifications.sms import messagemedia_credentials
    return messagemedia_credentials()


def _normalise_sms_number(value: str | None) -> str:
    return normalise_sms_number(value)


def _normalise_uk_mobile(value: str | None) -> str:
    return normalise_uk_mobile(value)


def _sms_number_options(key: str, unit_id: int | None = None) -> list[dict[str, str]]:
    return sms_configuration.number_options(key, unit_id)


def _sms_sender_options(unit_id: int | None = None) -> list[dict[str, str]]:
    return sms_configuration.sender_options(unit_id)


def _sms_operational_options(unit_id: int | None = None) -> list[dict[str, str]]:
    return sms_configuration.operational_options(unit_id)


def _sms_default_number(
    setting_key: str, options: list[dict[str, str]], unit_id: int | None = None
) -> str:
    return sms_configuration.default_number(setting_key, options, unit_id)


def _sms_service_configured() -> bool:
    return sms_configuration.service_configured()


def _email_service_configured() -> bool:
    return email_service_configured()


def _send_account_email(to_address: str, subject: str, body: str) -> bool:
    return send_account_email(to_address, subject, body, app.logger)


def _valid_email(value: str) -> str:
    return valid_email(value)


def _platform_support_emails() -> list[str]:
    return platform_support_emails(
        PlatformIdentity,
        os.getenv("ATCROSTER_SUPPORT_EMAIL", ""),
        _valid_email,
    )


def _unit_admin_emails(unit_id: int) -> list[str]:
    return unit_admin_emails(db, PlatformIdentity, UnitMembership, unit_id)


def _send_sms_via_messagemedia(
    to_number: str, body: str, from_number: str | None = None,
) -> tuple[bool, str]:
    return send_via_messagemedia(to_number, body, from_number)


def _send_sms(to_number: str, body: str, from_number: str | None = None) -> tuple[bool, str]:
    return _send_sms_via_messagemedia(to_number, body, from_number)


def _record_sms_audit(
    *,
    sender_number: str,
    recipient_number: str,
    recipient_label: str,
    body: str,
    message_type: str,
    provider_message_id: str,
    delivery_status: str = "submitted",
) -> None:
    sms_audit_service.record(
        sender_number=sender_number,
        recipient_number=recipient_number,
        recipient_label=recipient_label,
        body=body,
        message_type=message_type,
        provider_message_id=provider_message_id,
        delivery_status=delivery_status,
    )


def _send_overtime_sms_notifications(
    staff_list: list["Staff"], message: str
) -> tuple[int, list[tuple[Optional["Staff"], str]]]:
    return overtime_sms_service.notify(staff_list, message)


def _default_overtime_sms_body(chosen_date: date | None, shift_code: str | None) -> str:
    return default_overtime_sms_body(chosen_date, shift_code)


def _flash_sms_result(
    sent: int, failures: list[tuple[Optional["Staff"], str]]
) -> None:
    if sent:
        flash(f"SMS sent to {sent} recipient{'s' if sent != 1 else ''}.", "ok")
    if failures:
        details = "; ".join(
            f"{staff.name if staff else 'System'}: {reason}"
            for staff, reason in failures[:8]
        )
        if len(failures) > 8:
            details += f"; and {len(failures) - 8} more"
        flash(f"Some messages were not sent. {details}", "error")


def _load_operational_models():
    """Load model declarations after the canonical database extension exists."""
    from atcroster.models import operational

    return operational


_operational_models = _load_operational_models()
AnnotationAudit = _operational_models.AnnotationAudit
AnnotationType = _operational_models.AnnotationType
Assignment = _operational_models.Assignment
ChangeLog = _operational_models.ChangeLog
Leave = _operational_models.Leave
Notification = _operational_models.Notification
RequestAudit = _operational_models.RequestAudit
Requirement = _operational_models.Requirement
RosterSetting = _operational_models.RosterSetting
ShiftRequest = _operational_models.ShiftRequest
ShiftType = _operational_models.ShiftType
Sickness = _operational_models.Sickness
SmsAudit = _operational_models.SmsAudit
SmsSenderRegistration = _operational_models.SmsSenderRegistration
SpecialRequirement = _operational_models.SpecialRequirement
Staff = _operational_models.Staff
StaffWatchHistory = _operational_models.StaffWatchHistory
TrainingLevel = _operational_models.TrainingLevel
TrainingObjective = _operational_models.TrainingObjective
TrainingScore = _operational_models.TrainingScore
TrainingSession = _operational_models.TrainingSession
Unit = _operational_models.Unit
Watch = _operational_models.Watch
HandoverEquipment = _operational_models.HandoverEquipment
HandoverField = _operational_models.HandoverField
HandoverOperationalState = _operational_models.HandoverOperationalState
HandoverRecord = _operational_models.HandoverRecord
# Control-plane and advanced product entities live in a separate module so
# they can move to the central database without rewriting the legacy UI.
SaaS = register_saas_models(db, utcnow)
PlatformIdentity = SaaS.PlatformIdentity
PlatformMfaCredential = SaaS.PlatformMfaCredential
UnitMembership = SaaS.UnitMembership
SecureInvitation = SaaS.SecureInvitation
SignupWorkflow = SaaS.SignupWorkflow
RecoveryRequest = SaaS.RecoveryRequest
DatabaseRoutingMetadata = SaaS.DatabaseRoutingMetadata
ProvisioningJob = SaaS.ProvisioningJob
WorkerHeartbeat = SaaS.WorkerHeartbeat
FeatureFlag = SaaS.FeatureFlag
PlanHistory = SaaS.PlanHistory
AggregateUsageEvent = SaaS.AggregateUsageEvent
SuperAdminAudit = SaaS.SuperAdminAudit
CentralSecurityAudit = SaaS.CentralSecurityAudit
QualificationType = SaaS.QualificationType
PersonQualification = SaaS.PersonQualification
PersonQualificationHistory = SaaS.PersonQualificationHistory
RosterPublication = SaaS.RosterPublication
RosterAcknowledgement = SaaS.RosterAcknowledgement
Scenario = SaaS.Scenario
OperationalPosition = SaaS.OperationalPosition
OperationalPositionTimeAllowance = SaaS.OperationalPositionTimeAllowance
OperationalPositionGroup = SaaS.OperationalPositionGroup
PositionCurrencyCategory = SaaS.PositionCurrencyCategory
PositionParticipantRole = SaaS.PositionParticipantRole
PositionStatusEvent = SaaS.PositionStatusEvent
PositionSession = SaaS.PositionSession
PositionSessionParticipant = SaaS.PositionSessionParticipant
ControllerKioskCredential = SaaS.ControllerKioskCredential
PositionSessionAudit = SaaS.PositionSessionAudit
PositionEndorsement = SaaS.PositionEndorsement
PositionRequirement = SaaS.PositionRequirement
BreakPlan = SaaS.BreakPlan
AchievedDuty = SaaS.AchievedDuty
FatigueReport = SaaS.FatigueReport
ToilTransaction = SaaS.ToilTransaction
WorkPattern = SaaS.WorkPattern
WorkPatternDay = SaaS.WorkPatternDay
WorkPatternDayAllowedShift = SaaS.WorkPatternDayAllowedShift
StaffPatternAssignment = SaaS.StaffPatternAssignment
StaffRule = SaaS.StaffRule
BankHoliday = SaaS.BankHoliday
RosterProposal = SaaS.RosterProposal
RosterProposalAssignment = SaaS.RosterProposalAssignment
RosterRuleVersion = SaaS.RosterRuleVersion
RosterPeriod = SaaS.RosterPeriod
RosterImpactEvent = SaaS.RosterImpactEvent
RosterImpactException = SaaS.RosterImpactException
MfaCredential = SaaS.MfaCredential

_briefing = load_briefing_module()
BriefingAssuranceRun = _briefing.BriefingAssuranceRun
BriefingAudit = _briefing.BriefingAudit
BriefingDelivery = _briefing.BriefingDelivery
BriefingItem = _briefing.BriefingItem
BriefingMessageType = _briefing.BriefingMessageType
briefing_blueprint = _briefing.blueprint
briefing_enabled = _briefing.enabled
briefing_local_now = _briefing.local_now

# Enforce the authenticated airport on all legacy operational SELECTs and
# stamp new rows. This protects older routes while they move to repositories.

TENANT_OPERATIONAL_MODELS = (
    RosterSetting, AnnotationType, Watch, Staff, ShiftType, Requirement,
    SpecialRequirement, SmsAudit, Leave, Sickness,
    Assignment, ShiftRequest, RequestAudit, Notification, AnnotationAudit,
    ChangeLog, StaffWatchHistory, QualificationType,
    PersonQualification, PersonQualificationHistory,
    RosterPublication, RosterAcknowledgement, Scenario,
    OperationalPosition, PositionCurrencyCategory, PositionParticipantRole,
    PositionStatusEvent, PositionSession, PositionSessionParticipant,
    ControllerKioskCredential, PositionSessionAudit,
    PositionEndorsement, PositionRequirement, BreakPlan,
    AchievedDuty, FatigueReport, RosterRuleVersion,
    RosterPeriod, RosterImpactEvent, RosterImpactException,
    MfaCredential, BriefingMessageType, BriefingItem, BriefingDelivery,
    BriefingAudit,
    BriefingAssuranceRun,
    HandoverField, HandoverRecord, HandoverOperationalState,
    HandoverEquipment,
    TrainingLevel, TrainingObjective, TrainingSession, TrainingScore,
    ToilTransaction, WorkPattern, WorkPatternDay,
    WorkPatternDayAllowedShift, StaffPatternAssignment, StaffRule, BankHoliday,
)

roster_month_cache = RosterMonthCache(
    float(os.environ.get("ATCROSTER_ROSTER_CACHE_SECONDS", "30"))
)

APPEND_ONLY_AUDIT_MODELS = (
    SmsAudit,
    RequestAudit,
    AnnotationAudit,
    ChangeLog,
    SuperAdminAudit,
    CentralSecurityAudit,
    PositionSessionAudit,
    BriefingAudit,
    ToilTransaction,
)


@event.listens_for(OrmSession, "do_orm_execute")
def _scope_operational_selects(execute_state):
    if not execute_state.is_select or execute_state.execution_options.get("skip_tenant_scope"):
        return
    try:
        unit_id = authenticated_unit_id()
    except RuntimeError:
        return
    statement = execute_state.statement
    for model in TENANT_OPERATIONAL_MODELS:
        statement = statement.options(with_loader_criteria(
            model, lambda cls: cls.unit_id == unit_id,
            include_aliases=True,
            track_closure_variables=True,
        ))
    execute_state.statement = statement


@event.listens_for(OrmSession, "before_flush")
def _stamp_operational_writes(session_obj, _flush_context, _instances):
    touched_units = session_obj.info.setdefault(
        "roster_cache_touched_units", set()
    )
    for record in session_obj.new | session_obj.dirty | session_obj.deleted:
        if isinstance(record, TENANT_OPERATIONAL_MODELS):
            unit_id = getattr(record, "unit_id", None)
            if unit_id:
                touched_units.add(int(unit_id))
    for record in session_obj.dirty:
        if isinstance(record, APPEND_ONLY_AUDIT_MODELS) and session_obj.is_modified(
            record, include_collections=False
        ):
            # Provider delivery is external lifecycle metadata, not historical
            # evidence. SmsAudit content remains immutable; only this one field
            # may be updated when MessageMedia posts a signed delivery report.
            if isinstance(record, SmsAudit):
                changed = {
                    attribute.key for attribute in sa_inspect(record).attrs
                    if attribute.history.has_changes()
                }
                if changed == {"delivery_status"}:
                    continue
            raise PermissionError("Audit evidence is append-only")
    for record in session_obj.deleted:
        if isinstance(record, APPEND_ONLY_AUDIT_MODELS):
            raise PermissionError("Audit evidence is append-only")
    try:
        unit_id = authenticated_unit_id()
    except RuntimeError:
        return
    for record in session_obj.new:
        if isinstance(record, TENANT_OPERATIONAL_MODELS):
            supplied = getattr(record, "unit_id", None)
            if supplied not in (None, unit_id):
                raise PermissionError("Cross-unit writes are forbidden")
            record.unit_id = unit_id


@event.listens_for(OrmSession, "after_commit")
def _invalidate_roster_cache_after_commit(session_obj):
    for unit_id in session_obj.info.pop("roster_cache_touched_units", set()):
        roster_month_cache.invalidate_unit(unit_id)


@event.listens_for(OrmSession, "after_rollback")
def _discard_roster_cache_invalidation_after_rollback(session_obj):
    session_obj.info.pop("roster_cache_touched_units", None)

# -------------------- Reference data helpers --------------------


def _normalise_codes(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = []
    for val in values:
        code = (val or "").strip().upper()
        if code and code not in seen:
            seen.append(code)
    return seen


@lru_cache(maxsize=128)
def _roster_settings_snapshot(unit_id: int) -> dict[str, str]:
    rows = RosterSetting.query.filter_by(unit_id=unit_id).all()
    return {row.key: row.value for row in rows}


def refresh_roster_settings_cache() -> None:
    _roster_settings_snapshot.cache_clear()
    try:
        _shift_groups_snapshot.cache_clear()
    except NameError:
        pass


def _load_codes_setting(
    key: str, default: list[str], unit_id: int | None = None
) -> set[str]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get(key)
    if not raw:
        parsed = default
    else:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = default
    configured = set(_normalise_codes(parsed))
    existing = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(
            unit_id=resolved_unit_id
        ).all()
    }
    return configured & existing


def get_working_codes() -> set[str]:
    return _load_codes_setting("working_codes", DEFAULT_WORKING_CODES)


def get_absence_types(
    category: str | None = None,
    active_only: bool = True,
    unit_id: int | None = None,
) -> list[dict[str, object]]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get("absence_types")
    try:
        parsed = json.loads(raw) if raw else DEFAULT_ABSENCE_TYPES
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = DEFAULT_ABSENCE_TYPES
    if not isinstance(parsed, list):
        parsed = DEFAULT_ABSENCE_TYPES
    result = []
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().upper()
        item_category = str(item.get("category") or "").strip().lower()
        if (
            not re.fullmatch(r"[A-Z0-9]{1,10}", code)
            or item_category not in {"leave", "sickness"}
            or code in seen
        ):
            continue
        seen.add(code)
        normalised = {
            "code": code,
            "label": str(item.get("label") or code).strip()[:80] or code,
            "category": item_category,
            "active": bool(item.get("active", True)),
        }
        if category and item_category != category:
            continue
        if active_only and not normalised["active"]:
            continue
        result.append(normalised)
    return result


def _save_absence_types(items: list[dict[str, object]]) -> None:
    _save_roster_setting(
        "absence_types", json.dumps(items, separators=(",", ":"))
    )
    db.session.commit()
    refresh_roster_settings_cache()


def get_banned_roster_codes() -> set[str]:
    return _load_codes_setting("banned_codes", DEFAULT_BANNED_ROSTER_CODES)


def get_exclude_from_counters() -> set[str]:
    return _load_codes_setting("exclude_from_counters", DEFAULT_EXCLUDE_FROM_COUNTERS)


def get_non_working_codes() -> set[str]:
    return _load_codes_setting("non_working_codes", DEFAULT_NON_WORKING_CODES)


def staff_is_countable_on(person: Staff, on_date: date) -> bool:
    """Require a current medical and at least one current operational rating UE."""
    if person.id is None:
        # Lightweight compatibility for unsaved profile previews and pure
        # validation tests; persisted roster rows use the authoritative dated
        # qualification service below.
        medical_valid = bool(
            person.medical_expiry and person.medical_expiry >= on_date
        )
        independent_ue = any(
            expiry and expiry >= on_date
            for expiry in (person.tower_ue_expiry, person.radar_ue_expiry)
        )
        return bool(medical_valid and independent_ue)
    return get_staff_operational_capability(
        person.id, on_date
    ).counts_as_operational


def operational_capability_service():
    return OperationalCapabilityService(OperationalCapabilityDependencies(
        db=db, Staff=Staff, QualificationType=QualificationType,
        PersonQualification=PersonQualification,
    ))


def get_staff_operational_capability(staff_id: int, on_date: date):
    return operational_capability_service().get_staff_operational_capability(
        staff_id, on_date
    )


def get_operational_capability_matrix(staff: list[Staff], days: list[date]):
    return operational_capability_service().get_capability_matrix(staff, days)


def get_shift_counter_map(unit_id: int | None = None) -> dict[str, str]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get(
        "shift_counter_map", "{}"
    )
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        values = {}
    if not isinstance(values, dict):
        return {}
    return {
        str(code).upper(): str(group).upper()
        for code, group in values.items()
        if str(group).upper() in {"", "M", "D", "A", "N"}
    }


sms_configuration = SmsConfigurationService(
    settings_snapshot=_roster_settings_snapshot,
    current_unit_id=_current_unit_id,
)
sms_audit_service = SmsAuditService(
    db=db,
    SmsAudit=SmsAudit,
    current_unit_id=_current_unit_id,
    current_user=lambda: current_user,
)
overtime_sms_service = OvertimeSmsService(
    configuration=sms_configuration,
    audit=sms_audit_service,
    send=_send_sms,
)


def shift_counter_group(
    code: str | None, unit_id: int | None = None
) -> str:
    value = (code or "").strip().upper()
    if not value:
        return ""
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    mapping = get_shift_counter_map(resolved_unit_id)
    if value in mapping:
        return mapping[value]
    # Legacy default grouping is only valid for a working shift that actually
    # exists for this airport. Pattern letters are not, by themselves, enough
    # to claim that somebody is covering that staffing group.
    shift = get_shift(value, resolved_unit_id)
    if (
        not shift
        or not shift.is_active
        or not shift.is_working
        or shift.is_training
    ):
        return ""
    if value == "EM":
        return "M"
    if value == "LA":
        return "A"
    return value if value in {"M", "D", "A", "N"} else ""


def shift_counter_group_for_day(
    code: str | None, on_date: date, unit_id: int | None = None
) -> str:
    """Return the staffing group, suppressing nights when the unit is closed."""
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    group = shift_counter_group(code, resolved_unit_id)
    if group == "N" and not _night_active_on(resolved_unit_id, on_date):
        return ""
    return group


@lru_cache(maxsize=128)
def _annotation_snapshot(unit_id: int) -> dict[str, object]:
    rows = (AnnotationType.query
            .filter(AnnotationType.unit_id == unit_id)
            .order_by(AnnotationType.code)
            .all())
    items = []
    for row in rows:
        tags = tuple(sorted({
            t.strip().lower() for t in (row.tags or "").split(",") if t.strip()
        }))
        suffixes = "".join(sorted({c for c in (row.suffixes or "").upper()}))
        items.append({
            "id": row.id,
            "code": (row.code or "").upper(),
            "label": row.label or row.code.upper(),
            "category": row.category or "Other",
            "colour": row.colour or "#6c757d",
            "description": row.description or "",
            "allow_suffix": bool(row.allow_suffix),
            "suffixes": suffixes,
            "toil_half_days": int(row.toil_half_days or 0),
            "tags": tags,
            "note_required": bool(row.note_required),
            "admin_only": bool(row.admin_only),
            "is_active": bool(row.is_active),
            "sort_order": row.sort_order if row.sort_order is not None else 0,
        })
    by_code = {item["code"]: item for item in items}
    return {"items": items, "by_code": by_code}


def refresh_annotation_cache() -> None:
    _annotation_snapshot.cache_clear()


def get_annotation_types(
    active_only: bool = True, unit_id: int | None = None
) -> list[dict[str, object]]:
    snap = _annotation_snapshot(int(unit_id or _current_unit_id() or 1))
    items = snap["items"]
    if active_only:
        items = [item for item in items if item["is_active"]]
    return items


def get_annotation_config(
    code: str | None, unit_id: int | None = None
) -> dict[str, object] | None:
    if not code:
        return None
    return _annotation_snapshot(
        int(unit_id or _current_unit_id() or 1)
    )["by_code"].get(code.strip().upper())


def get_annotation_groups() -> OrderedDict[str, list[dict[str, object]]]:
    groups: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for item in get_annotation_types(active_only=True):
        groups.setdefault(item["category"], []).append(item)
    return groups


def annotation_tags_for(code: str | None) -> set[str]:
    info = get_annotation_config(code)
    if not info:
        return set()
    tags = info.get("tags") or ()
    return {t for t in tags}


def annotation_codes_for_tag(tag: str, active_only: bool = True) -> list[str]:
    needle = (tag or "").lower().strip()
    if not needle:
        return []
    codes = []
    for item in get_annotation_types(active_only=active_only):
        tags = {t for t in (item.get("tags") or ())}
        if needle in tags:
            codes.append(item["code"])
    return codes


def _parse_codes_input(raw: str) -> list[str]:
    tokens = re.split(r"[\s,]+", raw or "")
    return _normalise_codes(tokens)


def _save_codes_setting(key: str, values: list[str]) -> None:
    payload = json.dumps(_normalise_codes(values))
    unit_id = int(_current_unit_id() or 1)
    row = RosterSetting.query.filter_by(unit_id=unit_id, key=key).first()
    if not row:
        row = RosterSetting(unit_id=unit_id, key=key, value=payload)
        db.session.add(row)
    else:
        row.value = payload
    db.session.commit()
    refresh_roster_settings_cache()


def _prune_roster_code_settings(unit_id: int) -> int:
    """Remove list entries that have no ShiftType in this airport."""
    valid_codes = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(
            unit_id=unit_id
        ).all()
    }
    changed = 0
    rows = RosterSetting.query.filter(
        RosterSetting.unit_id == unit_id,
        RosterSetting.key.in_(DEFAULT_ROSTER_SETTINGS),
    ).all()
    for row in rows:
        try:
            values = json.loads(row.value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            values = []
        normalised = _normalise_codes(values)
        cleaned = [code for code in normalised if code in valid_codes]
        if cleaned != normalised:
            row.value = json.dumps(cleaned)
            changed += 1
    if changed:
        refresh_roster_settings_cache()
    return changed


def _save_roster_setting(key: str, value: str) -> None:
    unit_id = int(_current_unit_id() or 1)
    row = RosterSetting.query.filter_by(unit_id=unit_id, key=key).first()
    if not row:
        row = RosterSetting(unit_id=unit_id, key=key, value=value)
        db.session.add(row)
    else:
        row.value = value
    refresh_roster_settings_cache()


def _operational_currency_requirement(unit_id: int | None = None) -> dict[str, Any]:
    return load_currency_requirement(
        unit_id,
        current_unit_id=_current_unit_id,
        settings_snapshot=_roster_settings_snapshot,
        setting_key=OPERATIONAL_CURRENCY_SETTING_KEY,
        defaults=DEFAULT_OPERATIONAL_CURRENCY_REQUIREMENT,
    )


def _save_operational_currency_requirement(data: dict[str, Any]) -> None:
    requirement = dict(DEFAULT_OPERATIONAL_CURRENCY_REQUIREMENT)
    requirement.update(data)
    _save_roster_setting(
        OPERATIONAL_CURRENCY_SETTING_KEY, json.dumps(requirement, sort_keys=True)
    )


def _operational_currency_window(
    requirement: dict[str, Any], today: date | None = None
) -> tuple[date, date]:
    return currency_window(requirement, today or utcnow().date())


def _minutes_between(start: datetime, end: datetime) -> int:
    return calculate_minutes_between(start, end)


def _operational_currency_shortfalls(unit_id: int) -> dict[str, Any]:
    return operational_currency_shortfalls(
        unit_id,
        db=db,
        Staff=Staff,
        PositionEndorsement=PositionEndorsement,
        PositionSession=PositionSession,
        PositionParticipantRole=PositionParticipantRole,
        PositionSessionParticipant=PositionSessionParticipant,
        requirement_for=_operational_currency_requirement,
        live_position_enabled=live_position_enabled,
        now=utcnow,
    )


def _parse_sms_number_lines(raw: str) -> tuple[list[dict[str, str]], list[str]]:
    return parse_sms_number_lines(raw)


def bootstrap_reference_data() -> None:
    return bootstrap_roster_reference_data(
        db=db,
        Unit=Unit,
        AnnotationType=AnnotationType,
        RosterSetting=RosterSetting,
        annotation_defaults=DEFAULT_ANNOTATION_TYPES,
        roster_defaults=DEFAULT_ROSTER_SETTINGS,
        normalise_codes=_normalise_codes,
        refresh_annotation_cache=refresh_annotation_cache,
        refresh_roster_settings_cache=refresh_roster_settings_cache,
    )


if (
    DEPLOYMENT_ENV != "production"
    and os.environ.get("ATCROSTER_SKIP_BOOTSTRAP", "").lower()
    not in {"1", "true", "yes"}
):
    try:
        with app.app_context():
            bootstrap_reference_data()
    except Exception:
        with app.app_context():
            db.session.rollback()

# Cached shift lookup (define after models so ShiftType exists when called)


@lru_cache(maxsize=256)
def _shift_by_code(unit_id: int, code: str):
    return ShiftType.query.filter_by(unit_id=unit_id, code=code).first()


def refresh_shift_cache():
    _shift_by_code.cache_clear()

# -------------------- Login --------------------


@login_manager.user_loader
def load_user(user_id):
    value = str(user_id or "")
    if value.startswith("membership:"):
        try:
            membership_id = int(value.split(":", 1)[1])
        except ValueError:
            return None
        membership = db.session.get(UnitMembership, membership_id)
        if not membership or membership.status != "active":
            return None
        routing = db.session.get(
            DatabaseRoutingMetadata, membership.unit_id
        )
        if DEPLOYMENT_ENV == "production" and not routing:
            return None
        token = bind_authenticated_unit(
            membership.unit_id,
            routing.secret_name if routing else None,
        )
        g.tenant_context_token = token
        return db.session.get(Staff, membership.person_id)
    if value.startswith("platform-identity:"):
        try:
            return db.session.get(
                PlatformIdentity, int(value.split(":", 1)[1])
            )
        except ValueError:
            return None
    if value.startswith("legacy:") and DEPLOYMENT_ENV != "production":
        try:
            _, raw_unit_id, raw_person_id = value.split(":", 2)
            token = bind_authenticated_unit(int(raw_unit_id))
            g.tenant_context_token = token
            return db.session.get(Staff, int(raw_person_id))
        except ValueError:
            return None
    return None

# --------- Fast month loader & cache (uses functions defined later but safe) ----------


def _load_month_roster_core(unit_id: int, y: int, m: int):
    return load_month_roster(
        unit_id,
        y,
        m,
        MonthRosterLoadDependencies(
            db=db,
            Assignment=Assignment,
            Requirement=Requirement,
            Staff=Staff,
            Watch=Watch,
            ensure_month_requirement=ensure_month_requirement,
            log_exception=app.logger.exception,
        ),
    )


# IMPORTANT: overwrite any previously memoized wrapper
_load_month_roster_fast = _memoize(seconds=300)(_load_month_roster_core)


# -------------------- Helpers --------------------
# === Unified permissions (admins, editors, WM, DWM) ===


def is_admin_user(u) -> bool:
    return is_admin(u)


def is_editor_user(u) -> bool:
    return is_editor(u)


def user_permissions(u) -> dict[str, bool]:
    return permissions_for(u)


def has_unit_permission(u, permission: str) -> bool:
    return has_permission(u, permission)


def is_under_training(person) -> bool:
    return is_trainee(person)


def can_record_training(u) -> bool:
    return may_record_training(u)


def can_manage_training(u) -> bool:
    return may_manage_training(u)


def training_enabled(unit_id: int) -> bool:
    return bool(FeatureFlag.query.filter_by(
        unit_id=unit_id, key="training_module", enabled=True
    ).first())


def competency_enabled(unit_id: int) -> bool:
    row = FeatureFlag.query.filter_by(
        unit_id=unit_id, key="competency_module"
    ).first()
    # Existing airports inherit their current combined-module entitlement
    # until Super Admin explicitly chooses a separate competency setting.
    return bool(row.enabled) if row else training_enabled(unit_id)


def live_position_enabled(unit_id: int) -> bool:
    return bool(FeatureFlag.query.filter_by(
        unit_id=unit_id, key="live_position_monitoring", enabled=True
    ).first())


def can_edit_roster(u) -> bool:
    return may_edit_roster(u)


def can_apply_annotations(u) -> bool:
    return may_apply_annotations(u)


def can_send_unit_messages(u) -> bool:
    return may_send_unit_messages(u)


def can_override_roster_conflicts(u) -> bool:
    return may_override_roster_conflicts(u)


def tenant_get(model, record_id: int):
    """Fetch one operational record with an explicit mutation-safe boundary."""
    return model.query.filter_by(
        id=int(record_id), unit_id=_current_unit_id()
    ).first()


def roster_edit_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not can_edit_roster(current_user):
            return ("Forbidden", 403)
        return f(*args, **kwargs)
    return wrapper


def month_has_data(year: int, month: int) -> bool:
    return roster_month_has_data(db, Assignment, year, month, _month_add)


def _lock_roster_month(unit_id: int, year: int, month: int) -> Requirement:
    return lock_roster_period(
        db, Requirement, unit_id, year, month, ensure_month_requirement,
    )


def month_range(year: int, month: int):
    return month_days(year, month)


def watch_id_for_staff_on(staff_id: int, on_date: date) -> int | None:
    return _watch_id_for_staff_on(
        authenticated_unit_id(), staff_id, on_date
    )


def watch_ids_for_staff_on(
    staff: list[Staff], on_date: date
) -> dict[int, int | None]:
    return resolve_watch_ids(StaffWatchHistory, staff, authenticated_unit_id(), on_date)


@lru_cache(maxsize=4096)
def _watch_id_for_staff_on(
    unit_id: int, staff_id: int, on_date: date
) -> int | None:
    """Return the watch_id that applies to this staff on a given date
    using StaffWatchHistory; fall back to Staff.watch_id if no history."""
    return resolve_watch_id(db, StaffWatchHistory, Staff, unit_id, staff_id, on_date)


def parse_ym(ym: str):
    return parse_roster_year_month(ym, parse_year_month)


def get_shift(code: str, unit_id: int | None = None):
    # hot path → use cached lookup
    return _shift_by_code(
        int(unit_id or _current_unit_id() or 1), (code or "").upper()
    )


@lru_cache(maxsize=128)
def _shift_groups_snapshot(unit_id: int):
    return shift_groups_snapshot(ShiftType, unit_id, get_banned_roster_codes)


PATTERN_CODES = ("M", "A", "D", "N", "OPS", "OFF")
DEFAULT_BASE_PATTERN = "M,M,A,A,N,N,OFF,OFF,OFF,OFF"


def _expand_pattern(raw_value: str | None) -> list[str]:
    return expand_roster_pattern(raw_value, expand_pattern)


def _validated_pattern(raw_value: str | None) -> list[str]:
    return validate_roster_pattern(raw_value, validated_pattern)


def _effective_watch(staff: Staff, on_date: date) -> Watch | None:
    return resolve_effective_watch(db, StaffWatchHistory, staff, on_date)


def _unit_pattern_context(unit_id: int) -> tuple[list[str], date]:
    settings = _roster_settings_snapshot(unit_id)
    pattern = _validated_pattern(
        settings.get("base_pattern_csv") or DEFAULT_BASE_PATTERN
    )
    try:
        anchor = date.fromisoformat(
            settings.get("base_pattern_anchor") or "2025-01-01"
        )
    except ValueError:
        anchor = date(2025, 1, 1)
    return pattern or _validated_pattern(DEFAULT_BASE_PATTERN), anchor


def _pattern_context(staff: Staff, on_date: date) -> tuple[list[str], date]:
    if staff.pattern_override:
        personal = _validated_pattern(staff.pattern_csv)
        if personal:
            return personal, staff.pattern_anchor or on_date
    unit_pattern, unit_anchor = _unit_pattern_context(staff.unit_id)
    watch = _effective_watch(staff, on_date)
    if watch:
        move = StaffWatchHistory.query.filter(
            StaffWatchHistory.unit_id == staff.unit_id,
            StaffWatchHistory.staff_id == staff.id,
            StaffWatchHistory.effective_date <= on_date,
            db.or_(
                StaffWatchHistory.effective_to.is_(None),
                StaffWatchHistory.effective_to >= on_date,
            ),
        ).order_by(StaffWatchHistory.effective_date.desc()).first()
        watch_pattern = _validated_pattern(watch.pattern_csv)
        # A watch anchor phases both a watch-specific pattern and the inherited
        # unit pattern. This is what makes two watches on the same base cycle
        # start on different cycle days.
        return (
            watch_pattern or unit_pattern,
            (move.pattern_anchor if move and move.pattern_anchor else None)
            or watch.pattern_anchor or unit_anchor,
        )
    return unit_pattern, unit_anchor


def pattern_for(staff: Staff, on_date: date | None = None):
    return _pattern_context(staff, on_date or date.today())[0]


def _night_active_on(unit_id: int, on_date: date) -> bool:
    raw = _roster_settings_snapshot(unit_id).get(
        "night_active_weekdays", "0,1,2,3,4,5,6"
    )
    try:
        active_days = {
            int(value) for value in raw.split(",")
            if value.strip() != ""
        }
    except ValueError:
        active_days = set(range(7))
    return on_date.weekday() in active_days


def day_leave_for(staff: Staff, d: date):
    for lv in staff.leaves:
        if lv.start <= d <= lv.end:
            return lv.leave_type
    return None


def code_from_pattern(staff: Staff, d: date):
    pat, anchor = _pattern_context(staff, d)
    if not pat:
        return "OFF"
    idx = (d - anchor).days % len(pat)
    code = pat[idx]
    return "OFF" if code == "N" and not _night_active_on(staff.unit_id, d) else code


def _effective_watch_id(staff: Staff, duty_day: date) -> int | None:
    watch = _effective_watch(staff, duty_day)
    return watch.id if watch else None


def deterministic_roster_population_service():
    """Build the shared baseline-population service for application callers."""
    return DeterministicRosterPopulationService(PopulationDependencies(
        db=db,
        Unit=Unit,
        Staff=Staff,
        Assignment=Assignment,
        ShiftType=ShiftType,
        WorkPattern=WorkPattern,
        WorkPatternDay=WorkPatternDay,
        WorkPatternDayAllowedShift=WorkPatternDayAllowedShift,
        StaffPatternAssignment=StaffPatternAssignment,
        utcnow=utcnow,
        legacy_code_resolver=code_from_pattern,
        watch_id_resolver=_effective_watch_id,
        RosterPeriod=globals().get("RosterPeriod"),
    ))


def _generated_roster_horizon_end(unit_id: int, effective_from: date) -> date | None:
    """Return the last generated date; future events wait for future generation."""
    return db.session.query(db.func.max(Assignment.day)).filter(
        Assignment.unit_id == unit_id,
        Assignment.day >= effective_from,
    ).scalar()


def _invalidate_roster_impact_coverage(
    unit_id: int,
    effective_from: date,
    effective_to: date,
    _staff_ids: tuple[int, ...],
    _watch_ids: tuple[int, ...],
):
    """Invalidate each affected monthly coverage/roster cache entry."""
    cursor = effective_from.replace(day=1)
    final = effective_to.replace(day=1)
    while cursor <= final:
        if _cache:
            try:
                _cache.delete_memoized(
                    _load_month_roster_fast, int(unit_id), cursor.year, cursor.month
                )
            except Exception:
                pass
        next_year, next_month = add_months(cursor.year, cursor.month, 1)
        cursor = date(next_year, next_month, 1)


def roster_impact_service():
    return RosterImpactService(RosterImpactDependencies(
        db=db,
        Unit=Unit,
        RosterImpactEvent=RosterImpactEvent,
        RosterImpactException=RosterImpactException,
        population_service=deterministic_roster_population_service(),
        generated_horizon_end=_generated_roster_horizon_end,
        recalculate_coverage=_invalidate_roster_impact_coverage,
        override_classifier=globals().get("override_classification_service"),
        utcnow=utcnow,
    ))


def record_roster_impact(
    event_type: RosterImpactEventType | str,
    effective_from: date,
    *,
    effective_to: date | None = None,
    staff_ids=(),
    watch_ids=(),
    rebuild_baseline=False,
    recalculate_coverage=True,
    reason="",
):
    """Record and apply a unit-scoped roster trigger in the caller's transaction."""
    actor_id = getattr(current_user, "person_id", None)
    if actor_id is None and getattr(current_user, "is_authenticated", False):
        actor_id = getattr(current_user, "id", None)
    return roster_impact_service().handle_roster_impact_event(
        _current_unit_id(), event_type, effective_from, effective_to,
        staff_ids=staff_ids, watch_ids=watch_ids,
        rebuild_baseline=rebuild_baseline,
        recalculate_coverage=recalculate_coverage,
        reason=reason, triggered_by_user_id=actor_id,
    )


def _qualification_impact_type(
    code: str,
    old_status: str | None,
    old_valid_from: date | None,
    old_expires_on: date | None,
    new_status: str | None,
    new_valid_from: date | None,
    new_expires_on: date | None,
) -> tuple[RosterImpactEventType | None, date]:
    return classify_qualification_impact(
        code,
        old_status,
        old_valid_from,
        old_expires_on,
        new_status,
        new_valid_from,
        new_expires_on,
        impact_types=RosterImpactEventType,
        today=date.today(),
    )


def _person_has_other_valid_ue(
    unit_id: int,
    person_id: int,
    excluded_type_id: int,
    on_date: date,
) -> bool:
    return has_other_valid_ue(
        unit_id,
        person_id,
        excluded_type_id,
        on_date,
        db=db,
        PersonQualification=PersonQualification,
        QualificationType=QualificationType,
    )


def record_qualification_roster_impact(
    person,
    qtype,
    old_status,
    old_valid_from,
    old_expires_on,
    record,
    *,
    reason="Qualification changed.",
):
    return record_roster_impact_for_qualification(
        person,
        qtype,
        old_status,
        old_valid_from,
        old_expires_on,
        record,
        impact_types=RosterImpactEventType,
        today=date.today(),
        has_other_ue=_person_has_other_valid_ue,
        record_roster_impact=record_roster_impact,
        reason=reason,
    )


def _cycle_day_for(staff: Staff, d: date) -> int | None:
    """Return the 1-indexed pattern cycle day for `staff` on date `d`."""
    pat, anchor = _pattern_context(staff, d)
    if not pat:
        return None
    return ((d - anchor).days % len(pat)) + 1


def _assignment_refresh_dependencies():
    return AssignmentRefreshDependencies(
        db=db,
        Assignment=Assignment,
        Staff=Staff,
        code_from_pattern=code_from_pattern,
        day_leave_for=day_leave_for,
        get_shift=get_shift,
        absence_types=get_absence_types,
    )


def set_assignment(staff: Staff, d: date, code: str, source="auto", note=""):
    return set_generated_assignment(
        staff,
        d,
        code,
        dependencies=_assignment_refresh_dependencies(),
        source=source,
        note=note,
    )


def overwrite_assignment(staff: Staff, d: date, code: str, note: str = ""):
    return set_absence_override(
        staff,
        d,
        code,
        dependencies=_assignment_refresh_dependencies(),
        note=note,
    )

# Respect manual edits & only clear annotations when auto changes the code


def refresh_day_from_pattern_and_leave(staff: Staff, d: date):
    return refresh_pattern_day(staff, d, _assignment_refresh_dependencies())


def shift_duration_minutes(shift: ShiftType):
    return roster_shift_duration_minutes(shift, shift_minutes)


def ensure_month_requirement(year, month, default=(4, 4, 4, 2)):
    return ensure_roster_month_requirement(db, Requirement, year, month, default)


def requirements_for_day(
    requirement: Requirement | None,
    day: date,
    special: SpecialRequirement | None = None,
) -> dict[str, int]:
    return resolve_roster_requirements_for_day(requirement, day, special, daily_requirements)

# Idempotent month generation that preserves manual entries


def generate_month(year: int, month: int, *args, **kwargs):
    return generate_month_assignments(
        year,
        month,
        db=db,
        Staff=Staff,
        month_range=month_range,
        refresh_day=refresh_day_from_pattern_and_leave,
    )


def _is_working_day_code(code: str) -> bool:
    """
    True for working 'Day' shifts (codes that start with 'D'),
    excluding non-working types like OFF/leave/TOIL/etc.
    Uses ShiftType.is_working when known; otherwise falls back to prefix check.
    """
    c = (code or "").strip().upper()
    if not c:
        return False

    NON_WORKING = {"OFF", "AL", "PL", "SPL", "TOU8", "TOUI",
                   "OSS", "OFFICE", "WFH", "CTB", "MTG"}
    if c in NON_WORKING:
        return False

    try:
        sh = get_shift(c)
    except NameError:
        sh = None
    if sh is None:
        try:
            sh = ShiftType.query.filter_by(code=c).first()
        except Exception:
            sh = None

    if sh is not None:
        return bool(getattr(sh, "is_working", False)) and c.startswith("D")
    return c.startswith("D")


_fatigue_rule_config_service = FatigueRuleConfigService(
    FatigueRuleConfigDependencies(
        db=db,
        RosterSetting=RosterSetting,
        current_unit_id=_current_unit_id,
    )
)
_fatigue_rule_config = _fatigue_rule_config_service.load
_save_fatigue_rule_config = _fatigue_rule_config_service.save




def _segments_from_assignments(staff: Staff, assignments, definitions):
    return segments_from_assignments(
        staff,
        assignments,
        definitions,
        get_shift=get_shift,
        is_working=_is_working,
        span=_span,
        is_night_duty=_is_night_duty,
        is_early_start=_is_early_start,
        is_morning_duty=_is_morning_duty,
    )


def _configured_fatigue_findings(segments, config, observation_start):
    return configured_fatigue_findings(
        segments,
        config,
        observation_start,
        analyze_segments=_analyze_segments,
        custom_fatigue_flags=_custom_fatigue_flags,
    )


def _segments_for_staff(staff: Staff, start_day: date, end_day: date):
    definitions = _fatigue_rule_config(staff.unit_id)["definitions"]
    assignments = (Assignment.query
                   .filter(Assignment.staff_id == staff.id,
                           Assignment.day >= start_day,
                           Assignment.day <= end_day)
                   .order_by(Assignment.day.asc()).all())
    return _segments_from_assignments(staff, assignments, definitions)




def fatigue_flags_for_range(staff: Staff, day_list, lookback_days=30):
    if not day_list:
        return {}
    day_list = sorted(day_list)
    start_lb = day_list[0] - timedelta(days=lookback_days)
    end_day = day_list[-1]
    segs = _segments_for_staff(staff, start_lb, end_day)
    config = _fatigue_rule_config(staff.unit_id)
    all_flags = _configured_fatigue_findings(
        segs, config, datetime.combine(start_lb, time.min)
    )
    target_set = set(day_list)
    return {
        d: findings for d, findings in all_flags.items()
        if d in target_set and findings
    }


def roster_fatigue_flags_for_range(
    staff: Staff,
    day_list,
    code_by_day: dict[date, str],
    unit_id: int | None = None,
) -> dict[date, list[str]]:
    """Expose fatigue warnings only on active working-duty roster cells."""
    resolved_unit_id = int(unit_id or staff.unit_id)
    findings = fatigue_flags_for_range(staff, day_list)
    return {
        finding_day: messages
        for finding_day, messages in findings.items()
        if (
            (shift := get_shift(
                code_by_day.get(finding_day), resolved_unit_id
            ))
            and shift.is_active
            and shift.is_working
        )
    }


def roster_fatigue_flags_matrix(
    staff: list[Staff], day_list: list[date],
    code_by_staff: dict[int, dict[date, str]], unit_id: int,
) -> dict[int, dict[date, list[str]]]:
    """Calculate every displayed person's fatigue flags from one duty query."""
    if not staff or not day_list:
        return {}
    ordered_days = sorted(day_list)
    start_lb = ordered_days[0] - timedelta(days=30)
    end_day = ordered_days[-1]
    staff_ids = [person.id for person in staff if person.id is not None]
    assignments = Assignment.query.filter(
        Assignment.unit_id == unit_id,
        Assignment.staff_id.in_(staff_ids or [0]),
        Assignment.day >= start_lb,
        Assignment.day <= end_day,
    ).order_by(Assignment.staff_id, Assignment.day).all()
    assignments_by_staff: dict[int, list[Assignment]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_staff[assignment.staff_id].append(assignment)
    config = _fatigue_rule_config(unit_id)
    target_days = set(ordered_days)
    result: dict[int, dict[date, list[str]]] = {}
    for person in staff:
        segments = _segments_from_assignments(
            person, assignments_by_staff.get(person.id, ()), config["definitions"]
        )
        findings = _configured_fatigue_findings(
            segments, config, datetime.combine(start_lb, time.min)
        )
        visible = {}
        for finding_day, messages in findings.items():
            shift = get_shift(
                code_by_staff.get(person.id, {}).get(finding_day), unit_id
            )
            if (
                finding_day in target_days and messages and shift
                and shift.is_active and shift.is_working
            ):
                visible[finding_day] = messages
        result[person.id] = visible
    return result


def would_trigger_fatigue(staff: Staff, day: date, code: str):
    sh = get_shift(code, staff.unit_id)
    if not _is_working(sh):
        return []
    start_lb = day - timedelta(days=30)
    end_day = day
    segs = [
        segment for segment in _segments_for_staff(staff, start_lb, end_day)
        if segment["day"] != day
    ]
    config = _fatigue_rule_config(staff.unit_id)
    definitions = config["definitions"]
    sdt, edt = _span(day, sh)
    if sdt:
        is_early, is_pre0600 = _is_early_start(sdt, definitions)
        segs.append({
            "day": day, "start": sdt, "end": edt,
            "mins": int((edt - sdt).total_seconds() // 60),
            "night": _is_night_duty(sdt, edt, definitions),
            "early": is_early,
            "early_pre0600": is_pre0600,
            "morning": _is_morning_duty(sdt),
        })
    flags = _configured_fatigue_findings(
        segs, config, datetime.combine(start_lb, time.min)
    )
    return flags.get(day, [])


def would_trigger_fatigue_with_plan(
    staff: Staff,
    day: date,
    code: str,
    proposed_codes: dict[date, str],
):
    """Evaluate a candidate together with earlier in-memory proposal duties."""
    shift = get_shift(code, staff.unit_id)
    if not _is_working(shift):
        return []
    start_day = min([day, *proposed_codes], default=day) - timedelta(days=30)
    end_day = max([day, *proposed_codes], default=day)
    proposed_days = set(proposed_codes) | {day}
    segments = [
        segment
        for segment in _segments_for_staff(staff, start_day, end_day)
        if segment["day"] not in proposed_days
    ]
    config = _fatigue_rule_config(staff.unit_id)
    definitions = config["definitions"]
    for proposed_day, proposed_code in {**proposed_codes, day: code}.items():
        proposed_shift = get_shift(proposed_code, staff.unit_id)
        start_dt, end_dt = _span(proposed_day, proposed_shift)
        if not start_dt:
            continue
        early, pre0600 = _is_early_start(start_dt, definitions)
        segments.append({
            "day": proposed_day,
            "start": start_dt,
            "end": end_dt,
            "mins": int((end_dt - start_dt).total_seconds() // 60),
            "night": _is_night_duty(start_dt, end_dt, definitions),
            "early": early,
            "early_pre0600": pre0600,
            "morning": _is_morning_duty(start_dt),
        })
    segments.sort(key=lambda item: item["start"])
    findings = _configured_fatigue_findings(
        segments, config, datetime.combine(start_day, time.min)
    )
    return findings.get(day, [])


def _year_month_iter(start_date: date, end_date: date):
    yield from iter_year_months(start_date, end_date)


def generate_range(start_day: date, end_day: date):
    return generate_assignment_range(
        start_day,
        end_day,
        iter_year_months=_year_month_iter,
        ensure_month_requirement=ensure_month_requirement,
        generate_month=generate_month,
    )


def ensure_assignments_for_range(start_day: date, end_day: date):
    return generate_assignment_range(
        start_day,
        end_day,
        iter_year_months=_year_month_iter,
        ensure_month_requirement=ensure_month_requirement,
        generate_month=generate_month,
    )


def would_create_new_fatigue_issues(
    staff: Staff,
    proposed_day: date,
    proposed_code: str,
    lookback_days: int = 30,
    lookahead_days: int = 14,
):
    return new_findings_for_proposed_assignment(
        staff,
        proposed_day,
        proposed_code,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
        get_shift=get_shift,
        is_working=_is_working,
        segments_for_staff=_segments_for_staff,
        fatigue_rule_config=_fatigue_rule_config,
        configured_fatigue_findings=_configured_fatigue_findings,
        span=_span,
        is_early_start=_is_early_start,
        is_night_duty=_is_night_duty,
        is_morning_duty=_is_morning_duty,
    )


_compliance_month = compliance_month


def _compliance_findings(year: int, month: int) -> dict:
    return monthly_compliance_findings(
        year,
        month,
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        month_range=month_range,
        fatigue_rule_config=_fatigue_rule_config,
        fatigue_flags_for_range=fatigue_flags_for_range,
    )


# -------------------- Migrations / seeding --------------------


def migrate_tenant_foundation_compat():
    return upgrade_tenant_foundation(db=db, Unit=Unit)


def migrate_add_role_and_calendar_token():
    return add_role_and_calendar_token(db=db, Staff=Staff)


def migrate_add_assignment_annotation():
    return add_assignment_annotation(db=db)


def migrate_add_unique_assignment_key():
    return add_unique_assignment_key(db=db)


def migrate_add_perf_indexes():
    return add_performance_indexes(db=db, app=app)


def migrate_add_requirement_req_d():
    return add_columns_if_missing(
        db=db,
        table="requirement",
        columns={"req_d": "req_d INTEGER DEFAULT 0"},
    )


def migrate_add_ut_flags():
    return add_columns_if_missing(
        db=db,
        table="staff",
        columns={
            "tower_ut": "tower_ut BOOLEAN DEFAULT 0",
            "radar_ut": "radar_ut BOOLEAN DEFAULT 0",
        },
    )


def migrate_add_is_training():
    return add_columns_if_missing(
        db=db,
        table="shift_type",
        columns={"is_training": "is_training BOOLEAN DEFAULT 0"},
    )


def migrate_add_wm_dwm_exclude():
    return add_columns_if_missing(
        db=db,
        table="staff",
        columns={
            "is_wm": "is_wm BOOLEAN DEFAULT 0",
            "is_dwm": "is_dwm BOOLEAN DEFAULT 0",
            "exclude_from_ot": "exclude_from_ot BOOLEAN DEFAULT 0",
        },
    )


def migrate_add_phone_number():
    return add_columns_if_missing(
        db=db,
        table="staff",
        columns={"phone_number": "phone_number VARCHAR(30) DEFAULT ''"},
    )


def migrate_add_watch_pattern_configuration():
    return add_watch_pattern_configuration(db=db)


def migrate_add_invitation_target():
    return add_invitation_target(db=db)


def migrate_add_toil_half_days_and_convert():
    return add_toil_and_leave_fields(db=db)


def ensure_shift(code, name, start=None, end=None, is_working=False, is_training=False):
    return ensure_bootstrap_shift(
        code,
        name,
        db=db,
        ShiftType=ShiftType,
        start=start,
        end=end,
        is_working=is_working,
        is_training=is_training,
    )


def ensure_watch(name: str, order_index: int):
    return ensure_bootstrap_watch(
        name, order_index, db=db, Watch=Watch
    )


def seed_once():
    return seed_legacy_operational_data(
        db=db,
        Unit=Unit,
        Watch=Watch,
        ShiftType=ShiftType,
        Staff=Staff,
    )

# -------------------- Small parse & AI helpers --------------------


def _is_empty_like(val) -> bool:
    """Treat '', '-', and em-dash as empty cells the AI may fill."""
    return str(val or "").strip() in {"", "-", "—"}


def _allocate_days_for_date(
    d: date,
    req,
    staff: list,                     # list[Staff]
    by_staff_day: dict,              # dict[int, dict[date, Assignment]]
    day_code_mon_sat: str,
    day_code_sun: str,
) -> int:
    return allocate_day_shift_shortfall(
        d,
        req,
        staff,
        by_staff_day,
        day_code_mon_sat,
        day_code_sun,
        db=db,
        Assignment=Assignment,
        is_working_day_code=_is_working_day_code,
        has_leave_or_sickness=_has_leave_or_sick,
        is_empty_like=_is_empty_like,
        passes_fatigue=_passes_fatigue_for,
        set_code=_set_code,
    )


def _parse_hhmm(val: str):
    return parse_roster_hhmm(val)


def _parse_date(val: str):
    return parse_roster_date(val)


def _normalise_phone_number(val: str | None) -> str:
    return normalise_phone_number(val)


def parse_annotation(s: str):
    return parse_roster_annotation(
        s,
        get_annotation_config=get_annotation_config,
        annotation_snapshot=_annotation_snapshot,
        current_unit_id=_current_unit_id,
    )


def _context_month_for_date(d: date | None) -> str | None:
    return context_month_for_date(d)


def log_change(entity_type: str, entity_id: int, field: str, old, new, note: str = "", context_day: date | None = None):
    record_change(
        db=db, ChangeLog=ChangeLog, user=current_user, now=utcnow,
        entity_type=entity_type, entity_id=entity_id, field=field,
        old=old, new=new, note=note, context_day=context_day,
    )

# --- Month math (no dateutil) ---


def _month_add(y: int, m: int, delta: int) -> Tuple[int, int]:
    return roster_period_add(y, m, delta, add_months)


def lock_date_for_month(y: int, m: int) -> date:
    return roster_period_lock_date(y, m, roster_lock_date)


def is_month_locked(y: int, m: int, today: Optional[date] = None) -> bool:
    return roster_period_is_locked(y, m, today, roster_month_is_locked)


def _assignment(staff_id: int, d: date) -> "Assignment":
    return assignment_for_day(db, Assignment, staff_id, d)


def _cell_is_protected(a: "Assignment") -> bool:
    return cell_is_protected(a)


def _set_code(a: "Assignment", code: str, source: str, note: str = "", ctx_month: Optional[str] = None):
    return set_assignment_code(a, code, source, note, _invalidate_month_cache_for_day, log_change)


def _has_leave_or_sick(staff_id: int, d: date) -> bool:
    return has_leave_or_sickness(Leave, Sickness, staff_id, d)


def _fatigue_ok(staff: "Staff", day: date, code: str) -> bool:
    return assignment_is_fatigue_safe(staff, day, code, would_trigger_fatigue)

# Back-compat shim so all AI code can call the same name


def _passes_fatigue_for(staff: "Staff", day: date, code: str) -> bool:
    return _fatigue_ok(staff, day, code)


def _weekday_is_sun(d: date) -> bool:
    return roster_date_is_sunday(d)

# ---------- Shift code helpers ----------


def _normalize_code(code) -> str:
    return normalize_roster_code(code)


def _is_non_working(code: str) -> bool:
    return roster_code_is_non_working(code, get_non_working_codes)


def _is_working_code_prefix(code: str, prefix: str) -> bool:
    return roster_code_is_working_with_prefix(
        code, prefix, get_non_working_codes, get_shift,
    )


def _is_working_day_code(code: str) -> bool:
    return _is_working_code_prefix(code, "D")


def _is_working_m_code(code: str) -> bool:
    return _is_working_code_prefix(code, "M")


def _is_working_n_code(code: str) -> bool:
    return _is_working_code_prefix(code, "N")

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not is_admin_user(current_user):
            abort(403)
        return f(*args, **kwargs)
    return wrapper



def _clamp_prev_next(year, month):
    """Clamp navigation so you cannot go earlier than MIN_MONTH."""
    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)
    prev_allowed = date(prev_y, prev_m, 1) >= date(
        MIN_MONTH.year, MIN_MONTH.month, 1)
    return (f"{prev_y}-{prev_m:02d}" if prev_allowed else None,
            f"{next_y}-{next_m:02d}")


@app.context_processor
def inject_perms():
    return build_navigation_context(
        current_user,
        request.endpoint,
        NavigationContextDependencies(
            db=db,
            Unit=Unit,
            Staff=Staff,
            ShiftRequest=ShiftRequest,
            FeatureFlag=FeatureFlag,
            Notification=Notification,
            BriefingDelivery=BriefingDelivery,
            BriefingItem=BriefingItem,
            is_admin_user=is_admin_user,
            is_editor_user=is_editor_user,
            briefing_enabled=briefing_enabled,
            training_enabled=training_enabled,
            competency_enabled=competency_enabled,
            live_position_enabled=live_position_enabled,
            briefing_local_now=briefing_local_now,
        ),
    )


# -------------------- Admin --------------------


def _admin_action_dependencies():
    return AdminActionDependencies(
        db=db,
        Watch=Watch,
        Staff=Staff,
        WorkPattern=WorkPattern,
        StaffWatchHistory=StaffWatchHistory,
        StaffPatternAssignment=StaffPatternAssignment,
        QualificationType=QualificationType,
        PersonQualification=PersonQualification,
        ShiftType=ShiftType,
        Requirement=Requirement,
        SpecialRequirement=SpecialRequirement,
        RosterImpactEventType=RosterImpactEventType,
        JoinerDependencies=JoinerDependencies,
        WatchConfigurationDependencies=WatchConfigurationDependencies,
        ShiftConfigurationDependencies=ShiftConfigurationDependencies,
        current_unit_id=_current_unit_id,
        validate_csrf=_validate_csrf,
        update_absence_types=update_absence_types,
        get_absence_types=get_absence_types,
        save_absence_types=_save_absence_types,
        save_sms_settings=save_sms_settings,
        parse_sms_number_lines=_parse_sms_number_lines,
        save_roster_setting=_save_roster_setting,
        update_unit_roster_setup=update_unit_roster_setup,
        validate_pattern=_validated_pattern,
        parse_date=_parse_date,
        record_roster_impact=record_roster_impact,
        update_watch_configuration=update_watch_configuration,
        save_counter_mapping=save_counter_mapping,
        create_joiner=create_joiner,
        work_pattern_service=work_pattern_service,
        record_qualification_history=_record_qualification_history,
        sync_qualification=_sync_qualification_to_roster_profile,
        now=utcnow,
        update_shift_definition=update_shift_definition,
        parse_hhmm=_parse_hhmm,
        prune_roster_code_settings=_prune_roster_code_settings,
        refresh_shift_cache=refresh_shift_cache,
        clear_shift_groups_cache=_shift_groups_snapshot.cache_clear,
        save_monthly_requirements=save_monthly_requirements,
        save_special_requirement=save_special_requirement,
        delete_special_requirement=delete_special_requirement,
        seed_toil_balances=seed_toil_balances,
    )


@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():

    if request.method == "POST":
        form = request.form.get("form", "")
        response = dispatch_admin_action(
            form, request.form, _admin_action_dependencies()
        )
        if response is not None:
            return response

    return render_template(
        "admin.html",
        **build_admin_context(AdminContextDependencies(
            db=db,
            Watch=Watch,
            ShiftType=ShiftType,
            QualificationType=QualificationType,
            WorkPattern=WorkPattern,
            Staff=Staff,
            Requirement=Requirement,
            SpecialRequirement=SpecialRequirement,
            Leave=Leave,
            Unit=Unit,
            current_unit_id=_current_unit_id,
            roster_settings_snapshot=_roster_settings_snapshot,
            validate_pattern=_validated_pattern,
            shift_counter_group=shift_counter_group,
            sms_number_options=_sms_number_options,
            sms_operational_options=_sms_operational_options,
            sms_default_number=_sms_default_number,
            absence_types=get_absence_types,
            default_base_pattern=DEFAULT_BASE_PATTERN,
            pattern_codes=PATTERN_CODES,
        )),
    )



# Keep your dedicated staff edit route (ATCO edit)








# -------------------- Leave / Sickness / TOIL --------------------


def _group_sickness_instances(assignments, month_start=None, month_end=None):
    return group_sickness_instances(assignments, month_start, month_end)


# -------------------- Staff profile --------------------


def _training_profile_allowed(person):
    return bool(
        person.id == current_user.id
        or is_editor_user(current_user)
        or can_manage_training(current_user)
        or can_record_training(current_user)
    )





# -------------------- Metrics + CSV (date range; FYTD default) --------------------
# (… unchanged metrics functions from your file …)


def _compute_metrics_range(
    start_day: date, end_day: date, watch_id: int | None = None
):
    return compute_annotation_metrics(
        start_day,
        end_day,
        watch_id=watch_id,
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        annotation_items=_annotation_snapshot(
            int(_current_unit_id() or 1)
        )["items"],
        parse_annotation=parse_annotation,
    )


def _compute_fairness_range(start_day: date, end_day: date):
    return FairnessReportService(
        FairnessDependencies(
            Assignment=Assignment,
            BankHoliday=BankHoliday,
            ChangeLog=ChangeLog,
            ShiftType=ShiftType,
            Staff=Staff,
            FairnessAssignment=FairnessAssignment,
            FairnessStaff=FairnessStaff,
            current_unit_id=_current_unit_id,
            work_pattern_service=work_pattern_service,
            code_from_pattern=code_from_pattern,
            shift_duration_minutes=shift_duration_minutes,
            calculate_fairness=calculate_fairness,
        )
    ).compute(start_day, end_day)


def _fy_start_for(d: date) -> date:
    return financial_year_start(d)


def _count_aava_soal_since_prev_april(staff_id: int, upto: date):
    counts = count_tagged_assignments(
        staff_id,
        upto,
        ("aava", "soal"),
        Assignment=Assignment,
        parse_annotation=parse_annotation,
        annotation_tags_for=annotation_tags_for,
    )
    return counts["aava"], counts["soal"]


def _worked_like_consecutive_days(staff: Staff, upto_day: date, lookback_days: int = 10) -> int:
    return worked_like_consecutive_days(
        staff,
        upto_day,
        Assignment=Assignment,
        working_codes=get_working_codes,
        lookback_days=lookback_days,
    )


def _had_sc_within_48h(staff: Staff, ref_day: date, ref_shift: ShiftType) -> bool:
    return had_sickness_within_48_hours(
        staff,
        ref_day,
        ref_shift,
        Assignment=Assignment,
        span=_span,
        get_shift=get_shift,
    )


def _has_in_date_ue(s: Staff, ref_day: date) -> bool:
    return has_in_date_endorsement(s, ref_day)


# -------------------- Overtime finder (admin/editor) --------------------
# (… unchanged from your file …)


def _count_ot_since_prev_april(staff_id: int, upto: date):
    return count_tagged_assignments(
        staff_id,
        upto,
        ("ot",),
        Assignment=Assignment,
        parse_annotation=parse_annotation,
        annotation_tags_for=annotation_tags_for,
    )["ot"]

# … keep the rest of your overtime helpers exactly as pasted …


def _compute_overtime_candidates(chosen_date: date | None, chosen_shift_code: str):
    return _overtime_candidate_service.compute(chosen_date, chosen_shift_code)



def _leave_summary_for_month(year: int, month: int, watch_id: int | None = None):
    return leave_summary_for_month(
        year,
        month,
        watch_id,
        unit_id=_current_unit_id(),
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        month_range=month_range,
        active_leave_types=get_absence_types("leave", active_only=True),
    )


# ===== Leave-Year report (per-person config; AL only; includes TOIL days) =====
# (unchanged from your post)

def _current_leave_year_window(s: Staff, today: date | None = None):
    return current_leave_year_window(s, today)


def _toil_accrual_half_days_from_annotation(parsed):
    return annotation_accrual_half_days(
        parsed, annotation_config=get_annotation_config
    )


def _record_toil_transaction(
    person_id: int,
    delta_half_days: int,
    reason: str,
    actor_id: int,
    transaction_key: str | None = None,
    source_type: str = "manual",
    source_id: int | None = None,
):
    return apply_toil_transaction(
        db,
        Staff,
        ToilTransaction,
        unit_id=_current_unit_id(),
        person_id=person_id,
        delta_half_days=delta_half_days,
        reason=reason,
        actor_id=actor_id,
        utcnow=utcnow,
        transaction_key=transaction_key,
        source_type=source_type,
        source_id=source_id,
    )


def _apply_toil_annotation_delta(
    staff: Staff,
    old_annot: str,
    new_annot: str,
    *,
    actor_id: int,
    transaction_key: str | None = None,
    source_id: int | None = None,
):
    return apply_annotation_toil_delta(
        staff,
        old_annot,
        new_annot,
        actor_id=actor_id,
        parse_annotation=parse_annotation,
        accrual_half_days=_toil_accrual_half_days_from_annotation,
        record_transaction=_record_toil_transaction,
        transaction_key=transaction_key,
        source_id=source_id,
    )


def _toil_accrued_used_in_range_half_days(staff_id: int, start_day: date, end_day: date):
    return accrued_and_used_half_days(
        staff_id,
        start_day,
        end_day,
        Assignment=Assignment,
        parse_annotation=parse_annotation,
        accrual_half_days=_toil_accrual_half_days_from_annotation,
    )


# ===== Sickness Report (unchanged) =====


def _group_consecutive_days(days_set):
    return group_consecutive_days(days_set)


# -------------------- Request Sheets (shift requests) --------------------


def _unit_request_rules(unit_id: int | None = None) -> tuple[int, int]:
    unit = db.session.get(Unit, unit_id or _current_unit_id())
    return normalise_request_rules(
        getattr(unit, "request_months_ahead", 3),
        getattr(unit, "request_lock_day", 20),
    )


def _lock_date_for_target_month(y: int, m: int, unit_id: int | None = None):
    _, lock_day = _unit_request_rules(unit_id)
    return request_lock_date(y, m, lock_day)


def _is_month_locked(y: int, m: int, today: date | None = None, unit_id: int | None = None):
    _, lock_day = _unit_request_rules(unit_id)
    return request_month_is_locked(y, m, lock_day, today)


def _add_months(first: date, count: int) -> date:
    return add_request_months(first, count)


def _request_date_bounds(today: date, unit_id: int) -> tuple[date, date]:
    months, _ = _unit_request_rules(unit_id)
    return request_date_bounds(today, months)


def _request_audit(req: ShiftRequest, actor_id: int, transition: str,
                   old_value: object, new_value: object, reason: str = "") -> None:
    db.session.add(RequestAudit(
        unit_id=req.unit_id,
        request_id=req.id,
        actor_id=actor_id,
        transition=transition,
        old_value=json.dumps(old_value, default=str, sort_keys=True),
        new_value=json.dumps(new_value, default=str, sort_keys=True),
        reason=(reason or "")[:500],
    ))


def _notify_requester(req: ShiftRequest) -> None:
    if req.status not in {"pending", "approved", "rejected", "fulfilled"}:
        return
    if req.status == "fulfilled":
        outcome = "was approved and added to the roster"
    elif req.status == "rejected":
        outcome = "was refused"
    else:
        outcome = f"is now {req.status}"
    comment = (req.admin_response or "").strip()
    comment_text = f" Manager comment: {comment}" if comment else ""
    db.session.add(Notification(
        unit_id=req.unit_id,
        recipient_id=req.staff_id,
        kind=f"shift_request_{req.status}",
        message=(
            f"Your {req.code} shift request for "
            f"{req.day.strftime('%d %B %Y')} {outcome}.{comment_text}"
        ),
    ))


def _safe_request_admin_month(raw_value: str | None, fallback: date) -> str:
    return safe_admin_month(raw_value, fallback)


def staff_has_qualification(
    staff: Staff, qualification_code: str, duty_date: date
) -> bool:
    return qualification_status_for_staff(
        staff,
        qualification_code,
        duty_date,
        QualificationType=QualificationType,
        PersonQualification=PersonQualification,
        authenticated_unit_id=authenticated_unit_id,
    )


def _staff_has_shift_qualification(
    staff: Staff, shift: ShiftType, duty_date: date | None = None
) -> bool:
    return staff_has_qualification(
        staff,
        shift.required_qualification,
        duty_date or date.today(),
    )


_overtime_candidate_service = OvertimeCandidateService(
    OvertimeCandidateDependencies(
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        current_unit_id=_current_unit_id,
        get_shift=get_shift,
        ensure_assignments_for_range=ensure_assignments_for_range,
        annotation_codes_for_tag=annotation_codes_for_tag,
        get_annotation_config=get_annotation_config,
        staff_has_shift_qualification=_staff_has_shift_qualification,
        has_in_date_ue=_has_in_date_ue,
        worked_like_consecutive_days=_worked_like_consecutive_days,
        would_create_new_fatigue_issues=would_create_new_fatigue_issues,
        count_aava_soal=_count_aava_soal_since_prev_april,
        had_sc_within_48h=_had_sc_within_48h,
    )
)




# >>> Admin can respond to a specific request








_signup_saga_dependencies = SignupSagaDependencies(
    db=db,
    ShiftType=ShiftType,
    SignupWorkflow=SignupWorkflow,
    PlatformIdentity=PlatformIdentity,
    Staff=Staff,
    UnitMembership=UnitMembership,
    Unit=Unit,
    SecureInvitation=SecureInvitation,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    now=utcnow,
    valid_email=_valid_email,
    password_hash=generate_password_hash,
)


def _normalized_login(value: str) -> str:
    return normalized_login(value)


def _run_invitation_signup(
    invitation, unit, name, username, password, email="", fail_after=None,
):
    return run_invitation_signup(
        _signup_saga_dependencies,
        invitation,
        unit,
        name,
        username,
        password,
        email,
        fail_after,
    )






def _qualification_snapshot(record: PersonQualification) -> dict:
    return qualification_snapshot(record)


def _record_qualification_history(
    record: PersonQualification, action: str
) -> None:
    return add_qualification_history(
        record,
        action,
        db=db,
        PersonQualificationHistory=PersonQualificationHistory,
        actor_id=current_user.id,
    )


def _sync_qualification_to_roster_profile(
    person: Staff, qtype: QualificationType, expires_on: date | None
) -> None:
    return sync_legacy_roster_profile(person, qtype, expires_on)




def _valid_endorsement(person_id: int, position_id: int, on_day: date) -> bool:
    return has_valid_endorsement(
        person_id,
        position_id,
        on_day,
        PositionEndorsement=PositionEndorsement,
    )


def _position_assurance(year: int, month: int) -> list[dict]:
    return monthly_position_assurance(
        year,
        month,
        Assignment=Assignment,
        OperationalPosition=OperationalPosition,
        PositionRequirement=PositionRequirement,
        month_range=month_range,
        valid_endorsement=_valid_endorsement,
    )




LOGIN_RATE_WINDOW = timedelta(minutes=15)
LOGIN_RATE_LIMIT = 10


def _login_rate_key(username: str) -> str:
    return privacy_rate_limit_key(
        str(app.config["SECRET_KEY"]), "login", request.remote_addr or "unknown",
        username.lower(), privacy_key,
    )


def _consume_rate_limit(
    scope: str, subject: object, limit: int = LOGIN_RATE_LIMIT,
    window: timedelta = LOGIN_RATE_WINDOW, fail_closed: bool = True,
) -> bool:
    return consume_rate_limit(
        limiter=_rate_limiter,
        key=privacy_rate_limit_key(str(app.config["SECRET_KEY"]), scope, request.remote_addr or "unknown", subject, privacy_key),
        limit=limit, window=window, unavailable=LimiterUnavailable,
        security_event=_security_event, scope=scope, fail_closed=fail_closed,
    )


def _reset_rate_limit(scope: str, subject: object) -> None:
    reset_rate_limit(
        limiter=_rate_limiter,
        key=privacy_rate_limit_key(str(app.config["SECRET_KEY"]), scope, request.remote_addr or "unknown", subject, privacy_key),
        unavailable=LimiterUnavailable, security_event=_security_event, scope=scope,
    )


def _security_event(event: str, **safe_fields) -> None:
    record_security_event(
        metrics=_operational_metrics,
        logger=app.logger,
        request_id=getattr(g, "request_id", ""),
        structured_event=structured_event,
        event=event,
        **safe_fields,
    )


def _credential_for_auth_stamp(user):
    return credential_for_auth_stamp(user, PlatformMfaCredential, MfaCredential)


_session_lifecycle = SessionLifecycle(
    SessionLifecycleDependencies(
        now=utcnow,
        credential_for_user=_credential_for_auth_stamp,
        security_event=lambda event, **facts: _security_event(event, **facts),
    )
)
_current_auth_stamp = _session_lifecycle.auth_stamp
_initialize_authenticated_session = _session_lifecycle.initialize


def _central_security_event(
    event_type: str, outcome: str, identity_id: int | None = None,
    principal: str = "", detail: str = "",
) -> None:
    record_central_security_event(
        db,
        CentralSecurityAudit,
        event_type,
        outcome,
        identity_id,
        principal,
        detail,
    )


def _record_successful_login(user: Staff) -> None:
    record_successful_login(
        db=db,
        PlatformIdentity=PlatformIdentity,
        Unit=Unit,
        AggregateUsageEvent=AggregateUsageEvent,
        user=user,
        now=utcnow,
    )


def _active_recovery_from_digest(
    field_name: str, raw_token: str, expected_state: str
):
    return active_recovery_from_digest(
        RecoveryRequest, field_name, raw_token, expected_state, utcnow
    )

def _decrypt_mfa_secret(credential) -> str:
    return decrypt_secret(credential, _decrypt_field)


def _matching_totp_step(secret: str, code: str) -> int | None:
    return matching_totp_step(secret, code, utcnow)


def _pending_platform_login():
    identity_id = int(session.get("_platform_mfa_identity_id") or 0)
    user_id = int(session.get("_platform_mfa_user_id") or 0)
    if not identity_id or user_id != identity_id:
        return None, None
    identity = db.session.get(PlatformIdentity, identity_id)
    if not identity or identity.role != "superadmin":
        return None, None
    return identity, identity


def _complete_platform_login(identity, user, recovery_used=False):
    next_url = session.get("_platform_mfa_next", "")
    session.clear()
    login_user(user)
    _initialize_authenticated_session(user, platform_mfa=True)
    identity.last_active_at = utcnow()
    _central_security_event(
        "platform_recovery_code_used" if recovery_used
        else "platform_mfa_verified",
        "success", identity.id,
        hashlib.sha256(identity.username.lower().encode()).hexdigest()[:16],
    )
    db.session.commit()
    return redirect(_canonical_login_redirect(
        next_url,
        default_endpoint="platform_admin",
        user_id=user.id,
    ))


def _totp_qr_data_uri(provisioning_uri: str) -> str:
    return totp_qr_data_uri(provisioning_uri)


for cli_command in create_cli_commands(CliDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    PlatformMfaCredential=PlatformMfaCredential,
    Unit=Unit,
    SignupWorkflow=SignupWorkflow,
    SecureInvitation=SecureInvitation,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    Staff=Staff,
    UnitMembership=UnitMembership,
    MfaCredential=MfaCredential,
    now=utcnow,
    central_security_event=_central_security_event,
    encrypt_field=_encrypt_field,
    decrypt_field=_decrypt_field,
    generate_password_hash=generate_password_hash,
    operational_unit_context=operational_unit_context,
)):
    app.cli.add_command(cli_command)


# -------------------- DB init (single, safe block) --------------------

with app.app_context():
    if (
        DEPLOYMENT_ENV != "production"
        and os.environ.get("ATCROSTER_SKIP_RUNTIME_SCHEMA") != "1"
    ):
        db.create_all()
        seed_once()
        refresh_shift_cache()

# Expose helpers & models needed by Jinja templates that refer to them directly
app.jinja_env.globals['month_range'] = month_range
app.jinja_env.globals['ShiftType'] = ShiftType

# -------------------- Run --------------------

work_pattern_service = WorkPatternService(WorkPatternDependencies(
    Staff=Staff,
    ShiftType=ShiftType,
    Leave=Leave,
    Assignment=Assignment,
    WorkPattern=WorkPattern,
    WorkPatternDay=WorkPatternDay,
    WorkPatternDayAllowedShift=WorkPatternDayAllowedShift,
    StaffPatternAssignment=StaffPatternAssignment,
    StaffRule=StaffRule,
    shift_group=lambda shift: shift_counter_group(shift.code, shift.unit_id),
))
get_pattern_day_for_staff = work_pattern_service.get_pattern_day_for_staff
get_effective_staff_rules = work_pattern_service.get_effective_staff_rules
is_staff_eligible_for_shift = work_pattern_service.is_staff_eligible_for_shift
calculate_soft_rule_penalty = work_pattern_service.calculate_soft_rule_penalty

work_pattern_admin_service = WorkPatternAdminService(
    WorkPatternAdminDependencies(
        db=db,
        WorkPattern=WorkPattern,
        WorkPatternDay=WorkPatternDay,
        WorkPatternDayAllowedShift=WorkPatternDayAllowedShift,
        ShiftType=ShiftType,
        pattern_service=work_pattern_service,
    )
)
roster_validation_service = RosterValidationService(
    RosterValidationDependencies(
        Staff=Staff,
        ShiftType=ShiftType,
        Assignment=Assignment,
        StaffPatternAssignment=StaffPatternAssignment,
        StaffRule=StaffRule,
        work_pattern_service=work_pattern_service,
    )
)
roster_proposal_service = RosterProposalService(
    RosterProposalDependencies(
        db=db,
        Staff=Staff,
        ShiftType=ShiftType,
        Assignment=Assignment,
        Sickness=Sickness,
        Requirement=Requirement,
        SpecialRequirement=SpecialRequirement,
        RosterProposal=RosterProposal,
        RosterProposalAssignment=RosterProposalAssignment,
        ChangeLog=ChangeLog,
        work_pattern_service=work_pattern_service,
        requirements_for_day=requirements_for_day,
        shift_group_for_day=shift_counter_group_for_day,
        shift_minutes=shift_duration_minutes,
        staff_is_countable_on=staff_is_countable_on,
        staff_has_qualification=_staff_has_shift_qualification,
        would_trigger_fatigue=would_trigger_fatigue_with_plan,
        compute_fairness_range=_compute_fairness_range,
        utcnow=utcnow,
    )
)
override_classification_service = OverrideClassificationService(
    OverrideClassificationDependencies(
        Assignment=Assignment, Staff=Staff, ShiftType=ShiftType,
        work_pattern_service=work_pattern_service,
    )
)
roster_period_service = RosterPeriodService(RosterPeriodDependencies(
    db=db, RosterPeriod=RosterPeriod, utcnow=utcnow,
))
work_pattern_migration_service = WorkPatternMigrationService(
    WorkPatternMigrationDependencies(
        db=db,
        Staff=Staff,
        WorkPattern=WorkPattern,
        WorkPatternDay=WorkPatternDay,
        ShiftType=ShiftType,
        StaffPatternAssignment=StaffPatternAssignment,
        pattern_context=_pattern_context,
        pattern_service=work_pattern_service,
    )
)
app.register_blueprint(create_work_pattern_blueprint(
    WorkPatternBlueprintDependencies(
        db=db,
        Staff=Staff,
        ShiftType=ShiftType,
        WorkPattern=WorkPattern,
        WorkPatternDay=WorkPatternDay,
        WorkPatternDayAllowedShift=WorkPatternDayAllowedShift,
        StaffPatternAssignment=StaffPatternAssignment,
        StaffRule=StaffRule,
        BankHoliday=BankHoliday,
        is_admin_user=is_admin_user,
        current_unit_id=_current_unit_id,
        validate_csrf=_validate_csrf,
        pattern_service=work_pattern_service,
        admin_service=work_pattern_admin_service,
        migration_service=work_pattern_migration_service,
        record_roster_impact=record_roster_impact,
    )
))

app.register_blueprint(create_live_position_blueprint(LivePositionDependencies(
    db=db, Unit=Unit, OperationalPosition=OperationalPosition,
    OperationalPositionTimeAllowance=OperationalPositionTimeAllowance,
    OperationalPositionGroup=OperationalPositionGroup,
    PositionCurrencyCategory=PositionCurrencyCategory,
    PositionStatusEvent=PositionStatusEvent, PositionSession=PositionSession,
    PositionSessionParticipant=PositionSessionParticipant,
    PositionParticipantRole=PositionParticipantRole,
    PositionSessionAudit=PositionSessionAudit,
    PositionEndorsement=PositionEndorsement, Staff=Staff,
    Watch=Watch,
    utcnow=utcnow, is_admin_user=is_admin_user,
    live_position_enabled=live_position_enabled,
    competency_enabled=competency_enabled,
    authenticated_database_route_optional=authenticated_database_route_optional,
    authenticated_unit_context=authenticated_unit_context,
)))

app.register_blueprint(create_handover_blueprint(HandoverDependencies(
    db=db, Unit=Unit, Staff=Staff, ShiftType=ShiftType, Assignment=Assignment,
    Requirement=Requirement, SpecialRequirement=SpecialRequirement,
    FeatureFlag=FeatureFlag, HandoverField=HandoverField,
    HandoverRecord=HandoverRecord,
    HandoverOperationalState=HandoverOperationalState,
    HandoverEquipment=HandoverEquipment,
    OperationalPosition=OperationalPosition, PositionSession=PositionSession,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf, is_admin_user=is_admin_user,
    is_editor_user=is_editor_user, requirements_for_day=requirements_for_day,
    shift_group_for_day=shift_counter_group_for_day, utcnow=utcnow,
    live_position_enabled=live_position_enabled,
)))

app.register_blueprint(create_auth_blueprint(AuthDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    UnitMembership=UnitMembership,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    Staff=Staff,
    Unit=Unit,
    PlatformMfaCredential=PlatformMfaCredential,
    MfaCredential=MfaCredential,
    validate_csrf=_validate_csrf,
    normalized_login=_normalized_login,
    login_rate_key=_login_rate_key,
    consume_rate_limit=_consume_rate_limit,
    reset_rate_limit=_reset_rate_limit,
    security_event=_security_event,
    central_security_event=_central_security_event,
    bind_authenticated_unit=bind_authenticated_unit,
    canonical_login_redirect=_canonical_login_redirect,
    airport_login_endpoint=_airport_login_endpoint,
    initialize_authenticated_session=_initialize_authenticated_session,
    record_successful_login=_record_successful_login,
)))
app.register_blueprint(create_mfa_blueprint(MfaRouteDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    PlatformMfaCredential=PlatformMfaCredential,
    Staff=Staff,
    MfaCredential=MfaCredential,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    deployment_environment=DEPLOYMENT_ENV,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    decrypt_secret=_decrypt_mfa_secret,
    matching_totp_step=_matching_totp_step,
    encrypt_field=_encrypt_field,
    now=utcnow,
    central_security_event=_central_security_event,
    bind_authenticated_unit=bind_authenticated_unit,
    initialize_authenticated_session=_initialize_authenticated_session,
    security_event=_security_event,
    record_successful_login=_record_successful_login,
    canonical_login_redirect=_canonical_login_redirect,
    current_unit_id=_current_unit_id,
    current_auth_stamp=_current_auth_stamp,
    totp_qr_data_uri=_totp_qr_data_uri,
)))
app.register_blueprint(create_qualification_blueprint(QualificationDependencies(
    db=db,
    Staff=Staff,
    QualificationType=QualificationType,
    PersonQualification=PersonQualification,
    PersonQualificationHistory=PersonQualificationHistory,
    RosterImpactEventType=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    is_editor_user=is_editor_user,
    is_admin_user=is_admin_user,
    now=utcnow,
    qualification_impact_type=_qualification_impact_type,
    person_has_other_valid_ue=_person_has_other_valid_ue,
    record_roster_impact=record_roster_impact,
)))
app.register_blueprint(create_reports_blueprint(ReportsDependencies(
    Assignment=Assignment,
    Staff=Staff,
    Watch=Watch,
    is_admin_user=is_admin_user,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    compute_metrics_range=_compute_metrics_range,
    financial_year_start=_fy_start_for,
    parse_year_month=parse_ym,
    ensure_month_requirement=ensure_month_requirement,
    generate_month=generate_month,
    leave_summary_for_month=_leave_summary_for_month,
    current_leave_year_window=_current_leave_year_window,
    toil_accrued_used=_toil_accrued_used_in_range_half_days,
    group_consecutive_days=_group_consecutive_days,
    get_absence_types=get_absence_types,
    compute_fairness_range=_compute_fairness_range,
    live_position_enabled=live_position_enabled,
    competency_enabled=competency_enabled,
    operational_currency_shortfalls=_operational_currency_shortfalls,
)))
publication_service = create_publication_service(PublicationDependencies(
    db=db,
    Assignment=Assignment,
    RosterPublication=RosterPublication,
    Staff=Staff,
    Requirement=Requirement,
    RosterRuleVersion=RosterRuleVersion,
    FatigueReport=FatigueReport,
    OperationalPosition=OperationalPosition,
    PositionRequirement=PositionRequirement,
    BreakPlan=BreakPlan,
    Unit=Unit,
    current_unit_id=_current_unit_id,
    now=utcnow,
    month_add=_month_add,
    month_range=month_range,
    is_admin_user=is_admin_user,
    normalise_snapshot=normalise_assignment_snapshot,
    get_shift=get_shift,
    staff_has_shift_qualification=_staff_has_shift_qualification,
    excluded_codes=get_exclude_from_counters,
    staff_is_countable_on=staff_is_countable_on,
    shift_counter_group_for_day=shift_counter_group_for_day,
    night_active_on=_night_active_on,
    compliance_findings=_compliance_findings,
    position_assurance=_position_assurance,
    valid_email=_valid_email,
    send_account_email=_send_account_email,
))
app.register_blueprint(create_roster_blueprint(RosterDependencies(
    db=db,
    RosterPublication=RosterPublication,
    Staff=Staff,
    Notification=Notification,
    Assignment=Assignment,
    Leave=Leave,
    Watch=Watch,
    Requirement=Requirement,
    SpecialRequirement=SpecialRequirement,
    ShiftRequest=ShiftRequest,
    AnnotationType=AnnotationType,
    AnnotationAudit=AnnotationAudit,
    can_publish_roster=publication_service.can_publish,
    validate_csrf=_validate_csrf,
    parse_year_month=parse_ym,
    current_unit_id=_current_unit_id,
    month_has_data=month_has_data,
    ensure_month_requirement=ensure_month_requirement,
    generate_month=generate_month,
    active_publication=publication_service.active_publication,
    publication_matches_live=publication_service.matches_live,
    roster_snapshot=publication_service.snapshot,
    utcnow=utcnow,
    send_publication_emails=publication_service.send_emails,
    log_change=log_change,
    consume_rate_limit=_consume_rate_limit,
    month_range=month_range,
    requirements_for_day=requirements_for_day,
    staff_is_countable_on=staff_is_countable_on,
    operational_capability_matrix=get_operational_capability_matrix,
    exclude_from_counters=get_exclude_from_counters,
    get_shift=get_shift,
    shift_counter_group_for_day=shift_counter_group_for_day,
    night_active_on=_night_active_on,
    can_edit_roster=can_edit_roster,
    banned_roster_codes=get_banned_roster_codes,
    can_apply_annotations=can_apply_annotations,
    parse_annotation=parse_annotation,
    is_admin_user=is_admin_user,
    apply_toil_annotation_delta=_apply_toil_annotation_delta,
    load_month_roster=_load_month_roster_fast,
    add_months=_month_add,
    shift_groups=_shift_groups_snapshot,
    watch_ids_for_staff_on=watch_ids_for_staff_on,
    roster_fatigue_flags=roster_fatigue_flags_for_range,
    roster_fatigue_matrix=roster_fatigue_flags_matrix,
    roster_validation=roster_validation_service,
    roster_month_cache=roster_month_cache,
    metrics=_operational_metrics,
    RosterProposal=RosterProposal,
    RosterProposalAssignment=RosterProposalAssignment,
    roster_proposal_service=roster_proposal_service,
    get_annotation_groups=get_annotation_groups,
    lock_roster_month=_lock_roster_month,
)))
app.register_blueprint(create_absence_requests_blueprint(
    AbsenceRequestDependencies(
        db=db,
        Staff=Staff,
        Watch=Watch,
        Leave=Leave,
        Assignment=Assignment,
        is_admin_user=is_admin_user,
        parse_year_month=parse_ym,
        month_range=month_range,
        clamp_prev_next=_clamp_prev_next,
        validate_csrf=_validate_csrf,
        get_absence_types=get_absence_types,
        save_absence_types=_save_absence_types,
        tenant_get=tenant_get,
        current_unit_id=_current_unit_id,
        refresh_day_from_pattern_and_leave=refresh_day_from_pattern_and_leave,
        group_sickness_instances=_group_sickness_instances,
        ShiftType=ShiftType,
        ShiftRequest=ShiftRequest,
        unit_request_rules=_unit_request_rules,
        request_date_bounds=_request_date_bounds,
        is_month_locked=_is_month_locked,
        request_audit=_request_audit,
        utcnow=utcnow,
        safe_request_admin_month=_safe_request_admin_month,
        request_statuses=REQUEST_STATUSES,
        request_transitions=REQUEST_TRANSITIONS,
        would_create_new_fatigue_issues=would_create_new_fatigue_issues,
        staff_has_shift_qualification=_staff_has_shift_qualification,
        can_override_roster_conflicts=can_override_roster_conflicts,
        notify_requester=_notify_requester,
        lock_roster_month=_lock_roster_month,
        record_toil_transaction=_record_toil_transaction,
    )
))
app.register_blueprint(create_training_blueprint(TrainingDependencies(
    db=db,
    Staff=Staff,
    TrainingLevel=TrainingLevel,
    TrainingSession=TrainingSession,
    TrainingScore=TrainingScore,
    current_unit_id=_current_unit_id,
    training_enabled=training_enabled,
    is_editor_user=is_editor_user,
    can_manage_training=can_manage_training,
    can_record_training=can_record_training,
    is_under_training=is_under_training,
    training_profile_allowed=_training_profile_allowed,
    validate_csrf=_validate_csrf,
    QualificationType=QualificationType,
    PersonQualification=PersonQualification,
    competency_enabled=competency_enabled,
    is_admin_user=is_admin_user,
    utcnow=utcnow,
    record_qualification_history=_record_qualification_history,
    sync_qualification_to_roster_profile=_sync_qualification_to_roster_profile,
    record_qualification_roster_impact=record_qualification_roster_impact,
    TrainingObjective=TrainingObjective,
)))
app.register_blueprint(create_fatigue_compliance_blueprint(
    FatigueComplianceDependencies(
        db=db,
        Unit=Unit,
        is_admin_user=is_admin_user,
        current_unit_id=_current_unit_id,
        validate_csrf=_validate_csrf,
        load_rule_config=_fatigue_rule_config,
        save_rule_config=_save_fatigue_rule_config,
    )
))
app.register_blueprint(create_operations_blueprint(OperationsDependencies(
    db=db,
    OperationalPosition=OperationalPosition,
    PositionEndorsement=PositionEndorsement,
    PositionRequirement=PositionRequirement,
    Staff=Staff,
    ShiftType=ShiftType,
    BreakPlan=BreakPlan,
    Assignment=Assignment,
    AchievedDuty=AchievedDuty,
    FatigueReport=FatigueReport,
    RosterRuleVersion=RosterRuleVersion,
    is_admin_user=is_admin_user,
    compliance_month=_compliance_month,
    validate_csrf=_validate_csrf,
    current_unit_id=_current_unit_id,
    utcnow=utcnow,
    log_change=log_change,
    month_add=_month_add,
    position_assurance=_position_assurance,
    can_edit_roster=can_edit_roster,
    parse_year_month=parse_ym,
    month_range=month_range,
    shift_counter_group_for_day=shift_counter_group_for_day,
    staff_has_shift_qualification=_staff_has_shift_qualification,
    Scenario=Scenario,
)))
app.register_blueprint(create_notification_blueprint(NotificationDependencies(
    db=db,
    Notification=Notification,
    current_unit_id=_current_unit_id,
    utcnow=utcnow,
    validate_csrf=_validate_csrf,
)))
app.register_blueprint(create_sms_administration_blueprint(
    SmsAdministrationDependencies(
        db=db,
        SmsAudit=SmsAudit,
        SmsSenderRegistration=SmsSenderRegistration,
        current_unit_id=_current_unit_id,
        is_admin_user=is_admin_user,
        validate_csrf=_validate_csrf,
        utcnow=utcnow,
    )
))
app.register_blueprint(create_messaging_blueprint(MessagingDependencies(
    db=db,
    Staff=Staff,
    Watch=Watch,
    Assignment=Assignment,
    SmsSenderRegistration=SmsSenderRegistration,
    current_unit_id=_current_unit_id,
    utcnow=utcnow,
    can_send_unit_messages=can_send_unit_messages,
    validate_csrf=_validate_csrf,
    sms_configuration=sms_configuration,
    normalise_sms_number=_normalise_sms_number,
    send_sms=_send_sms,
    record_sms_audit=_record_sms_audit,
    flash_sms_result=_flash_sms_result,
)))
app.register_blueprint(create_module_blueprint(ModuleDependencies(
    FeatureFlag=FeatureFlag,
    briefing_enabled=briefing_enabled,
    training_enabled=training_enabled,
    competency_enabled=competency_enabled,
    is_admin_user=is_admin_user,
)))
app.register_blueprint(create_calendar_feed_blueprint(CalendarFeedDependencies(
    Staff=Staff,
    Assignment=Assignment,
    get_shift=get_shift,
    db=db,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    validate_csrf=_validate_csrf,
)))
app.register_blueprint(create_administration_blueprint(AdministrationDependencies(
    is_admin_user=is_admin_user,
    live_position_enabled=live_position_enabled,
)))
app.register_blueprint(create_staff_edit_blueprint(StaffEditDependencies(
    db=db,
    Staff=Staff,
    Watch=Watch,
    QualificationType=QualificationType,
    PersonQualification=PersonQualification,
    UnitMembership=UnitMembership,
    PlatformIdentity=PlatformIdentity,
    SecureInvitation=SecureInvitation,
    RosterImpactEventType=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    parse_date=_parse_date,
    valid_email=_valid_email,
    normalise_phone=_normalise_phone_number,
    validate_pattern=_validated_pattern,
    now=utcnow,
    record_qualification_history=_record_qualification_history,
    record_roster_impact=record_roster_impact,
    user_permissions=user_permissions,
    admin_required=admin_required,
    pattern_codes=PATTERN_CODES,
)))
app.register_blueprint(create_onboarding_blueprint(OnboardingDependencies(
    db=db,
    Unit=Unit,
    QualificationType=QualificationType,
    Watch=Watch,
    Staff=Staff,
    ShiftType=ShiftType,
    UnitMembership=UnitMembership,
    SecureInvitation=SecureInvitation,
    Requirement=Requirement,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    validate_csrf=_validate_csrf,
)))
app.register_blueprint(create_reference_data_blueprint(ReferenceDataDependencies(
    db=db,
    AnnotationType=AnnotationType,
    AnnotationAudit=AnnotationAudit,
    Assignment=Assignment,
    ShiftType=ShiftType,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf,
    refresh_annotation_cache=refresh_annotation_cache,
    normalise_codes=_normalise_codes,
    save_codes_setting=_save_codes_setting,
    prune_roster_code_settings=_prune_roster_code_settings,
    working_codes=get_working_codes,
    banned_codes=get_banned_roster_codes,
    excluded_codes=get_exclude_from_counters,
    non_working_codes=get_non_working_codes,
)))
app.register_blueprint(create_overtime_blueprint(OvertimeDependencies(
    ShiftType=ShiftType,
    Staff=Staff,
    current_unit_id=_current_unit_id,
    consume_rate_limit=_consume_rate_limit,
    is_editor_user=is_editor_user,
    validate_csrf=_validate_csrf,
    parse_date=_parse_date,
    compute_candidates=_compute_overtime_candidates,
    can_send_messages=can_send_unit_messages,
    send_sms=_send_overtime_sms_notifications,
    default_sms_body=_default_overtime_sms_body,
    sms_configured=_sms_service_configured,
)))
app.register_blueprint(create_staff_lifecycle_blueprint(StaffLifecycleDependencies(
    db=db,
    Staff=Staff,
    RosterImpactEventType=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    parse_date=_parse_date,
    record_roster_impact=record_roster_impact,
    admin_required=admin_required,
)))
app.register_blueprint(create_watch_move_blueprint(WatchMoveDependencies(
    db=db,
    Staff=Staff,
    Watch=Watch,
    StaffWatchHistory=StaffWatchHistory,
    RosterImpactEventType=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    record_roster_impact=record_roster_impact,
    log_change=log_change,
)))
app.register_blueprint(create_home_blueprint(HomeDependencies(
    db=db,
    Unit=Unit,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
)))
app.register_blueprint(create_recovery_request_blueprint(RecoveryRequestDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    UnitMembership=UnitMembership,
    RecoveryRequest=RecoveryRequest,
    Unit=Unit,
    Staff=Staff,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    valid_email=_valid_email,
    normalized_login=_normalized_login,
    platform_support_emails=_platform_support_emails,
    unit_admin_emails=_unit_admin_emails,
    send_email=_send_account_email,
    now=utcnow,
    active_recovery=_active_recovery_from_digest,
    is_admin_user=is_admin_user,
    bind_authenticated_unit=bind_authenticated_unit,
    generate_password_hash=generate_password_hash,
)))
app.register_blueprint(create_unit_accounts_blueprint(UnitAccountsDependencies(
    db=db,
    Unit=Unit,
    Staff=Staff,
    PlatformIdentity=PlatformIdentity,
    UnitMembership=UnitMembership,
    SecureInvitation=SecureInvitation,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    validate_csrf=_validate_csrf,
    normalized_login=_normalized_login,
    now=utcnow,
    tenant_get=tenant_get,
)))
app.register_blueprint(create_invitation_acceptance_blueprint(
    InvitationAcceptanceDependencies(
        db=db,
        SecureInvitation=SecureInvitation,
        Unit=Unit,
        DatabaseRoutingMetadata=DatabaseRoutingMetadata,
        Staff=Staff,
        deployment_environment=DEPLOYMENT_ENV,
        consume_rate_limit=_consume_rate_limit,
        now=utcnow,
        bind_authenticated_unit=bind_authenticated_unit,
        validate_csrf=_validate_csrf,
        valid_email=_valid_email,
        run_signup=_run_invitation_signup,
        signup_error=SignupWorkflowError,
    )
))
app.register_blueprint(create_staff_profile_blueprint(StaffProfileDependencies(
    db=db,
    Staff=Staff,
    UnitMembership=UnitMembership,
    PlatformIdentity=PlatformIdentity,
    SmsSenderRegistration=SmsSenderRegistration,
    MfaCredential=MfaCredential,
    Assignment=Assignment,
    Notification=Notification,
    current_unit_id=_current_unit_id,
    is_editor_user=is_editor_user,
    validate_csrf=_validate_csrf,
    normalise_uk_mobile=_normalise_uk_mobile,
    valid_email=_valid_email,
    normalise_phone=_normalise_phone_number,
    now=utcnow,
    qr_data_uri=_totp_qr_data_uri,
    absence_types=get_absence_types,
    month_range=month_range,
    get_shift=get_shift,
    shift_duration_minutes=shift_duration_minutes,
)))
app.register_blueprint(create_password_blueprint(PasswordDependencies(
    db=db,
    Staff=Staff,
    PlatformIdentity=PlatformIdentity,
    tenant_get=tenant_get,
    validate_csrf=_validate_csrf,
    generate_password_hash=generate_password_hash,
)))
app.register_blueprint(create_kiosk_account_blueprint(KioskAccountDependencies(
    db=db,
    Unit=Unit,
    Staff=Staff,
    UnitMembership=UnitMembership,
    SecureInvitation=SecureInvitation,
    current_unit_id=_current_unit_id,
    live_position_enabled=live_position_enabled,
    tenant_get=tenant_get,
    utcnow=utcnow,
    validate_csrf=_validate_csrf,
    is_admin_user=is_admin_user,
)))
app.register_blueprint(create_operational_currency_blueprint(
    OperationalCurrencyDependencies(
        db=db,
        current_unit_id=_current_unit_id,
        is_admin_user=is_admin_user,
        live_position_enabled=live_position_enabled,
        currency_requirement=_operational_currency_requirement,
        save_currency_requirement=_save_operational_currency_requirement,
        currency_shortfalls=_operational_currency_shortfalls,
        validate_csrf=_validate_csrf,
    )
))
app.register_blueprint(create_toil_administration_blueprint(
    ToilAdministrationDependencies(
        db=db,
        Staff=Staff,
        current_unit_id=_current_unit_id,
        is_admin_user=is_admin_user,
        validate_csrf=_validate_csrf,
        record_toil_transaction=_record_toil_transaction,
    )
))
app.register_blueprint(create_admin_utility_blueprint(AdminUtilityDependencies(
    ChangeLog=ChangeLog,
    is_admin_user=is_admin_user,
)))
def _worker_health_snapshot(application_module, *, stale_after_seconds: int):
    """Load the worker dependency only when the privileged probe is requested."""
    from platform_provisioning import worker_health_snapshot

    return worker_health_snapshot(
        application_module,
        stale_after_seconds=stale_after_seconds,
    )


app.register_blueprint(create_worker_health_blueprint(WorkerHealthDependencies(
    application_module=sys.modules[__name__],
    metrics=_operational_metrics,
    worker_health_snapshot=_worker_health_snapshot,
)))
app.register_blueprint(create_platform_admin_blueprint(PlatformAdminDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    Unit=Unit,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    PlanHistory=PlanHistory,
    SuperAdminAudit=SuperAdminAudit,
    UnitMembership=UnitMembership,
    SecureInvitation=SecureInvitation,
    ProvisioningJob=ProvisioningJob,
    SignupWorkflow=SignupWorkflow,
    FeatureFlag=FeatureFlag,
    AggregateUsageEvent=AggregateUsageEvent,
    now=utcnow,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    security_event=_security_event,
    feature_flags=PLATFORM_FEATURE_FLAGS,
    module_feature_flags=PLATFORM_MODULE_FLAGS,
)))
app.register_blueprint(briefing_blueprint)

def _operational_routes_ready() -> bool:
    active_units = Unit.query.filter(
        Unit.status == "active", Unit.code != "CTRL"
    ).all()
    for unit in active_units:
        routing = db.session.get(DatabaseRoutingMetadata, unit.id)
        if not routing or not routing.secret_name or not os.environ.get(
            routing.secret_name
        ):
            return False
    return True


register_operations_routes(
    app,
    db=db,
    environment=DEPLOYMENT_ENV,
    limiter=_rate_limiter,
    metrics=_operational_metrics,
    required_tables=CONTROL_TABLES,
    additional_readiness_check=_operational_routes_ready,
)


app.cli.add_command(create_roster_cli(RosterCliDependencies(
    db=db,
    Unit=Unit,
    RosterImpactEventType=RosterImpactEventType,
    add_months=add_months,
    roster_period_service=roster_period_service,
    roster_impact_service=roster_impact_service,
)))

# -------------------- WSGI entry point --------------------
# Compatibility alias for WSGI servers that import ``application``.
application = app

# -------------------- Local dev server --------------------
if __name__ == "__main__":
    # bind explicitly & avoid debug reloader port conflicts
    app.run(host="127.0.0.1", port=5001, debug=False)
