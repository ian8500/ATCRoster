"""Application assembly and legacy compatibility implementation."""

from typing import Any, cast
from flask import redirect, url_for, flash, abort, session, g
import json as _json
import os
import sys
from functools import partial
from datetime import date, timedelta

from flask_login import (
    LoginManager, login_user, logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash
from sqlalchemy.orm import Session as OrmSession
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
    FatigueRuleConfigDependencies,
    FatigueRuleConfigService,
    compliance_month,
)
from roster_population_service import (
    create_deterministic_roster_population_service,
)
from roster_impact_service import (
    RosterImpactDependencies,
    RosterImpactEventType,
    RosterImpactService,
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
    month_add as roster_period_add,
    ShiftCounterService,
    parse_hhmm as parse_roster_hhmm, parse_iso_date as parse_roster_date,
    parse_year_month as parse_roster_year_month,
    ensure_month_requirement as ensure_roster_month_requirement,
    requirements_for_day as resolve_roster_requirements_for_day,
    PatternRuntime,
    PatternRuntimeDependencies,
    create_roster_month_service,
    ShiftLookupService,
)
from atcroster.roster.editing import (
    RosterEditingRuntime,
    create_roster_editing_dependencies,
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
from atcroster.compatibility import module_callback
from atcroster.composition import DeferredReference
from atcroster.web_assets import register_template_helpers
from atcroster.reports import ReportingRuntime, create_reporting_runtime_dependencies
from atcroster.auth import (
    AuthRuntime,
    AuthRuntimeDependencies,
    create_auth_registration_dependencies,
    register_auth_blueprints,
    load_identity,
    canonical_login_redirect,
    airport_login_endpoint,
    complete_platform_login,
)
from atcroster.qualifications import (
    QualificationRuntime,
    QualificationRuntimeDependencies,
    create_eligibility_dependencies,
    EligibilityService,
    ComplianceRuntime,
    create_compliance_runtime_dependencies,
    OperationalCurrencyRuntime,
    OperationalCurrencyRuntimeDependencies,
    create_qualification_registration_dependencies,
    register_qualification_blueprints,
    classify_qualification_impact,
    has_other_valid_ue,
    record_roster_impact_for_qualification,
    may_view_training_profile,
)
from atcroster.audit import (
    ChangeAuditService,
    context_month_for_date,
    record_central_security_event,
)
from atcroster.workforce.joiners import JoinerDependencies, create_joiner
from atcroster.fatigue import (
    FatigueRuntime,
    create_fatigue_runtime_dependencies,
    FatigueCompatibilityService,
)
from atcroster.errors import ErrorHandlerDependencies, register_error_handlers
from atcroster.extensions import (
    OPERATIONAL_TABLE_NAMES as _OPERATIONAL_TABLE_NAMES,
    create_tenant_database,
)
from atcroster.public import public_blueprint
from atcroster.notifications import (
    create_notification_registration_dependencies,
    NotificationRuntime,
    NotificationRuntimeDependencies,
    register_notification_blueprints,
    parse_sms_number_lines,
    send_via_messagemedia,
    valid_email,
    SmsConfigurationService,
    SmsAuditService,
    OvertimeSmsService,
    default_overtime_sms_body,
)
from atcroster.notifications.configuration import save_sms_settings
from atcroster.roster.publication import (
    create_publication_dependencies,
    create_publication_service,
)
from atcroster.roster.overtime import (
    OvertimeSupport,
    OvertimeCandidateService,
    create_overtime_candidate_dependencies,
    create_overtime_dependencies,
    create_overtime_blueprint,
    create_overtime_support_dependencies,
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
from atcroster.roster.reference_data import (
    bootstrap_reference_data as bootstrap_roster_reference_data,
)
from atcroster.roster.month_view import (
    MonthRosterLoadDependencies,
    load_month_roster,
)
from atcroster.roster.assignments import (
    AssignmentRuntime,
    create_assignment_refresh_dependencies,
    create_assignment_runtime_dependencies,
    AllocationRuntime,
    create_allocation_runtime_dependencies,
)
from atcroster.roster.annotations import AnnotationCatalogue
from atcroster.roster.settings import create_roster_settings_catalogue
from atcroster.models.tenant_events import register_tenant_session_events
from atcroster.models import append_only_audit_models, operational_models
from atcroster.roster.impacts import (
    RosterImpactRuntime,
    create_roster_impact_runtime_dependencies,
)
from atcroster.cli import CliDependencies, create_cli_commands
from atcroster.cli_roster import create_roster_cli, create_roster_cli_dependencies
from atcroster.modules import (
    ModuleAvailability,
    create_module_blueprint,
    create_module_dependencies,
)
from atcroster.calendar_feed import (
    create_calendar_feed_blueprint,
    create_calendar_feed_dependencies,
)
from atcroster.administration import (
    AdminDashboardDependencies,
    AdministrationDependencies,
    ToilService,
    create_admin_dashboard_blueprint,
    create_admin_action_dependencies,
    create_admin_context_dependencies,
    create_administration_blueprint,
    create_toil_administration_blueprint,
    create_toil_administration_dependencies,
    create_toil_service_dependencies,
    seed_toil_balances,
)
from atcroster.administration.onboarding import (
    create_onboarding_dependencies,
    create_onboarding_blueprint,
)
from atcroster.administration.reference import (
    create_reference_data_dependencies,
    create_reference_data_blueprint,
)
from atcroster.administration.lifecycle import (
    create_staff_lifecycle_dependencies,
    create_staff_lifecycle_blueprint,
)
from atcroster.administration.watch_moves import (
    create_watch_move_dependencies,
    create_watch_move_blueprint,
)
from atcroster.administration.absence_types import update_absence_types
from atcroster.administration.staff_edit import (
    create_staff_edit_dependencies,
    create_staff_edit_blueprint,
)
from atcroster.home import create_home_blueprint, create_home_dependencies
from atcroster.navigation import (
    create_navigation_context_dependencies,
    register_navigation_context,
)
from atcroster.requests import (
    REQUEST_STATUSES,
    REQUEST_TRANSITIONS,
    RequestWorkflowService,
    create_request_workflow_dependencies,
    clamp_request_navigation,
)
from atcroster.accounts import (
    create_account_registration_dependencies,
    register_account_blueprints,
    active_recovery_from_digest,
    platform_support_emails,
    normalise_phone_number,
    record_successful_login,
    unit_admin_emails,
)
from atcroster.accounts.signup import (
    SignupSaga,
    SignupSagaDependencies,
    SignupWorkflowError,
)
from atcroster.admin_utilities import (
    create_admin_utility_blueprint,
    create_admin_utility_dependencies,
)
from atcroster.platform import (
    LegacyBootstrapService,
    PLATFORM_FEATURE_FLAGS,
    PLATFORM_MODULE_FLAGS,
    WorkerHealthDependencies,
    create_worker_health_blueprint,
    load_worker_health_snapshot,
    operational_routes_ready,
)
from atcroster.platform.admin import (
    create_platform_admin_dependencies,
    create_platform_admin_blueprint,
)
from atcroster.security.csrf import register_csrf_protection
from atcroster.security.encryption import FieldEncryptionService
from atcroster.security.headers import (
    SecurityHeaderDependencies,
    register_security_headers,
)
from atcroster.security import (
    PrincipalBoundaryDependencies,
    create_admin_required,
    create_roster_edit_required,
    register_principal_boundaries,
)
from atcroster.security.sessions import (
    SessionLifecycle,
    SessionLifecycleDependencies,
)
from atcroster.tenancy_hooks import TenantHookDependencies, register_tenant_hooks
from atcroster.tenancy_writes import (
    current_unit_id as resolve_current_unit_id,
    discard_touched_units,
    enforce_operational_writes,
    invalidate_touched_units,
    tenant_get as tenant_scoped_get,
)
from atcroster.briefing_bootstrap import load_briefing_module
from migrations.fresh_schema import CONTROL_TABLES
from absence_requests_blueprint import (
    create_absence_request_dependencies,
    create_absence_requests_blueprint,
)
from reports_blueprint import create_reports_blueprint, create_reports_dependencies
from roster_blueprint import create_roster_blueprint, create_roster_dependencies
from training_blueprint import create_training_blueprint, create_training_dependencies
from roster_period_service import RosterPeriodDependencies, RosterPeriodService
from atcroster.planning import create_planning_dependencies, create_planning_services
from atcroster.live_position import (
    create_operational_registration_dependencies,
    register_operational_blueprints,
)
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
# Legacy callers imported the JSON module from the former monolith.
json = _json
_runtime_settings = get_runtime_settings(app)
configure_production_logging(app, _runtime_settings.deployment_environment)
_operational_metrics = MetricsRegistry()
_security_event_reference: DeferredReference[Any] = DeferredReference("security event")
_session_lifecycle_reference: DeferredReference[Any] = DeferredReference(
    "session lifecycle"
)
_work_pattern_service_reference: DeferredReference[Any] = DeferredReference(
    "work pattern service"
)
_override_classifier_reference: DeferredReference[Any] = DeferredReference(
    "override classifier"
)
# Legacy database-isolation callers import this registry from the public app
# module. Ownership remains with the tenant database extension.
OPERATIONAL_TABLE_NAMES = _OPERATIONAL_TABLE_NAMES
# Preserve the legacy public repository root for scripts and compatibility
# callers; Flask itself owns the package-local application module.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPLOYMENT_ENV = _runtime_settings.deployment_environment
FIELD_ENCRYPTION_KEY = _runtime_settings.field_encryption_key
FIELD_ENCRYPTION_KEYS = _runtime_settings.field_encryption_keys

_rate_limiter: Any
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

_asset_version, _asset_url = register_template_helpers(app)

_current_unit_id = partial(resolve_current_unit_id, current_user)

@app.before_request
def _start_request_tenant_boundary():
    clear_request_context()
    g.metrics_started_at = begin_request(_operational_metrics)

_validate_csrf, _enforce_csrf = register_csrf_protection(app)

app.register_blueprint(public_blueprint)

_error_handlers = register_error_handlers(
    app,
    ErrorHandlerDependencies(
        security_event=lambda event, **safe_fields: _security_event_reference.get()(
            event, **safe_fields
        ),
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
        enforce_session=lambda user: _session_lifecycle_reference.get().enforce_request(user),
        routing_for_unit=lambda unit_id: db.session.get(
            cast(Any, DatabaseRoutingMetadata), unit_id
        ),
        clear_context=clear_request_context,
        bind_authenticated_unit=bind_authenticated_unit,
        reset_authenticated_unit=reset_authenticated_unit,
        bind_platform_control=bind_platform_control,
        reset_platform_control=reset_platform_control,
    ),
)

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

_canonical_login_redirect = partial(canonical_login_redirect, url_for=url_for)
_airport_login_endpoint = airport_login_endpoint

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

_memoize = partial(memoize, _cache)

def _invalidate_month_cache_for_day(d: date):
    invalidate_month_for_day(
        _cache, _load_month_roster_fast, _current_unit_id(), d,
    )

def _load_operational_models():
    """Load model declarations after the canonical database extension exists."""
    from atcroster.models import operational

    return operational

_operational_models = _load_operational_models()
(
    AnnotationAudit, AnnotationType, Assignment, ChangeLog, Leave, Notification,
    RequestAudit, Requirement, RosterSetting, ShiftRequest, ShiftType, Sickness,
    SmsAudit, SmsSenderRegistration, SpecialRequirement, Staff, StaffWatchHistory,
    TrainingLevel, TrainingObjective, TrainingScore, TrainingSession, Unit, Watch,
    HandoverEquipment, HandoverField, HandoverOperationalState, HandoverRecord,
) = (
    _operational_models.AnnotationAudit, _operational_models.AnnotationType,
    _operational_models.Assignment, _operational_models.ChangeLog,
    _operational_models.Leave, _operational_models.Notification,
    _operational_models.RequestAudit, _operational_models.Requirement,
    _operational_models.RosterSetting, _operational_models.ShiftRequest,
    _operational_models.ShiftType, _operational_models.Sickness,
    _operational_models.SmsAudit, _operational_models.SmsSenderRegistration,
    _operational_models.SpecialRequirement, _operational_models.Staff,
    _operational_models.StaffWatchHistory, _operational_models.TrainingLevel,
    _operational_models.TrainingObjective, _operational_models.TrainingScore,
    _operational_models.TrainingSession, _operational_models.Unit,
    _operational_models.Watch, _operational_models.HandoverEquipment,
    _operational_models.HandoverField, _operational_models.HandoverOperationalState,
    _operational_models.HandoverRecord,
)
# Control-plane and advanced product entities live in a separate module so
# they can move to the central database without rewriting the legacy UI.
SaaS = register_saas_models(db, utcnow)
DatabaseRoutingMetadata: Any = SaaS.DatabaseRoutingMetadata
(
    PlatformIdentity, PlatformMfaCredential, UnitMembership, SecureInvitation,
    SignupWorkflow, RecoveryRequest, ProvisioningJob, WorkerHeartbeat, FeatureFlag,
    PlanHistory, AggregateUsageEvent, SuperAdminAudit, CentralSecurityAudit,
    QualificationType, PersonQualification, PersonQualificationHistory,
    RosterPublication, RosterAcknowledgement, Scenario, OperationalPosition,
    OperationalPositionTimeAllowance, OperationalPositionGroup,
    PositionCurrencyCategory, PositionParticipantRole, PositionStatusEvent,
    PositionSession, PositionSessionParticipant, ControllerKioskCredential,
    PositionSessionAudit, PositionEndorsement, PositionRequirement, BreakPlan,
    AchievedDuty, FatigueReport, ToilTransaction, WorkPattern, WorkPatternDay,
    WorkPatternDayAllowedShift, StaffPatternAssignment, StaffRule, BankHoliday,
    RosterProposal, RosterProposalAssignment, RosterRuleVersion, RosterPeriod,
    RosterImpactEvent, RosterImpactException, MfaCredential,
) = (
    SaaS.PlatformIdentity, SaaS.PlatformMfaCredential, SaaS.UnitMembership,
    SaaS.SecureInvitation, SaaS.SignupWorkflow, SaaS.RecoveryRequest,
    SaaS.ProvisioningJob, SaaS.WorkerHeartbeat, SaaS.FeatureFlag, SaaS.PlanHistory,
    SaaS.AggregateUsageEvent, SaaS.SuperAdminAudit, SaaS.CentralSecurityAudit,
    SaaS.QualificationType, SaaS.PersonQualification, SaaS.PersonQualificationHistory,
    SaaS.RosterPublication, SaaS.RosterAcknowledgement, SaaS.Scenario,
    SaaS.OperationalPosition, SaaS.OperationalPositionTimeAllowance,
    SaaS.OperationalPositionGroup, SaaS.PositionCurrencyCategory,
    SaaS.PositionParticipantRole, SaaS.PositionStatusEvent, SaaS.PositionSession,
    SaaS.PositionSessionParticipant, SaaS.ControllerKioskCredential,
    SaaS.PositionSessionAudit, SaaS.PositionEndorsement, SaaS.PositionRequirement,
    SaaS.BreakPlan, SaaS.AchievedDuty, SaaS.FatigueReport, SaaS.ToilTransaction,
    SaaS.WorkPattern, SaaS.WorkPatternDay, SaaS.WorkPatternDayAllowedShift,
    SaaS.StaffPatternAssignment, SaaS.StaffRule, SaaS.BankHoliday,
    SaaS.RosterProposal, SaaS.RosterProposalAssignment, SaaS.RosterRuleVersion,
    SaaS.RosterPeriod, SaaS.RosterImpactEvent, SaaS.RosterImpactException,
    SaaS.MfaCredential,
)

_briefing = load_briefing_module()
(
    BriefingAssuranceRun, BriefingAudit, BriefingDelivery, BriefingItem,
    BriefingMessageType, briefing_blueprint, briefing_enabled, briefing_local_now,
) = (
    _briefing.BriefingAssuranceRun, _briefing.BriefingAudit,
    _briefing.BriefingDelivery, _briefing.BriefingItem,
    _briefing.BriefingMessageType, _briefing.blueprint, _briefing.enabled,
    _briefing.local_now,
)

# Enforce the authenticated airport on all legacy operational SELECTs and
# stamp new rows. This protects older routes while they move to repositories.

TENANT_OPERATIONAL_MODELS = operational_models(_operational_models, SaaS, _briefing)

roster_month_cache = RosterMonthCache(
    float(os.environ.get("ATCROSTER_ROSTER_CACHE_SECONDS", "30"))
)

APPEND_ONLY_AUDIT_MODELS = append_only_audit_models(SaaS, _operational_models, _briefing)

register_tenant_session_events(
    OrmSession,
    operational_models=TENANT_OPERATIONAL_MODELS,
    append_only_models=APPEND_ONLY_AUDIT_MODELS,
    SmsAudit=SmsAudit,
    authenticated_unit_id=authenticated_unit_id,
    enforce_operational_writes=enforce_operational_writes,
    invalidate_touched_units=invalidate_touched_units,
    discard_touched_units=discard_touched_units,
    invalidate_unit=roster_month_cache.invalidate_unit,
)

_enforce_principal_boundaries = register_principal_boundaries(
    app,
    PrincipalBoundaryDependencies(
        UnitMembership=UnitMembership,
        MfaCredential=MfaCredential,
        deployment_environment=DEPLOYMENT_ENV,
        logout_user=logout_user,
        redirect=redirect,
        url_for=url_for,
        abort=abort,
    ),
)

# -------------------- Reference data helpers --------------------

roster_settings_catalogue = create_roster_settings_catalogue(
    db=db,
    operational_models=_operational_models,
    current_unit_id=_current_unit_id,
    defaults=DEFAULT_ROSTER_SETTINGS,
    absence_defaults=DEFAULT_ABSENCE_TYPES,
    working_codes=DEFAULT_WORKING_CODES,
    banned_codes=DEFAULT_BANNED_ROSTER_CODES,
    excluded_codes=DEFAULT_EXCLUDE_FROM_COUNTERS,
    non_working_codes=DEFAULT_NON_WORKING_CODES,
)
_normalise_codes = roster_settings_catalogue.normalise
_roster_settings_snapshot = roster_settings_catalogue.snapshot
refresh_roster_settings_cache = roster_settings_catalogue.refresh
_load_codes_setting = roster_settings_catalogue.load_codes
get_working_codes = roster_settings_catalogue.get_working_codes
get_absence_types = roster_settings_catalogue.get_absence_types
_save_absence_types = roster_settings_catalogue.save_absence_types
get_banned_roster_codes = roster_settings_catalogue.get_banned_codes
get_exclude_from_counters = roster_settings_catalogue.get_excluded_counter_codes
get_non_working_codes = roster_settings_catalogue.get_non_working_codes

eligibility_service = EligibilityService(create_eligibility_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    authenticated_unit_id=authenticated_unit_id,
))
staff_is_countable_on = eligibility_service.is_countable
operational_capability_service = eligibility_service.capability_service
get_staff_operational_capability = eligibility_service.operational_capability
get_operational_capability_matrix = eligibility_service.capability_matrix

get_shift_counter_map = roster_settings_catalogue.get_shift_counter_map

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
    send=send_via_messagemedia,
)
notification_runtime = NotificationRuntime(NotificationRuntimeDependencies(
    db=db,
    app_logger=app.logger,
    support_emails=lambda: platform_support_emails(
        PlatformIdentity, os.getenv("ATCROSTER_SUPPORT_EMAIL", ""), valid_email,
    ),
    admin_emails=lambda unit_id: unit_admin_emails(
        db, PlatformIdentity, UnitMembership, unit_id,
    ),
    sms_configuration=sms_configuration,
    sms_audit=sms_audit_service,
    overtime_sms=overtime_sms_service,
    flash=flash,
    sms_sender=module_callback(__name__, "_send_sms_via_messagemedia"),
))
_messagemedia_credentials = notification_runtime.credentials
_normalise_sms_number = notification_runtime.normalise_number
_normalise_uk_mobile = notification_runtime.normalise_uk_mobile
_sms_number_options = notification_runtime.number_options
_sms_sender_options = notification_runtime.sender_options
_sms_operational_options = notification_runtime.operational_options
_sms_default_number = notification_runtime.default_number
_sms_service_configured = notification_runtime.sms_configured
_email_service_configured = notification_runtime.email_configured
_send_account_email = notification_runtime.send_email
_valid_email = notification_runtime.valid_email
_platform_support_emails = notification_runtime.support_emails
_unit_admin_emails = notification_runtime.admin_emails
_send_sms_via_messagemedia = notification_runtime.send_sms
_send_sms = notification_runtime.send_sms
_record_sms_audit = notification_runtime.record_sms
_send_overtime_sms_notifications = notification_runtime.send_overtime
_default_overtime_sms_body = default_overtime_sms_body
_flash_sms_result = notification_runtime.flash_result

annotation_catalogue = AnnotationCatalogue(AnnotationType, _current_unit_id)
_annotation_snapshot = annotation_catalogue.snapshot
refresh_annotation_cache = annotation_catalogue.refresh
get_annotation_types = annotation_catalogue.types
get_annotation_config = annotation_catalogue.config
get_annotation_groups = annotation_catalogue.groups
annotation_tags_for = annotation_catalogue.tags_for
annotation_codes_for_tag = annotation_catalogue.codes_for_tag

_parse_codes_input = roster_settings_catalogue.parse_codes_input
_save_codes_setting = roster_settings_catalogue.save_codes_setting
_prune_roster_code_settings = roster_settings_catalogue.prune_code_settings
_save_roster_setting = roster_settings_catalogue.save_setting

module_availability = ModuleAvailability(FeatureFlag)
training_enabled = module_availability.training
competency_enabled = module_availability.competency
live_position_enabled = module_availability.live_position

operational_currency_runtime = OperationalCurrencyRuntime(
    OperationalCurrencyRuntimeDependencies(
        db=db,
        Staff=Staff,
        PositionEndorsement=PositionEndorsement,
        PositionSession=PositionSession,
        PositionParticipantRole=PositionParticipantRole,
        PositionSessionParticipant=PositionSessionParticipant,
        current_unit_id=_current_unit_id,
        settings_snapshot=_roster_settings_snapshot,
        save_setting=_save_roster_setting,
        live_position_enabled=live_position_enabled,
        now=utcnow,
        setting_key=OPERATIONAL_CURRENCY_SETTING_KEY,
        defaults=DEFAULT_OPERATIONAL_CURRENCY_REQUIREMENT,
    )
)
_operational_currency_requirement = operational_currency_runtime.requirement
_save_operational_currency_requirement = operational_currency_runtime.save_requirement
_operational_currency_window = operational_currency_runtime.window
_minutes_between = operational_currency_runtime.minutes_between
_operational_currency_shortfalls = operational_currency_runtime.shortfalls

_parse_sms_number_lines = parse_sms_number_lines
bootstrap_reference_data = partial(
    bootstrap_roster_reference_data,
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

# -------------------- Login --------------------

@login_manager.user_loader
def load_user(user_id):
    return load_identity(
        user_id,
        db=db,
        UnitMembership=UnitMembership,
        DatabaseRoutingMetadata=DatabaseRoutingMetadata,
        PlatformIdentity=PlatformIdentity,
        Staff=Staff,
        deployment_environment=DEPLOYMENT_ENV,
        bind_authenticated_unit=bind_authenticated_unit,
        remember_tenant_token=lambda token: setattr(
            g, "tenant_context_token", token
        ),
    )

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

is_admin_user = is_admin
is_editor_user = is_editor
user_permissions = permissions_for
has_unit_permission = has_permission
is_under_training = is_trainee
can_record_training = may_record_training
can_manage_training = may_manage_training
can_edit_roster = may_edit_roster
can_apply_annotations = may_apply_annotations
can_send_unit_messages = may_send_unit_messages
can_override_roster_conflicts = may_override_roster_conflicts

tenant_get = partial(tenant_scoped_get, current_unit_id=_current_unit_id)
roster_edit_required = create_roster_edit_required(current_user, can_edit_roster)

shift_lookup_service = ShiftLookupService(
    ShiftType=ShiftType,
    current_unit_id=_current_unit_id,
    banned_codes=get_banned_roster_codes,
)
_shift_by_code = shift_lookup_service.by_code
refresh_shift_cache = shift_lookup_service.refresh
get_shift = shift_lookup_service.get
_shift_groups_snapshot = shift_lookup_service.groups

roster_settings_catalogue.set_secondary_cache_clear(
    _shift_groups_snapshot.cache_clear
)

PATTERN_CODES = ("M", "A", "D", "N", "OPS", "OFF")
DEFAULT_BASE_PATTERN = "M,M,A,A,N,N,OFF,OFF,OFF,OFF"

pattern_runtime = PatternRuntime(PatternRuntimeDependencies(
    db=db,
    Staff=Staff,
    StaffWatchHistory=StaffWatchHistory,
    authenticated_unit_id=authenticated_unit_id,
    settings_snapshot=_roster_settings_snapshot,
    expand_pattern=expand_pattern,
    validated_pattern=validated_pattern,
    default_pattern=DEFAULT_BASE_PATTERN,
))
watch_id_for_staff_on = pattern_runtime.watch_id
watch_ids_for_staff_on = pattern_runtime.watch_ids
_watch_id_for_staff_on = pattern_runtime.cached_watch_id
_expand_pattern = pattern_runtime.expand
_validated_pattern = pattern_runtime.validate
_effective_watch = pattern_runtime.effective_watch
_unit_pattern_context = pattern_runtime.unit_context
_pattern_context = pattern_runtime.context
pattern_for = pattern_runtime.pattern_for
_night_active_on = pattern_runtime.night_active
day_leave_for = pattern_runtime.leave_for
code_from_pattern = pattern_runtime.code_for
_effective_watch_id = pattern_runtime.effective_watch_id

shift_counter_service = ShiftCounterService(
    current_unit_id=_current_unit_id,
    counter_map=get_shift_counter_map,
    get_shift=get_shift,
    night_active_on=_night_active_on,
)
shift_counter_group = shift_counter_service.group
shift_counter_group_for_day = shift_counter_service.group_for_day

def deterministic_roster_population_service():
    """Build the shared baseline-population service for application callers."""
    return create_deterministic_roster_population_service(
        db=db,
        operational_models=_operational_models,
        saas_models=SaaS,
        utcnow=utcnow,
        legacy_code_resolver=code_from_pattern,
        watch_id_resolver=_effective_watch_id,
    )

roster_impact_runtime = RosterImpactRuntime(create_roster_impact_runtime_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    roster_impact_event_type=RosterImpactEventType,
    cache=_cache,
    cached_loader=_load_month_roster_fast,
    add_months=add_months,
    current_unit_id=_current_unit_id,
    current_user=lambda: current_user,
    population_service=deterministic_roster_population_service,
    override_classifier=_override_classifier_reference.get,
    service_factory=RosterImpactService,
    service_dependencies=RosterImpactDependencies,
    classify_qualification_impact=classify_qualification_impact,
    has_other_valid_ue=has_other_valid_ue,
    record_qualification_impact=record_roster_impact_for_qualification,
    now=utcnow,
))
_generated_roster_horizon_end = roster_impact_runtime.generated_horizon_end
_invalidate_roster_impact_coverage = roster_impact_runtime.invalidate_coverage
roster_impact_service = roster_impact_runtime.service
record_roster_impact = roster_impact_runtime.record
_qualification_impact_type = roster_impact_runtime.qualification_impact_type
_person_has_other_valid_ue = roster_impact_runtime.person_has_other_valid_ue
record_qualification_roster_impact = roster_impact_runtime.record_qualification

_cycle_day_for = pattern_runtime.cycle_day

assignment_runtime = AssignmentRuntime(create_assignment_runtime_dependencies(
    operational_models=_operational_models,
    refresh=create_assignment_refresh_dependencies(
        db=db,
        operational_models=_operational_models,
        code_from_pattern=code_from_pattern,
        day_leave_for=day_leave_for,
        get_shift=get_shift,
        absence_types=get_absence_types,
    ),
    month_range=month_days,
    shift_minutes=shift_minutes,
    daily_requirements=daily_requirements,
    ensure_month_requirement=ensure_roster_month_requirement,
    requirements_for_day=resolve_roster_requirements_for_day,
    iter_year_months=iter_year_months,
))
set_assignment = assignment_runtime.set_assignment
overwrite_assignment = assignment_runtime.overwrite_assignment
refresh_day_from_pattern_and_leave = assignment_runtime.refresh_day
shift_duration_minutes = assignment_runtime.shift_duration_minutes
ensure_month_requirement = assignment_runtime.ensure_month_requirement
requirements_for_day = assignment_runtime.requirements_for_day
generate_month = assignment_runtime.generate_month
generate_range = assignment_runtime.generate_range
ensure_assignments_for_range = assignment_runtime.generate_range

roster_month_service = create_roster_month_service(
    db=db,
    operational_models=_operational_models,
    add_months=partial(roster_period_add, add_months=add_months),
    days_for_month=month_days,
    parse_year_month=partial(parse_roster_year_month, parser=parse_year_month),
    ensure_month_requirement=ensure_month_requirement,
)
month_has_data = roster_month_service.has_data
_lock_roster_month = roster_month_service.lock
month_range = roster_month_service.range
parse_ym = roster_month_service.parse

_fatigue_rule_config_service = FatigueRuleConfigService(
    FatigueRuleConfigDependencies(
        db=db,
        RosterSetting=RosterSetting,
        current_unit_id=_current_unit_id,
    )
)
_fatigue_rule_config = _fatigue_rule_config_service.load
_save_fatigue_rule_config = _fatigue_rule_config_service.save

fatigue_runtime = FatigueRuntime(create_fatigue_runtime_dependencies(
    operational_models=_operational_models,
    get_shift=get_shift,
    is_working=_is_working,
    span=_span,
    is_night_duty=_is_night_duty,
    is_early_start=_is_early_start,
    is_morning_duty=_is_morning_duty,
    analyze_segments=_analyze_segments,
    custom_fatigue_flags=_custom_fatigue_flags,
    fatigue_rule_config=_fatigue_rule_config,
))
_segments_from_assignments = fatigue_runtime.segments_from_assignments
_configured_fatigue_findings = fatigue_runtime.configured_findings
_segments_for_staff = fatigue_runtime.segments_for_staff
fatigue_flags_for_range = fatigue_runtime.findings_for_range
roster_fatigue_flags_matrix = fatigue_runtime.findings_matrix

fatigue_compatibility_service = FatigueCompatibilityService(
    range_findings=module_callback(__name__, "fatigue_flags_for_range"),
    get_shift=module_callback(__name__, "get_shift"),
    is_working=module_callback(__name__, "_is_working"),
    segments_for_staff=module_callback(__name__, "_segments_for_staff"),
    fatigue_rule_config=module_callback(__name__, "_fatigue_rule_config"),
    configured_findings=module_callback(__name__, "_configured_fatigue_findings"),
    span=module_callback(__name__, "_span"),
    is_early_start=module_callback(__name__, "_is_early_start"),
    is_night_duty=module_callback(__name__, "_is_night_duty"),
    is_morning_duty=module_callback(__name__, "_is_morning_duty"),
)
roster_fatigue_flags_for_range = fatigue_compatibility_service.roster_findings
would_trigger_fatigue_with_plan = fatigue_compatibility_service.proposed_findings
would_trigger_fatigue = fatigue_compatibility_service.would_trigger

would_create_new_fatigue_issues = fatigue_runtime.new_findings

_compliance_month = compliance_month
compliance_runtime = ComplianceRuntime(create_compliance_runtime_dependencies(
        operational_models=_operational_models,
        month_range=month_range,
        fatigue_rule_config=_fatigue_rule_config,
        fatigue_flags_for_range=fatigue_flags_for_range,
))
_compliance_findings = compliance_runtime.findings

# -------------------- Migrations / seeding --------------------

_legacy_bootstrap = LegacyBootstrapService(
    db=db,
    app=app,
    Unit=Unit,
    Staff=Staff,
    Watch=Watch,
    ShiftType=ShiftType,
)
migrate_tenant_foundation_compat = _legacy_bootstrap.migrate_tenant_foundation
migrate_add_role_and_calendar_token = _legacy_bootstrap.add_role_and_calendar_token
migrate_add_assignment_annotation = _legacy_bootstrap.add_assignment_annotation
migrate_add_unique_assignment_key = _legacy_bootstrap.add_unique_assignment_key
migrate_add_perf_indexes = _legacy_bootstrap.add_performance_indexes
migrate_add_requirement_req_d = _legacy_bootstrap.add_requirement_day_column
migrate_add_ut_flags = _legacy_bootstrap.add_undertraining_flags
migrate_add_is_training = _legacy_bootstrap.add_training_shift_flag
migrate_add_wm_dwm_exclude = _legacy_bootstrap.add_workforce_flags
migrate_add_phone_number = _legacy_bootstrap.add_phone_number
migrate_add_watch_pattern_configuration = (
    _legacy_bootstrap.add_watch_pattern_configuration
)
migrate_add_invitation_target = _legacy_bootstrap.add_invitation_target
migrate_add_toil_half_days_and_convert = _legacy_bootstrap.add_toil_and_leave_fields
ensure_shift = _legacy_bootstrap.ensure_shift
ensure_watch = _legacy_bootstrap.ensure_watch
seed_once = _legacy_bootstrap.seed_once

_parse_hhmm = parse_roster_hhmm
_parse_date = parse_roster_date
_normalise_phone_number = normalise_phone_number
parse_annotation = annotation_catalogue.parse
_context_month_for_date = context_month_for_date
change_audit_service = ChangeAuditService(
    db=db,
    ChangeLog=ChangeLog,
    current_user=lambda: current_user,
    now=utcnow,
)
log_change = change_audit_service.record

# --- Month math (no dateutil) ---

_month_add = partial(roster_period_add, add_months=add_months)
lock_date_for_month = partial(roster_period_lock_date, roster_lock_date=roster_lock_date)
is_month_locked = partial(
    roster_period_is_locked, roster_month_is_locked=roster_month_is_locked
)

roster_editing_runtime = RosterEditingRuntime(create_roster_editing_dependencies(
    db=db,
    operational_models=_operational_models,
    invalidate_month_for_day=_invalidate_month_cache_for_day,
    log_change=log_change,
    would_trigger_fatigue=would_trigger_fatigue,
    non_working_codes=get_non_working_codes,
    get_shift=get_shift,
))
_assignment = roster_editing_runtime.assignment
_cell_is_protected = roster_editing_runtime.cell_is_protected
_set_code = roster_editing_runtime.set_code
_has_leave_or_sick = roster_editing_runtime.has_leave_or_sickness
_fatigue_ok = roster_editing_runtime.fatigue_ok
_passes_fatigue_for = roster_editing_runtime.fatigue_ok
_weekday_is_sun = roster_editing_runtime.weekday_is_sunday
_normalize_code = roster_editing_runtime.normalize_code
_is_non_working = roster_editing_runtime.code_is_non_working
_is_working_code_prefix = roster_editing_runtime.working_code_prefix
_is_working_day_code = roster_editing_runtime.working_day_code
_is_working_m_code = roster_editing_runtime.working_morning_code
_is_working_n_code = roster_editing_runtime.working_night_code

allocation_runtime = AllocationRuntime(create_allocation_runtime_dependencies(
    db=db,
    operational_models=_operational_models,
    is_working_day_code=_is_working_day_code,
    has_leave_or_sickness=_has_leave_or_sick,
    passes_fatigue=_passes_fatigue_for,
    set_code=_set_code,
))
_is_empty_like = allocation_runtime.is_empty_like
_allocate_days_for_date = allocation_runtime.allocate

admin_required = create_admin_required(is_admin_user)

_clamp_prev_next = partial(clamp_request_navigation, minimum_month=MIN_MONTH)

inject_perms = register_navigation_context(
    app,
    create_navigation_context_dependencies(
        db=db,
        operational_models=_operational_models,
        saas_models=SaaS,
        briefing_models=_briefing,
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
    return create_admin_action_dependencies(
        db=db,
        operational_models=_operational_models,
        saas_models=SaaS,
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

# -------------------- Leave / Sickness / TOIL --------------------

_group_sickness_instances = group_sickness_instances

# -------------------- Staff profile --------------------

_training_profile_allowed = partial(
    may_view_training_profile,
    actor=current_user,
    is_editor=is_editor_user,
    can_manage_training=can_manage_training,
    can_record_training=can_record_training,
)

# -------------------- Metrics + CSV (date range; FYTD default) --------------------
# (… unchanged metrics functions from your file …)

reporting_runtime = ReportingRuntime(create_reporting_runtime_dependencies(
    operational_models=_operational_models,
    saas_models=SaaS,
    FairnessAssignment=FairnessAssignment,
    FairnessStaff=FairnessStaff,
    current_unit_id=_current_unit_id,
    annotation_snapshot=_annotation_snapshot,
    parse_annotation=parse_annotation,
    work_pattern_service=_work_pattern_service_reference.get,
    code_from_pattern=code_from_pattern,
    shift_duration_minutes=shift_duration_minutes,
    calculate_fairness=calculate_fairness,
    month_range=month_range,
    get_absence_types=get_absence_types,
))
_compute_metrics_range = reporting_runtime.compute_metrics
_compute_fairness_range = reporting_runtime.compute_fairness
_fy_start_for = reporting_runtime.financial_year_start

overtime_support = OvertimeSupport(create_overtime_support_dependencies(
    operational_models=_operational_models,
    parse_annotation=parse_annotation,
    annotation_tags_for=annotation_tags_for,
    working_codes=get_working_codes,
    span=_span,
    get_shift=get_shift,
))
_count_aava_soal_since_prev_april = overtime_support.count_aava_soal
_worked_like_consecutive_days = overtime_support.worked_like_consecutive_days
_had_sc_within_48h = overtime_support.had_sickness_within_48_hours
_has_in_date_ue = overtime_support.has_in_date_endorsement

# -------------------- Overtime finder (admin/editor) --------------------
# (… unchanged from your file …)

_count_ot_since_prev_april = overtime_support.count_overtime

# … keep the rest of your overtime helpers exactly as pasted …

_leave_summary_for_month = reporting_runtime.leave_summary

# ===== Leave-Year report (per-person config; AL only; includes TOIL days) =====
# (unchanged from your post)

_current_leave_year_window = reporting_runtime.current_leave_year_window

toil_service = ToilService(create_toil_service_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    current_unit_id=_current_unit_id,
    parse_annotation=parse_annotation,
    annotation_config=get_annotation_config,
    now=utcnow,
))
_toil_accrual_half_days_from_annotation = toil_service.accrual_half_days
_record_toil_transaction = toil_service.record_transaction
_apply_toil_annotation_delta = toil_service.apply_annotation_delta
_toil_accrued_used_in_range_half_days = toil_service.accrued_and_used

# ===== Sickness Report (unchanged) =====

_group_consecutive_days = reporting_runtime.group_consecutive_days

# -------------------- Request Sheets (shift requests) --------------------

request_workflow_service = RequestWorkflowService(create_request_workflow_dependencies(
    db=db,
    operational_models=_operational_models,
    current_unit_id=_current_unit_id,
    normalise_rules=normalise_request_rules,
    lock_date=request_lock_date,
    month_is_locked=request_month_is_locked,
    add_months=add_request_months,
    date_bounds=request_date_bounds,
    safe_admin_month=safe_admin_month,
))
_lock_date_for_target_month = request_workflow_service.lock_date_for_month
_request_date_bounds = request_workflow_service.request_date_bounds

staff_has_qualification = eligibility_service.has_qualification
_staff_has_shift_qualification = eligibility_service.has_shift_qualification

_overtime_candidate_service = OvertimeCandidateService(
    create_overtime_candidate_dependencies(
        operational_models=_operational_models,
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
_compute_overtime_candidates = _overtime_candidate_service.compute

# >>> Admin can respond to a specific request

signup_saga = SignupSaga(SignupSagaDependencies(
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
))
_normalized_login = signup_saga.normalized_login
_run_invitation_signup = signup_saga.run

qualification_runtime = QualificationRuntime(QualificationRuntimeDependencies(
    db=db,
    PersonQualificationHistory=PersonQualificationHistory,
    PositionEndorsement=PositionEndorsement,
    Assignment=Assignment,
    OperationalPosition=OperationalPosition,
    PositionRequirement=PositionRequirement,
    current_user=lambda: current_user,
    month_range=month_range,
))
_qualification_snapshot = qualification_runtime.snapshot
_record_qualification_history = qualification_runtime.record_history
_sync_qualification_to_roster_profile = qualification_runtime.sync_roster_profile
_valid_endorsement = qualification_runtime.valid_endorsement
_position_assurance = qualification_runtime.position_assurance

LOGIN_RATE_WINDOW = timedelta(minutes=15)
LOGIN_RATE_LIMIT = 10

_auth_runtime = AuthRuntime(AuthRuntimeDependencies(
    app=app,
    db=db,
    limiter=_rate_limiter,
    metrics=_operational_metrics,
    privacy_key=privacy_key,
    limiter_unavailable=LimiterUnavailable,
    structured_event=structured_event,
    PlatformIdentity=PlatformIdentity,
    PlatformMfaCredential=PlatformMfaCredential,
    MfaCredential=MfaCredential,
    RecoveryRequest=RecoveryRequest,
    decrypt_field=_decrypt_field,
    now=utcnow,
    active_recovery_from_digest=active_recovery_from_digest,
))
_login_rate_key = _auth_runtime.login_rate_key
_consume_rate_limit = _auth_runtime.consume_rate_limit
_reset_rate_limit = _auth_runtime.reset_rate_limit
_security_event = _auth_runtime.security_event
_credential_for_auth_stamp = _auth_runtime.credential_for_auth_stamp
_security_event_reference.set(_security_event)

_session_lifecycle = SessionLifecycle(
    SessionLifecycleDependencies(
        now=utcnow,
        credential_for_user=_credential_for_auth_stamp,
        security_event=lambda event, **facts: _security_event(event, **facts),
    )
)
_current_auth_stamp = _session_lifecycle.auth_stamp
_initialize_authenticated_session = _session_lifecycle.initialize
_session_lifecycle_reference.set(_session_lifecycle)

_central_security_event = partial(record_central_security_event, db, CentralSecurityAudit)
def _record_successful_login(user: Any) -> None:
    """Adapt the auth route callback to the account-domain keyword API."""
    record_successful_login(
        db=db,
        PlatformIdentity=PlatformIdentity,
        Unit=Unit,
        AggregateUsageEvent=AggregateUsageEvent,
        user=user,
        now=utcnow,
    )

_active_recovery_from_digest = _auth_runtime.active_recovery
_decrypt_mfa_secret = _auth_runtime.decrypt_mfa_secret
_matching_totp_step = _auth_runtime.matching_totp_step
_pending_platform_login = _auth_runtime.pending_platform_login

_complete_platform_login = partial(
    complete_platform_login,
    session=session,
    db=db,
    login_user=login_user,
    initialize_session=_initialize_authenticated_session,
    now=utcnow,
    security_event=_central_security_event,
    login_redirect=_canonical_login_redirect,
    redirect=redirect,
)

_totp_qr_data_uri = _auth_runtime.totp_qr_data_uri

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

planning_services = create_planning_services(app, create_planning_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    shift_group=lambda shift: shift_counter_group(shift.code, shift.unit_id),
    requirements_for_day=requirements_for_day,
    shift_group_for_day=shift_counter_group_for_day,
    shift_minutes=shift_duration_minutes,
    staff_is_countable_on=staff_is_countable_on,
    staff_has_qualification=_staff_has_shift_qualification,
    would_trigger_fatigue=would_trigger_fatigue_with_plan,
    compute_fairness_range=_compute_fairness_range,
    now=utcnow,
    pattern_context=_pattern_context,
    is_admin_user=is_admin_user,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf,
    record_roster_impact=record_roster_impact,
))
work_pattern_service = planning_services.patterns
work_pattern_admin_service = planning_services.admin
roster_validation_service = planning_services.validation
roster_proposal_service = planning_services.proposals
override_classification_service = planning_services.override_classification
_work_pattern_service_reference.set(work_pattern_service)
_override_classifier_reference.set(override_classification_service)
get_pattern_day_for_staff = work_pattern_service.get_pattern_day_for_staff
get_effective_staff_rules = work_pattern_service.get_effective_staff_rules
is_staff_eligible_for_shift = work_pattern_service.is_staff_eligible_for_shift
calculate_soft_rule_penalty = work_pattern_service.calculate_soft_rule_penalty
roster_period_service = RosterPeriodService(RosterPeriodDependencies(
    db=db, RosterPeriod=RosterPeriod, utcnow=utcnow,
))

register_operational_blueprints(app, create_operational_registration_dependencies(
    db=db, operational_models=_operational_models, saas_models=SaaS,
    now=utcnow, is_admin_user=is_admin_user, is_editor_user=is_editor_user,
    can_edit_roster=can_edit_roster,
    live_position_enabled=live_position_enabled,
    competency_enabled=competency_enabled,
    authenticated_database_route_optional=authenticated_database_route_optional,
    authenticated_unit_context=authenticated_unit_context,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf, requirements_for_day=requirements_for_day,
    shift_group_for_day=shift_counter_group_for_day,
    compliance_month=_compliance_month, log_change=log_change,
    month_add=_month_add, position_assurance=_position_assurance,
    parse_year_month=parse_ym, month_range=month_range,
    staff_has_shift_qualification=_staff_has_shift_qualification,
))

register_auth_blueprints(app, create_auth_registration_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    deployment_environment=DEPLOYMENT_ENV,
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
    decrypt_secret=_decrypt_mfa_secret,
    matching_totp_step=_matching_totp_step,
    encrypt_field=_encrypt_field,
    now=utcnow,
    current_unit_id=_current_unit_id,
    current_auth_stamp=_current_auth_stamp,
    totp_qr_data_uri=_totp_qr_data_uri,
))
register_qualification_blueprints(app, create_qualification_registration_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    roster_impact_event_type=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    is_editor_user=is_editor_user,
    is_admin_user=is_admin_user,
    now=utcnow,
    qualification_impact_type=_qualification_impact_type,
    person_has_other_valid_ue=_person_has_other_valid_ue,
    record_roster_impact=record_roster_impact,
    validate_csrf=_validate_csrf,
    load_rule_config=_fatigue_rule_config,
    save_rule_config=_save_fatigue_rule_config,
    live_position_enabled=live_position_enabled,
    currency_requirement=_operational_currency_requirement,
    save_currency_requirement=_save_operational_currency_requirement,
    currency_shortfalls=_operational_currency_shortfalls,
))
app.register_blueprint(create_reports_blueprint(create_reports_dependencies(
    operational_models=_operational_models,
    is_admin_user=is_admin_user,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    reporting_runtime=reporting_runtime,
    parse_year_month=parse_ym,
    assignment_runtime=assignment_runtime,
    toil_service=toil_service,
    get_absence_types=get_absence_types,
    live_position_enabled=live_position_enabled,
    competency_enabled=competency_enabled,
    operational_currency_runtime=operational_currency_runtime,
)))
publication_service = create_publication_service(create_publication_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
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
    send_account_email=module_callback(__name__, "_send_account_email"),
))
_roster_snapshot = publication_service.snapshot
_active_roster_publication = publication_service.active_publication
app.register_blueprint(create_roster_blueprint(create_roster_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    publication_service=publication_service,
    validate_csrf=_validate_csrf,
    parse_year_month=parse_ym,
    current_unit_id=_current_unit_id,
    roster_month_service=roster_month_service,
    assignment_runtime=assignment_runtime,
    utcnow=utcnow,
    log_change=log_change,
    consume_rate_limit=_consume_rate_limit,
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
    roster_proposal_service=roster_proposal_service,
    get_annotation_groups=get_annotation_groups,
)))
app.register_blueprint(create_absence_requests_blueprint(
    create_absence_request_dependencies(
        db=db,
        operational_models=_operational_models,
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
        workflow=request_workflow_service,
        utcnow=utcnow,
        request_statuses=REQUEST_STATUSES,
        request_transitions=REQUEST_TRANSITIONS,
        would_create_new_fatigue_issues=would_create_new_fatigue_issues,
        staff_has_shift_qualification=_staff_has_shift_qualification,
        can_override_roster_conflicts=can_override_roster_conflicts,
        lock_roster_month=_lock_roster_month,
        record_toil_transaction=_record_toil_transaction,
    )
))
app.register_blueprint(create_training_blueprint(create_training_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    current_unit_id=_current_unit_id,
    training_enabled=training_enabled,
    is_editor_user=is_editor_user,
    can_manage_training=can_manage_training,
    can_record_training=can_record_training,
    is_under_training=is_under_training,
    training_profile_allowed=_training_profile_allowed,
    validate_csrf=_validate_csrf,
    competency_enabled=competency_enabled,
    is_admin_user=is_admin_user,
    utcnow=utcnow,
    record_qualification_history=_record_qualification_history,
    sync_qualification_to_roster_profile=_sync_qualification_to_roster_profile,
    record_qualification_roster_impact=record_qualification_roster_impact,
)))
register_notification_blueprints(app, create_notification_registration_dependencies(
    db=db,
    operational_models=_operational_models,
    current_unit_id=_current_unit_id,
    now=utcnow,
    validate_csrf=_validate_csrf,
    is_admin_user=is_admin_user,
    can_send_unit_messages=can_send_unit_messages,
    notifications=notification_runtime,
))
app.register_blueprint(create_module_blueprint(create_module_dependencies(
    saas_models=SaaS,
    briefing_enabled=briefing_enabled,
    training_enabled=training_enabled,
    competency_enabled=competency_enabled,
    is_admin_user=is_admin_user,
)))
app.register_blueprint(create_calendar_feed_blueprint(create_calendar_feed_dependencies(
    get_shift=get_shift,
    db=db,
    operational_models=_operational_models,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    validate_csrf=_validate_csrf,
)))
app.register_blueprint(create_administration_blueprint(AdministrationDependencies(
    is_admin_user=is_admin_user,
    live_position_enabled=live_position_enabled,
)))
app.register_blueprint(create_admin_dashboard_blueprint(AdminDashboardDependencies(
    is_admin_user=is_admin_user,
    actions=_admin_action_dependencies(),
    context=create_admin_context_dependencies(
        db=db,
        operational_models=_operational_models,
        saas_models=SaaS,
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
    ),
)))
app.register_blueprint(create_staff_edit_blueprint(create_staff_edit_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    roster_impact_event_type=RosterImpactEventType,
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
app.register_blueprint(create_onboarding_blueprint(create_onboarding_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    validate_csrf=_validate_csrf,
)))
app.register_blueprint(create_reference_data_blueprint(create_reference_data_dependencies(
    db=db,
    operational_models=_operational_models,
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
    admin_required=admin_required,
)))
app.register_blueprint(create_overtime_blueprint(create_overtime_dependencies(
    operational_models=_operational_models,
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
app.register_blueprint(create_staff_lifecycle_blueprint(create_staff_lifecycle_dependencies(
    db=db,
    operational_models=_operational_models,
    roster_impact_event_type=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    parse_date=_parse_date,
    record_roster_impact=record_roster_impact,
    admin_required=admin_required,
)))
app.register_blueprint(create_watch_move_blueprint(create_watch_move_dependencies(
    db=db,
    operational_models=_operational_models,
    roster_impact_event_type=RosterImpactEventType,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    record_roster_impact=record_roster_impact,
    log_change=log_change,
)))
app.register_blueprint(create_home_blueprint(create_home_dependencies(
    db=db,
    operational_models=_operational_models,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
)))
register_account_blueprints(app, create_account_registration_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    deployment_environment=DEPLOYMENT_ENV,
    current_unit_id=_current_unit_id,
    is_admin_user=is_admin_user,
    is_editor_user=is_editor_user,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    valid_email=_valid_email,
    normalized_login=_normalized_login,
    platform_support_emails=_platform_support_emails,
    unit_admin_emails=_unit_admin_emails,
    send_email=_send_account_email,
    now=utcnow,
    active_recovery=_active_recovery_from_digest,
    bind_authenticated_unit=bind_authenticated_unit,
    generate_password_hash=generate_password_hash,
    tenant_get=tenant_get,
    run_signup=_run_invitation_signup,
    signup_error=SignupWorkflowError,
    normalise_uk_mobile=_normalise_uk_mobile,
    normalise_phone=_normalise_phone_number,
    qr_data_uri=_totp_qr_data_uri,
    absence_types=get_absence_types,
    month_range=month_range,
    get_shift=get_shift,
    shift_duration_minutes=shift_duration_minutes,
    live_position_enabled=live_position_enabled,
))
app.register_blueprint(create_toil_administration_blueprint(
    create_toil_administration_dependencies(
        db=db,
        operational_models=_operational_models,
        current_unit_id=_current_unit_id,
        is_admin_user=is_admin_user,
        validate_csrf=_validate_csrf,
        record_toil_transaction=_record_toil_transaction,
    )
))
app.register_blueprint(create_admin_utility_blueprint(create_admin_utility_dependencies(
    operational_models=_operational_models,
    is_admin_user=is_admin_user,
)))
app.register_blueprint(create_worker_health_blueprint(WorkerHealthDependencies(
    application_module=sys.modules[__name__],
    metrics=_operational_metrics,
    worker_health_snapshot=load_worker_health_snapshot,
)))
app.register_blueprint(create_platform_admin_blueprint(create_platform_admin_dependencies(
    db=db,
    operational_models=_operational_models,
    saas_models=SaaS,
    now=utcnow,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    security_event=_security_event,
    feature_flags=PLATFORM_FEATURE_FLAGS,
    module_feature_flags=PLATFORM_MODULE_FLAGS,
)))
app.register_blueprint(briefing_blueprint)

register_operations_routes(
    app,
    db=db,
    environment=DEPLOYMENT_ENV,
    limiter=_rate_limiter,
    metrics=_operational_metrics,
    required_tables=CONTROL_TABLES,
    additional_readiness_check=lambda: operational_routes_ready(
        db=db,
        Unit=Unit,
        DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    ),
)

app.cli.add_command(create_roster_cli(create_roster_cli_dependencies(
    db=db,
    operational_models=_operational_models,
    roster_impact_event_type=RosterImpactEventType,
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
