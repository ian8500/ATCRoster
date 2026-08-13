# Application module inventory

Baseline commit: `d7428da166e3943f94175e8ed65fc46e4abe1996`. Baseline `app.py`: **10,029 lines**. This inventory describes the compatibility boundary before incremental extraction. Automatic Flask `HEAD` and `OPTIONS` methods are omitted from the route contract because Flask derives them from explicit methods.

Current modularisation milestone: the pure SRATCOH fatigue rules and analysis
engine live in `fatigue_engine.py`; airport-scoped fatigue rule persistence and
the legacy compliance/fatigue administration routes live in
`fatigue_compliance.py`. Compatibility aliases and the original unprefixed Flask
endpoint names remain available for existing callers. From baseline `6eb9a36`,
these extractions reduce `app.py` from 9,675 to 9,047 lines without changing its
route map.

## Responsibilities remaining in `app.py`

| Responsibility | Current ownership and security boundary | Intended boundary |
|---|---|---|
| Application bootstrap and configuration | Flask creation, production config validation, extension wiring and blueprint registration | `atcroster/application.py` compatibility assembly |
| Database initialisation and model aliases | Single SQLAlchemy instance, tenant-routed session, operational models and imported SaaS aliases | Existing extension plus explicit compatibility aliases; no duplicate models |
| Tenant binding | Before/after request hooks, verified membership binding, ORM read/write tenant enforcement | `atcroster/tenancy/hooks.py` reusing `tenancy.py` context |
| Authentication and sessions | Flask-Login loader, auth stamps, idle/absolute timeout and account/MFA invalidation | `atcroster/security/sessions.py` and existing auth blueprint |
| CSRF | Session token creation and global default-deny unsafe-method hook | `atcroster/security/csrf.py` |
| Headers and CSP | Nonce creation, CSP/HSTS/cache headers and metrics completion | `atcroster/security/headers.py` |
| Error handling | 400/403/404/500 rendering, request IDs and security logging | `atcroster/errors/handlers.py` |
| Public/legal | Favicon and privacy/cookie/terms/subprocessor pages | `atcroster/public/blueprint.py` |
| Account administration | Unit accounts, invitations, recovery, role/MFA/session lifecycle | `atcroster/accounts/` service, policies and blueprint |
| Platform administration | Tenant provisioning, platform MFA and worker status | `atcroster/platform/` reusing provisioning service |
| Qualifications/compliance | Qualifications, medical/UE validity, warnings, reports and live-position eligibility | `atcroster/qualifications/` narrow status service |
| Roster and publication | Roster rules, fatigue, publication transaction/snapshot and notification | Existing roster blueprint plus `atcroster/publication/` service |
| Notifications | SMS/email delivery, notification records and audit | `atcroster/notifications/service.py` |
| Audit helpers | Change, annotation, request and SMS audit creation | `atcroster/audit/service.py`; DB append-only controls remain central |
| Reports and metrics | Operational metrics calculations and report entry routes | Existing report/operations blueprints and reporting service |
| CLI and compatibility migrations | Seed/setup and legacy maintenance commands/functions | `atcroster/cli/commands.py`; Alembic remains authoritative |
| Encryption | Versioned Fernet field encryption and startup key validation | `atcroster/security/encryption.py` service |
| Background integration | Provisioning worker queue/health and external briefing storage | Existing services with explicit bootstrap dependencies |

## Registered route compatibility inventory

CSRF is enforced globally for every explicit unsafe method. Authentication and role entries state the governing boundary; individual service/policy checks remain authoritative. “JSON/redirect/response” means no statically discoverable template call in the registered view.

| URL | Endpoint | Methods | Authentication | Role/permission | Tenant scope | CSRF | Template/response | Owner/dependencies |
|---|---|---|---|---|---|---|---|---|
| / | `index` | GET | Anonymous or token-bound | Public/token workflow policy | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `app` + registered dependencies |
| /__can | `__can` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `app` + registered dependencies |
| /admin | `admin` | GET, POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | admin.html | `app` + registered dependencies |
| /admin/change-log | `change_log_page` | GET | Authenticated account | Admin/editor action policy | Verified active unit | Not applicable (safe method) | change_log.html | `app` + registered dependencies |
| /admin/fatigue-rules | `admin_fatigue_rules` | GET, POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | admin_fatigue_rules.html | `fatigue_compliance` + registered dependencies |
| /admin/reference | `admin_reference` | GET, POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | admin_reference.html | `app` + registered dependencies |
| /admin/requests/<int:rid>/respond | `admin_request_respond` | POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | JSON/redirect/response | `absence_requests_blueprint` + registered dependencies |
| /admin/sms-audit | `admin_sms_audit` | GET | Authenticated account | Admin/editor action policy | Verified active unit | Not applicable (safe method) | admin_sms_audit.html | `app` + registered dependencies |
| /admin/staff/<int:sid> | `admin_staff_edit` | GET, POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | staff_edit.html | `app` + registered dependencies |
| /admin/staff/<int:sid>/watch-move | `admin_watch_move` | POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /admin/staff/watch-move/<int:hid>/delete | `admin_watch_move_delete` | POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /admin/staff/watch-move/<int:hid>/edit | `admin_watch_move_edit` | POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /admin/toil/new | `admin_toil_new` | GET, POST | Authenticated account | Admin/editor action policy | Verified active unit | Global default-deny | admin_toil_new.html | `app` + registered dependencies |
| /administration | `administration_home` | GET | Authenticated account | UnitAdmin/domain policy | Verified active unit | Not applicable (safe method) | administration_home.html | `atcroster.administration` + injected dependencies |
| /administration/kiosk-accounts | `kiosk_accounts` | GET, POST | Authenticated account | UnitAdmin/domain policy | Verified active unit | Global default-deny | kiosk_accounts.html | `app` + registered dependencies |
| /assign/<int:staff_id>/<ym>/<day> | `assign_cell` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `roster_blueprint` + registered dependencies |
| /briefing/ | `briefing.home` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | briefing/home.html | `briefing_module` + registered dependencies |
| /briefing/admin | `briefing.admin` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | briefing/admin.html | `briefing_module` + registered dependencies |
| /briefing/admin/<int:item_id>/publish | `briefing.publish` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/admin/<int:item_id>/withdraw | `briefing.withdraw` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/admin/assurance | `briefing.legacy_assurance` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/admin/audit | `briefing.audit` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | briefing/audit.html | `briefing_module` + registered dependencies |
| /briefing/admin/message-types/configure | `briefing.configure_message_types` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/admin/reports | `briefing.assurance` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | briefing/assurance.html | `briefing_module` + registered dependencies |
| /briefing/admin/reports/<int:run_id>/delete | `briefing.delete_assurance_report` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/admin/settings | `briefing.settings` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | briefing/settings.html | `briefing_module` + registered dependencies |
| /briefing/archive | `briefing.archive` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | briefing/archive.html | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id> | `briefing.view_item` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | briefing/item.html | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id>/acknowledge | `briefing.acknowledge` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id>/archive | `briefing.archive_item` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id>/delete | `briefing.delete_item` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id>/document | `briefing.document` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `briefing_module` + registered dependencies |
| /briefing/item/<int:item_id>/heartbeat | `briefing.heartbeat` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `briefing_module` + registered dependencies |
| /calendar/<int:sid>/<token>.ics | `calendar_feed` | GET | Anonymous or token-bound | Public/token workflow policy | Server-token resolved | Not applicable (safe method) | JSON/redirect/response | `atcroster.calendar_feed` + injected dependencies |
| /competency/ | `competency_home` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | competency_home.html | `training_blueprint` + registered dependencies |
| /competency/<int:sid> | `competency_profile` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | competency_profile.html | `training_blueprint` + registered dependencies |
| /compliance | `qualification_compliance` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | qualification_compliance.html | `app` + registered dependencies |
| /compliance-centre | `compliance_centre` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `fatigue_compliance` + registered dependencies |
| /compliance-centre/export | `compliance_centre_export` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `fatigue_compliance` + registered dependencies |
| /cookies | `cookie_notice` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | cookies.html | `app` + registered dependencies |
| /favicon.ico | `favicon` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | static asset | `app` + registered dependencies |
| /health/live | `health_live` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | JSON/redirect/response | `production_operations` + registered dependencies |
| /health/ready | `health_ready` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | JSON/redirect/response | `production_operations` + registered dependencies |
| /internal/health | `internal_health` | GET | Internal bearer token | Metrics token | Unbound/public | Not applicable (safe method) | JSON/redirect/response | `production_operations` + registered dependencies |
| /internal/metrics | `internal_metrics` | GET | Internal bearer token | Metrics token | Unbound/public | Not applicable (safe method) | JSON/redirect/response | `production_operations` + registered dependencies |
| /invite/<token> | `accept_invitation` | GET, POST | Anonymous or token-bound | Public/token workflow policy | Server-token resolved | Global default-deny | invitation_accept.html | `app` + registered dependencies |
| /leave | `leave` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | leave.html | `absence_requests_blueprint` + registered dependencies |
| /live-positions/ | `live_position.admin_home` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/admin/positions | `live_position.position_configuration` | GET, POST | Authenticated account | UnitAdmin/domain policy | Verified active unit | Global default-deny | live_position/position_configuration.html | `live_position_blueprint` + registered dependencies |
| /live-positions/api/controllers | `live_position.controllers` | GET | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/events | `live_position.live_events` | GET | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/close | `live_position.close_position` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/handover | `live_position.handover` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/logoff | `live_position.logoff` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/logon | `live_position.logon` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/open | `live_position.open_position` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/participants | `live_position.add_participant` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/positions/<int:position_id>/participants/<int:participant_id>/logoff | `live_position.remove_participant` | POST | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/api/state | `live_position.live_state` | GET | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `live_position_blueprint` + registered dependencies |
| /live-positions/kiosk | `live_position.kiosk_hmi` | GET | Kiosk session or authorised user | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | live_position/kiosk.html | `live_position_blueprint` + registered dependencies |
| /live-positions/reports/operational-activity | `live_position.operational_activity` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | live_position/operational_activity_report.html | `live_position_blueprint` + registered dependencies |
| /login | `login` | GET, POST | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Global default-deny | login.html | `auth_blueprint` + registered dependencies |
| /login/mfa | `mfa_challenge` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | mfa_challenge.html | `app` + registered dependencies |
| /login/platform-mfa | `platform_mfa_challenge` | GET, POST | Platform identity + MFA | Endpoint/domain permission | Control DB/platform | Global default-deny | mfa_challenge.html | `app` + registered dependencies |
| /login/platform-mfa/setup | `platform_mfa_setup` | GET, POST | Platform identity + MFA | Endpoint/domain permission | Control DB/platform | Global default-deny | mfa_setup.html | `app` + registered dependencies |
| /logout | `logout` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `auth_blueprint` + registered dependencies |
| /messages | `unit_messages` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | messages.html | `app` + registered dependencies |
| /metrics | `metrics` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | metrics.html | `reports_blueprint` + registered dependencies |
| /metrics/export | `metrics_export` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `reports_blueprint` + registered dependencies |
| /modules | `module_home` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | module_home.html | `app` + registered dependencies |
| /notifications/<int:notification_id>/delete | `notification_delete` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /notifications/<int:notification_id>/read | `notification_read` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /notifications/read | `notifications_read` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /operations/<ym> | `operations_assurance` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | operations_assurance.html | `operations_blueprint` + registered dependencies |
| /overtime | `overtime` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | overtime.html | `app` + registered dependencies |
| /password | `password_change` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | password.html | `app` + registered dependencies |
| /planning/coverage/<ym> | `coverage_heatmap` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | coverage_heatmap.html | `operations_blueprint` + registered dependencies |
| /planning/scenarios | `scenarios_page` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | scenarios.html | `operations_blueprint` + registered dependencies |
| /platform/admin | `platform_admin` | GET, POST | Platform identity + MFA | SuperAdmin/platform policy | Control DB/platform | Global default-deny | platform_admin.html | `app` + registered dependencies |
| /platform/worker-health | `platform_worker_health` | GET | Platform identity + MFA | SuperAdmin/platform policy | Control DB/platform | Not applicable (safe method) | JSON/redirect/response | `app` + registered dependencies |
| /privacy | `privacy_notice` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | privacy.html | `app` + registered dependencies |
| /recover | `account_recovery` | GET, POST | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Global default-deny | account_recovery.html | `app` + registered dependencies |
| /recover/approve/<token> | `approve_account_recovery` | GET, POST | Anonymous or token-bound | Public/token workflow policy | Server-token resolved | Global default-deny | recovery_approve.html | `app` + registered dependencies |
| /recover/reset/<token> | `complete_account_recovery` | GET, POST | Anonymous or token-bound | Public/token workflow policy | Server-token resolved | Global default-deny | recovery_reset.html | `app` + registered dependencies |
| /reports | `reports_index` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | reports_index.html | `reports_blueprint` + registered dependencies |
| /reports/leave-year | `report_leave_year` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | report_leave_year.html | `reports_blueprint` + registered dependencies |
| /reports/leave.csv | `report_leave_csv` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `reports_blueprint` + registered dependencies |
| /reports/leave/<ym> | `report_leave` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | report_leave.html | `reports_blueprint` + registered dependencies |
| /reports/sickness | `report_sickness` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | report_sickness.html | `reports_blueprint` + registered dependencies |
| /requests | `requests_page` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | requests.html | `absence_requests_blueprint` + registered dependencies |
| /roster/<ym> | `roster_month` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | roster_month.html | `roster_blueprint` + registered dependencies |
| /roster/<ym>/export | `roster_export_csv` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `roster_blueprint` + registered dependencies |
| /roster/<ym>/print | `roster_print_view` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | JSON/redirect/response | `roster_blueprint` + registered dependencies |
| /roster/<ym>/publish | `roster_month_publish` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `roster_blueprint` + registered dependencies |
| /roster/<ym>/unpublish | `roster_month_unpublish` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `roster_blueprint` + registered dependencies |
| /security/mfa | `mfa_setup` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | mfa_setup.html | `app` + registered dependencies |
| /staff/<int:sid> | `staff_profile` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | staff_profile.html | `app` + registered dependencies |
| /staff/<int:sid>/calendar-token | `calendar_token_create` | POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | JSON/redirect/response | `app` + registered dependencies |
| /static/<path:filename> | `static` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | JSON/redirect/response | `flask.app` + registered dependencies |
| /subprocessors | `subprocessor_notice` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | subprocessors.html | `app` + registered dependencies |
| /terms | `terms_of_service` | GET | Anonymous or token-bound | Public/token workflow policy | Unbound/public | Not applicable (safe method) | terms.html | `app` + registered dependencies |
| /training/ | `training_home` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | training_home.html | `training_blueprint` + registered dependencies |
| /training/<int:sid> | `training_profile` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | training_profile.html | `training_blueprint` + registered dependencies |
| /training/admin | `training_admin` | GET, POST | Authenticated account | Endpoint/domain permission | Verified active unit | Global default-deny | training_admin.html | `training_blueprint` + registered dependencies |
| /training/analytics | `training_analytics` | GET | Authenticated account | Endpoint/domain permission | Verified active unit | Not applicable (safe method) | training_analytics.html | `training_blueprint` + registered dependencies |
| /unit/accounts | `unit_accounts` | GET, POST | Authenticated account | UnitAdmin/domain policy | Verified active unit | Global default-deny | unit_accounts.html | `app` + registered dependencies |
| /unit/onboarding | `unit_onboarding` | GET, POST | Authenticated account | UnitAdmin/domain policy | Verified active unit | Global default-deny | unit_onboarding.html | `app` + registered dependencies |

## Extraction constraints

- `app`, `db`, `login_manager`, model classes and `wsgi:application` remain singletons/compatibility exports.
- Blueprint endpoint names are contractual and covered by `tests/fixtures/route_map.json`.
- Tenant context must be empty before public work and cleared after every response or exception.
- Global unsafe-method CSRF and post-response security headers remain centrally registered until their explicit extraction commits.
- No domain extraction may bypass existing policy helpers, tenant-scoped queries, audit writes or transaction boundaries.
