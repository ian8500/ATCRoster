# Production hardening baseline

Evidence date: 31 July 2026

Repository: `ian8500/ATCRoster`

Baseline commit: `80177ec` (`main`)

Runtime used for local evidence: Python 3.12.7

## Executive summary

ATCRoster has a credible controlled-pilot security foundation: production
configuration fails closed, browser mutations have a default-deny CSRF
boundary, tenant routing is derived server-side, security and dependency
checks run in CI, and the complete local test suite is green.

It is not yet a production-grade multi-tenant SaaS platform. The principal
risks are:

1. **Railway topology and release drift (release blocker).** The Railway
   environment named `production` runs web commit `57b0189` from 24 July 2026
   and exposes only `web` and `Postgres`. The public custom domains
   `www.atcroster.com` and `pilot.atcroster.com` are attached to the
   `pilot-staging` web service, which runs commit `80177ec` and has the newer
   worker, Redis and private briefing-storage topology. Production naming,
   routing, dependencies and rollback ownership therefore do not describe the
   service users actually reach.
2. **Import-time application construction and side effects.** `app.py` still
   creates and configures the Flask application, databases, rate limiter,
   storage validation, models, routes, CLI and WSGI surface at module import.
   This blocks clean isolated application instances and makes configuration
   behaviour difficult to reason about.
3. **Monolithic domain coupling.** `app.py` is 12,171 lines, with 316
   top-level functions, 24 classes and 69 directly decorated routes. It still
   owns at least 18 responsibility groups.
4. **Authentication extraction remains broadly coupled.**
   `create_auth_blueprint(core: ModuleType)` receives the complete legacy
   module. The extracted login also strips passwords and enables a legacy
   operational lookup in every non-production environment instead of behind
   a dedicated opt-in flag.
5. **Authorisation is distributed.** There are 68 direct role/permission
   checks or privileged decorators across `app.py`, `briefing_module.py` and
   `auth_blueprint.py`. A single central policy layer and complete
   permission-matrix test do not yet exist.
6. **Coverage is adequate only as a starting floor.** Overall coverage is
   68.63%, but `app.py` is 65% and briefing storage is 57%. The current
   aggregate threshold does not demonstrate the required 90–95% assurance for
   authentication, recovery, tenant routing, policies, publication and
   permission boundaries.

No live Railway changes were made while recording this baseline. Moving
domains or deploying current `main` into the named production environment
without first provisioning and verifying its required Redis, worker,
operational-database and durable-storage dependencies would be unsafe.

## Baseline results

| Control | Command or method | Result |
| --- | --- | --- |
| Tests and coverage | `python -m pytest --cov --cov-report=term-missing -q` | 165 passed, 2 skipped; 68.63% total; 60% floor met |
| Ruff lint | Maintained Python files configured in CI | Passed with no findings |
| Ruff format | Formatted service and extracted modules configured in CI | Six files already formatted |
| Bandit enforced scan | `bandit -q -ll -r ...` | Passed; no medium or high findings |
| Bandit full baseline | Same maintained source set without severity floor | 37 low findings; 0 medium; 0 high |
| Dependency audit | `pip-audit -r requirements-prod.txt` | No known vulnerabilities |
| Alembic | `alembic heads` | One head: `20260730_28` |
| Flask URL map | Imported with runtime schema work disabled | 89 routes; 52 accept an unsafe method |
| Application size | AST and line inventory | `app.py`: 12,171 lines, 316 top-level functions, 24 classes |

The two skipped tests are the existing environment-dependent integration
skips. CI separately runs PostgreSQL 16 and Redis integration coverage.

### Coverage by maintained module

| Module | Coverage |
| --- | ---: |
| `account_limits.py` | 94% |
| `app.py` | 65% |
| `auth_blueprint.py` | 82% |
| `briefing_module.py` | 83% |
| `briefing_storage.py` | 57% |
| `platform_provisioning.py` | 75% |
| `rate_limiting.py` | 85% |
| `saas_models.py` | 99% |
| `signup_locking.py` | 74% |
| `tenancy.py` | 85% |

### Static-analysis debt

The 37 low-severity Bandit findings comprise:

- 21 broad `except Exception`/silent-pass findings in legacy startup,
  compatibility migration, cache and audit paths;
- six empty encrypted-secret comparisons or initial values reported as
  possible hard-coded passwords;
- fixed local process-launch findings in the Railway service wrappers;
- two assertions in the scale-assurance utility;
- one explicitly non-production acceptance-data password.

The enforced medium-and-higher scan is green. The silent exception paths are
real maintainability and observability debt even where they are not directly
exploitable.

## Application responsibilities

`app.py` remains responsible for at least these 18 distinct concerns:

1. Flask construction and production configuration validation;
2. database and extension initialisation;
3. control and operational model declarations;
4. tenant binding and ORM isolation hooks;
5. CSRF, CSP and response security headers;
6. authentication support, sessions and redirect validation;
7. MFA, recovery and invitation workflows;
8. permissions and role interpretation;
9. roster generation, editing and publication;
10. fatigue evaluation and fatigue-rule administration;
11. watches, staff, shifts, requirements and reference-data administration;
12. leave, sickness, overtime, TOIL and shift requests;
13. qualification, training and competency workflows;
14. notifications, email and SMS;
15. metrics, reports, exports and calendar feeds;
16. platform administration and account provisioning;
17. legacy schema repair, seeding and compatibility helpers;
18. CLI, WSGI and local-development entry points.

Authentication routes have begun moving into `auth_blueprint.py`, while
briefing has its own blueprint. Both still depend on core objects defined by
the monolith.

## Route inventory methodology

The inventory was produced from Flask's registered URL map and reviewed
against route bodies, decorators, tenant binding and model usage.

- **Mutating** means the route accepts `POST`, `PUT`, `PATCH` or `DELETE`.
- **Operational-data** means it reads or writes an airport's operational
  database or a tenant-bound briefing/training/competency record. Authentication
  routes that bind an operational person are included.
- **Privileged** means at least one branch performs an administrator,
  manager, roster-editor, sensitive-export, publication, security or
  configuration action. Mixed self-service/administrator routes are included.

Public legal, health, static and control-plane-only authentication reads are
not operational-data routes. A route can belong to more than one inventory.

## All mutating routes

The global before-request boundary validates the session CSRF token for every
unsafe method. There are currently no exemptions. Tests exercise anonymous
login/recovery/invitation mutations as well as authenticated mutations.

| Methods | Route | Endpoint |
| --- | --- | --- |
| GET, POST | `/admin` | `admin` |
| GET, POST | `/admin/fatigue-rules` | `admin_fatigue_rules` |
| GET, POST | `/admin/reference` | `admin_reference` |
| POST | `/admin/requests/<int:rid>/respond` | `admin_request_respond` |
| GET, POST | `/admin/staff/<int:sid>` | `admin_staff_edit` |
| POST | `/admin/staff/<int:sid>/watch-move` | `admin_watch_move` |
| POST | `/admin/staff/watch-move/<int:hid>/delete` | `admin_watch_move_delete` |
| POST | `/admin/staff/watch-move/<int:hid>/edit` | `admin_watch_move_edit` |
| GET, POST | `/admin/toil/new` | `admin_toil_new` |
| POST | `/assign/<int:staff_id>/<ym>/<day>` | `assign_cell` |
| GET, POST | `/briefing/admin` | `briefing.admin` |
| POST | `/briefing/admin/<int:item_id>/publish` | `briefing.publish` |
| POST | `/briefing/admin/<int:item_id>/withdraw` | `briefing.withdraw` |
| POST | `/briefing/admin/message-types/configure` | `briefing.configure_message_types` |
| GET, POST | `/briefing/admin/reports` | `briefing.assurance` |
| POST | `/briefing/admin/reports/<int:run_id>/delete` | `briefing.delete_assurance_report` |
| POST | `/briefing/item/<int:item_id>/acknowledge` | `briefing.acknowledge` |
| POST | `/briefing/item/<int:item_id>/archive` | `briefing.archive_item` |
| POST | `/briefing/item/<int:item_id>/delete` | `briefing.delete_item` |
| POST | `/briefing/item/<int:item_id>/heartbeat` | `briefing.heartbeat` |
| GET, POST | `/competency/<int:sid>` | `competency_profile` |
| GET, POST | `/compliance` | `qualification_compliance` |
| GET, POST | `/invite/<token>` | `accept_invitation` |
| GET, POST | `/leave` | `leave` |
| GET, POST | `/login` | `login` |
| GET, POST | `/login/mfa` | `mfa_challenge` |
| GET, POST | `/login/platform-mfa` | `platform_mfa_challenge` |
| GET, POST | `/login/platform-mfa/setup` | `platform_mfa_setup` |
| POST | `/logout` | `logout` |
| GET, POST | `/messages` | `unit_messages` |
| POST | `/notifications/<int:notification_id>/delete` | `notification_delete` |
| POST | `/notifications/<int:notification_id>/read` | `notification_read` |
| POST | `/notifications/read` | `notifications_read` |
| GET, POST | `/operations/<ym>` | `operations_assurance` |
| GET, POST | `/overtime` | `overtime` |
| GET, POST | `/password` | `password_change` |
| GET, POST | `/planning/scenarios` | `scenarios_page` |
| GET, POST | `/platform/admin` | `platform_admin` |
| GET, POST | `/recover` | `account_recovery` |
| GET, POST | `/recover/approve/<token>` | `approve_account_recovery` |
| GET, POST | `/recover/reset/<token>` | `complete_account_recovery` |
| GET, POST | `/reports` | `reports_index` |
| GET, POST | `/requests` | `requests_page` |
| POST | `/roster/<ym>/publish` | `roster_month_publish` |
| POST | `/roster/<ym>/unpublish` | `roster_month_unpublish` |
| GET, POST | `/security/mfa` | `mfa_setup` |
| GET, POST | `/staff/<int:sid>` | `staff_profile` |
| POST | `/staff/<int:sid>/calendar-token` | `calendar_token_create` |
| GET, POST | `/training/<int:sid>` | `training_profile` |
| GET, POST | `/training/admin` | `training_admin` |
| GET, POST | `/unit/accounts` | `unit_accounts` |
| GET, POST | `/unit/onboarding` | `unit_onboarding` |

## Operational tenant-data routes

Every route in the following inventory either establishes a trusted
operational tenant context or accesses tenant operational data:

```text
/
/admin
/admin/change-log
/admin/fatigue-rules
/admin/reference
/admin/requests/<int:rid>/respond
/admin/sms-audit
/admin/staff/<int:sid>
/admin/staff/<int:sid>/watch-move
/admin/staff/watch-move/<int:hid>/delete
/admin/staff/watch-move/<int:hid>/edit
/admin/toil/new
/assign/<int:staff_id>/<ym>/<day>
/briefing/
/briefing/admin
/briefing/admin/<int:item_id>/publish
/briefing/admin/<int:item_id>/withdraw
/briefing/admin/assurance
/briefing/admin/audit
/briefing/admin/message-types/configure
/briefing/admin/reports
/briefing/admin/reports/<int:run_id>/delete
/briefing/admin/settings
/briefing/archive
/briefing/item/<int:item_id>
/briefing/item/<int:item_id>/acknowledge
/briefing/item/<int:item_id>/archive
/briefing/item/<int:item_id>/delete
/briefing/item/<int:item_id>/document
/briefing/item/<int:item_id>/heartbeat
/calendar/<int:sid>/<token>.ics
/competency/
/competency/<int:sid>
/compliance
/compliance-centre
/compliance-centre/export
/invite/<token>
/leave
/login
/login/mfa
/messages
/metrics
/metrics/export
/modules
/notifications/<int:notification_id>/delete
/notifications/<int:notification_id>/read
/notifications/read
/operations/<ym>
/overtime
/password
/planning/coverage/<ym>
/planning/scenarios
/recover/reset/<token>
/reports
/reports/leave-year
/reports/leave.csv
/reports/leave/<ym>
/reports/sickness
/requests
/roster/<ym>
/roster/<ym>/export
/roster/<ym>/print
/roster/<ym>/publish
/roster/<ym>/unpublish
/security/mfa
/staff/<int:sid>
/staff/<int:sid>/calendar-token
/training/
/training/<int:sid>
/training/admin
/training/analytics
/unit/accounts
/unit/onboarding
```

The legacy development login branch is why `/login` can access operational
data. Production login begins in the control plane and binds operational data
only after trusted membership resolution.

## Privileged-action routes

Routes with at least one privileged or sensitive branch are:

```text
/admin
/admin/change-log
/admin/fatigue-rules
/admin/reference
/admin/requests/<int:rid>/respond
/admin/sms-audit
/admin/staff/<int:sid>
/admin/staff/<int:sid>/watch-move
/admin/staff/watch-move/<int:hid>/delete
/admin/staff/watch-move/<int:hid>/edit
/admin/toil/new
/assign/<int:staff_id>/<ym>/<day>
/briefing/admin
/briefing/admin/<int:item_id>/publish
/briefing/admin/<int:item_id>/withdraw
/briefing/admin/audit
/briefing/admin/message-types/configure
/briefing/admin/reports
/briefing/admin/reports/<int:run_id>/delete
/briefing/admin/settings
/briefing/item/<int:item_id>/delete
/briefing/item/<int:item_id>/document
/compliance
/compliance-centre
/compliance-centre/export
/competency/<int:sid>
/metrics
/metrics/export
/operations/<ym>
/overtime
/platform/admin
/platform/worker-health
/planning/scenarios
/recover/approve/<token>
/reports
/reports/leave-year
/reports/leave.csv
/reports/leave/<ym>
/reports/sickness
/roster/<ym>/export
/roster/<ym>/publish
/roster/<ym>/unpublish
/security/mfa
/staff/<int:sid>
/staff/<int:sid>/calendar-token
/training/<int:sid>
/training/admin
/training/analytics
/unit/accounts
/unit/onboarding
```

This inventory highlights why a central policy layer is required: many mixed
routes implement authorisation inside action branches rather than at one
route boundary.

## Existing controls confirmed

- Production startup rejects weak Flask secrets, SQLite, insecure cookies,
  absent trusted hosts/proxy configuration, missing Redis, missing encryption
  keys and non-durable briefing storage.
- Unsafe browser methods are protected at a single default-deny CSRF boundary.
- Login and logout are CSRF protected; logout is POST-only.
- Session state is cleared before successful authentication is established.
- Login redirects use an explicit local route allowlist.
- The operational tenant comes from authenticated membership and server-side
  database routing metadata.
- Platform administrators are denied operational database access by tests.
- CI runs Ruff, coverage, dependency audit, full maintained-source Bandit,
  CodeQL, gitleaks, container scanning and PostgreSQL/Redis integration tests.
- Production dependencies are pinned and currently have no known published
  vulnerabilities.
- The migration graph has one head and CI upgrades fresh control,
  operational and combined databases.

## Highest-risk prioritisation

### P0 — reconcile Railway environments before another production release

1. Decide which Railway environment is the authoritative production
   environment.
2. Inventory variable **names** and dependencies without copying secret values.
3. Confirm control PostgreSQL, per-airport PostgreSQL, Redis, worker and private
   briefing storage exist in that environment.
4. Record backups and a rollback image.
5. Deploy the same immutable tested commit to staging, run migrations as a
   release job, smoke test, then promote or deploy it to production.
6. Move `www.atcroster.com` only through a documented change window with a
   tested rollback.
7. Rename or retire the misleading environment only after traffic and data
   ownership are verified.

### P1 — Phase 2 application factory and configuration

Create `atcroster/__init__.py`, `config.py`, `extensions.py`, `errors.py` and
`cli.py`, preserving WSGI and Railway entry points. The first factory change
must retain all current fail-closed production checks and add direct
configuration-failure tests. Importing utility modules must not connect to
databases, Redis or storage.

### P1 — authentication correctness before broader extraction

In the first bounded authentication follow-up:

- stop stripping passwords;
- gate legacy operational login behind a dedicated disabled-by-default flag;
- replace the `ModuleType` dependency with narrow typed services;
- introduce distinct typed platform and operational principals;
- add the complete negative-path and session-fixation test matrix.

### P1 — central policy and permission assurance

Define explicit policy functions for sensitive actions and introduce a
parameterised role/action/resource matrix. Extract direct role comparisons
incrementally and require every privileged route above to call a policy.

### P2 — tenant, connection and migration assurance

Add cache and worker tenant-leakage tests, bounded/disposable tenant engines,
pool metrics and capacity calculations. Add per-tenant migration status,
compatibility gating, retry evidence and historical/interrupted migration
tests.

### P2 — browser, accessibility and resilience evidence

Add Playwright and axe coverage for authentication, roster editing,
publication, requests, briefing and mobile navigation. Automate backup
verification and isolated restore rehearsal, then establish external
monitoring independent of GitHub and Railway.

## Phase 1 conclusion

Phase 1 is complete as an evidence baseline. It does not assert production
readiness. The next implementation phase should not start with a broad
`app.py` rewrite; it should first reconcile Railway release topology, then
introduce the application factory with configuration-failure tests.
