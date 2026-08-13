# Incremental application-composition refactor progress

## Baseline

- Active refactor branch: `codex/application-composition-root`
- Starting commit: `d7428da166e3943f94175e8ed65fc46e4abe1996`
- Starting legacy application module: 10,029 lines
- Baseline tests: 264 passed, 11 skipped
- Baseline coverage: 71.50%
- Route snapshot: 107 routes

## Extracted domains

| Domain | New owner |
|---|---|
| Route compatibility contract | `tests/fixtures/route_map.json` and `tests/test_route_map_contract.py` |
| Security headers/CSP | `atcroster/security/headers.py` |
| Error handlers | `atcroster/errors.py` |
| Public/legal pages | `atcroster/public/blueprint.py` |
| CSRF | `atcroster/security/csrf.py` |
| Field encryption | `atcroster/security/encryption.py` |
| Session lifecycle | `atcroster/security/sessions.py` |
| Tenant request hooks | `atcroster/tenancy_hooks.py` |
| Notification inbox routes | `atcroster/notifications/blueprint.py` |
| Module launcher | `atcroster/modules/blueprint.py` |
| Calendar subscription feed | `atcroster/calendar_feed.py` |
| Administration landing | `atcroster/administration/blueprint.py` |
| Authenticated home redirect | `atcroster/home.py` |
| Password change | `atcroster/accounts/password.py` |
| Live Position kiosk accounts | `atcroster/accounts/kiosk.py` |
| Operational currency configuration | `atcroster/live_position/currency.py` |
| Manual TOIL adjustments | `atcroster/administration/toil.py` |
| Application composition root | `atcroster/application.py` |
| Permission summary and change-log routes | `atcroster/admin_utilities.py` |
| SMS normalization and provider delivery | `atcroster/notifications/sms.py` |
| Account email delivery and address validation | `atcroster/notifications/email.py` |
| Unit-scoped SMS configuration selection | `atcroster/notifications/configuration.py` |
| Successful SMS delivery audit persistence | `atcroster/notifications/audit.py` |
| Overtime SMS delivery orchestration | `atcroster/notifications/overtime.py` |
| SMS audit administration and delivery webhook | `atcroster/notifications/admin.py` |
| Unit messaging route | `atcroster/notifications/messaging.py` |
| Platform provisioning worker health | `atcroster/platform/worker_health.py` |

## Safe stopping boundary

`app.py` is now a small public compatibility entrypoint. The active
composition root is `atcroster/application.py`; it is approximately 2,300
lines and still exceeds the hard 2,000-line target. The remaining work is to
move feature registration and the few root-owned policies into their existing
domain packages without changing the legacy import surface.

## Verification at this boundary

- Route snapshot: unchanged, 107 routes.
- Ruff: passed for every tracked Python file.
- Mypy: passed for the configured typed scope and all new package modules.
- Bandit Medium/High gate: passed.
- Compileall: passed.
- Container build: passed.
- Migration and backup unit/fixture tests: 22 passed.
- PostgreSQL/Redis integration: 10 passed; the backup/restore integration case
  reached `pg_dump` but could not run locally because the host client is version
  14 while the isolated PostgreSQL server is version 16. Exact-commit CI must
  verify that remaining case with its matching client tools.

## Remaining domains, in recommended order

1. Move roster and absence-request registration into their existing packages,
   reducing the composition root below the hard target.
2. Replace oversized registration dependency objects with smaller feature
   contracts where a shared dependency has a natural owner.
3. Move the remaining root policy helpers (fatigue adapters, staff-profile
   authorisation, and admin-action assembly) to their domain packages.
4. Publish and test the legacy application export contract before removing any
   remaining aliases.

## Temporary compatibility and import state

Compatibility aliases are listed in `application-bootstrap.md`. There are no
duplicate model or extension objects. Two legacy callback hooks remain for
post-startup replacement of email and SMS delivery in integrations; they are
centralised in `atcroster.compatibility`. Construction cycles use explicit
single-assignment deferred references from `atcroster.composition`, which fail
clearly when assembly order is invalid. A clean-process import test detects new
circular-import failures.

The public `app.py` module is a small compatibility entrypoint. Application
composition and legacy model compatibility live in `atcroster.application`, so
WSGI, workers, scripts, and existing integrations retain the stable `app`
import while new code can depend on domain modules directly. `briefing_module`
now imports the composition module explicitly, avoiding a reverse dependency
on the public compatibility entrypoint.
