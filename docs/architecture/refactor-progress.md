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
| Administration route registration | `atcroster/administration/registration.py` |
| Roster and overtime route/CLI registration | `atcroster/roster/registration.py` |
| Request route registration | `atcroster/requests/registration.py` |
| Platform route and operations registration | `atcroster/platform/registration.py` |
| Notification and training route registration | `atcroster/notifications/registration.py`, `atcroster/training/registration.py` |

## Safe stopping boundary

`app.py` is now a small public compatibility entrypoint. The active
composition root is `atcroster/application.py` (1,722 lines), below the hard
2,000-line target. Its remaining size is predominantly explicit construction
and thin legacy exports, rather than route or domain implementation. The
preferred 1,500-line target remains a conservative follow-up only where an
existing domain can own a coherent registration contract without obscuring
the assembly flow or changing the legacy import surface.

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

1. Move administration and account registration into their existing packages,
   reducing the composition root toward the preferred target.
2. Continue replacing oversized registration dependency objects with smaller feature
   contracts where a shared dependency has a natural owner.
3. Maintain and extend the legacy application export contract before removing any
   remaining aliases.

## Temporary compatibility and import state

Compatibility aliases are listed in `application-bootstrap.md`. There are no
duplicate model or extension objects. Two legacy callback hooks remain for
post-startup replacement of email and SMS delivery in integrations; they are
centralised in `atcroster.compatibility`. Construction cycles use explicit
single-assignment deferred references from `atcroster.composition`, which fail
clearly when assembly order is invalid. A clean-process import test detects new
circular-import failures.

The public application export contract is covered by
`tests/test_application_compatibility.py`. Any extraction that removes a
legacy model, helper, or service alias must either retain that alias or update
the contract alongside a deliberate integration migration.

The public `app.py` module is a small compatibility entrypoint. Application
composition and legacy model compatibility live in `atcroster.application`, so
WSGI, workers, scripts, and existing integrations retain the stable `app`
import while new code can depend on domain modules directly. Optional briefing
receives its cross-domain collaborators through an explicit composition-time
contract; it no longer imports the composition root. This boundary is enforced
by `tests/test_application_compatibility.py`.
