# Incremental `app.py` refactor progress

## Baseline

- Branch: `agent/modularise-app-incrementally`
- Starting commit: `d7428da166e3943f94175e8ed65fc46e4abe1996`
- Starting `app.py`: 10,029 lines
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

Framework, cross-cutting security concerns and the authenticated notification
inbox are extracted and independently tested. `app.py` is still substantially above the target because the next
sections are multi-model business transactions. The first is `/unit/accounts`,
which coordinates the control database, an airport database, invitations,
capacity enforcement, audit/session revocation and compensating cleanup.
Moving it as a bulk route would increase production risk, so this branch stops
before that transaction and remains deployable at every commit.

At this boundary `app.py` is 9,670 lines (359 fewer than baseline). Five
public route functions have moved, 29 test functions were added, and the final
local suite reports 296 passed, 11 skipped and 72.02% coverage.

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

1. Account service and policies, then account routes and recovery/MFA routes.
2. Qualification status service and compliance blueprint, retaining the narrow
   live-position eligibility interface.
3. Roster publication transaction and immutable snapshot service.
4. Standardised audit creation helpers with rollback tests.
5. CLI command registration and CLI runner contract tests.
6. Application assembly and extension/model compatibility aliases.
7. Roster, fatigue, notification and remaining reporting business services.

## Temporary compatibility and import state

Compatibility aliases are listed in `application-bootstrap.md`. There are no
wildcard imports, runtime monkey patches or duplicate model/extension objects.
The dependency callbacks used by extracted hooks are deliberate late-bound
edges to model-backed services still defined in `app.py`; they avoid circular
imports while preserving startup order. A clean-process import test detects new
circular-import failures.

The main residual circular-dependency risk is the high number of model and
helper globals still consumed when existing blueprints are registered near the
end of `app.py`. In particular, `briefing_module` imports `db` and `utcnow`
from `app`, so importing that module before the canonical `wsgi -> app` startup
order fails. The clean-process test verifies every module in production startup
order and records this direct-import limitation. Future domain extractions
should replace these edges with small, domain-specific dependency objects.
