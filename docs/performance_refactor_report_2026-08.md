# Performance refactor report — August 2026

## Executive summary

This branch reduces the monthly roster's repeated editing markup and moves
monthly display calculations into `atcroster.roster.month_view`.  Versioned
static assets now receive immutable browser caching, while authenticated dynamic
responses remain private and non-cacheable.

## Measured and structural results

| Area | Result |
| --- | --- |
| Targeted roster/security tests | 11 passed |
| Editable roster cell structure | Replaced a form, two hidden inputs and a select with one button |
| Shift editor instances | One shared dialog per page instead of one form per editable cell |
| Static version resolution | Cached per process (256-entry LRU) |
| Versioned static cache policy | `public, max-age=31536000, immutable` |

No representative authenticated database fixture or browser runner was available
locally, so SQL count, HTML byte, gzip byte, DOM-node and render-time percentage
claims are intentionally not reported.

The production import path was also checked three times against the supported
SQLite smoke configuration after lazy-loading QR rendering: 0.51s, 0.49s and
0.49s. This is a repeatable sanity check, not a before/after performance claim,
because the earlier baseline could not use the same local database configuration.

## Monthly roster

`RosterMonthViewService` owns staff ordering, operational counters, RAG states,
expiry classes and watch separators. The route continues to own authoritative
queries, cache invalidation, fatigue/validation calculation and publication
checks. This keeps safety-critical write and publication decisions out of the
display service.

The shared modal shift editor preserves CSRF, assignment versions, baseline
reset, asynchronous saves, optimistic-concurrency errors, focus restoration and
immediate day-summary updates. Protected absences remain non-editable.

## Frontend and caching

Roster cells now expose their state via data attributes and use one page-level
editor. Existing design classes, fatigue/validation indicators, annotations,
request indicators, sticky headers and zoom controls are retained.

Only URLs with the application-generated `v` parameter receive immutable public
caching. Dynamic authenticated pages continue to use `no-store, private`.

## Architecture and safety

`app.py` remains the production bootstrap compatibility boundary, but the
notification inbox, module launcher, administration landing and calendar
subscription/token routes now own their route behavior in injected-dependency
modules. None creates another Flask or SQLAlchemy instance.

## Validation and residual issues

Executed successfully:

```text
PYTHONPATH=. python -m ruff check roster_blueprint.py atcroster/roster \
  atcroster/security/headers.py tests/test_roster_month_view.py \
  tests/test_security_headers.py
PYTHONPATH=. python -m pytest -q tests/test_roster_month_view.py \
  tests/test_roster_month_cache.py tests/test_security_headers.py \
  tests/test_roster_blueprint.py
```

The missing operational-currency duration helper was restored, stale
multi-database integration assertions were advanced to the actual Alembic head
(`20260808_56`), and the handover migration was made compatible with historical
`unit` schemas on both SQLite and PostgreSQL. The targeted Ruff check now
passes.

Railway staging is an isolated service with separate staging databases. This
branch was deployed successfully to `https://pilot.atcroster.com`: control and
operational databases both migrated to `20260812_57`, Waitress started, and
liveness, readiness and login-page checks passed. The deployment reports
`"environment":"staging"`. The separately deployed staging worker also
completed the same migrations and started successfully. Production was not
changed.

The staging repair configured explicit owner-backed migration URLs, disabled
runtime `db.create_all()` through `ATCROSTER_SKIP_RUNTIME_SCHEMA=1`, and aligned
both web and worker services to `ATCROSTER_ENVIRONMENT=staging`. Legacy schema
ownership was corrected to the staging migration role; a temporary runtime
schema-create grant used only to complete the legacy migration was revoked after
the healthy deployment.

## Next work

1. Run a seeded PostgreSQL/Redis benchmark on `main` and this branch and add
   stable query-count/HTML-size regression tests.
2. Add browser-based visual and interaction coverage for the shared roster
   editor before further presentation changes.
3. Continue extracting authoritative month data loading from `roster_blueprint`
   into the roster domain without moving publication eligibility out of its
   transaction boundary.
