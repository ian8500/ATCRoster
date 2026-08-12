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
SQLite smoke configuration after lazy-loading QR rendering: 0.47s, 0.48s and
0.47s. This is a repeatable sanity check, not a before/after performance claim,
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

`app.py` was 10,847 lines at the start of this work and remains the production
bootstrap compatibility boundary. The new view service does not create another
Flask or SQLAlchemy instance and has no dependency on `app.py` globals.

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

Railway staging is an isolated service with separate staging databases. Its
read-only liveness, readiness and login-page checks passed at
`https://pilot.atcroster.com`; this branch has not been deployed there or to
production. The draft pull request's CI workflow remains the release gate.

The staging service currently reports `"environment":"production"` from its
liveness payload despite being linked to Railway's staging environment. This is
an environment-label configuration discrepancy to correct before treating that
service as a final staging acceptance target; it does not change the separate
database or service target selected for the blocked pre-deploy attempt.

## Next work

1. Run a seeded PostgreSQL/Redis benchmark on `main` and this branch and add
   stable query-count/HTML-size regression tests.
2. Move the remaining roster-specific inline CSS and JavaScript into versioned
   static assets after visual browser regression coverage is in place.
3. Continue extracting authoritative month data loading from `roster_blueprint`
   into the roster domain without moving publication eligibility out of its
   transaction boundary.
