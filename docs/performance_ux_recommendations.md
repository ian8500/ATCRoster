# Roster performance and UX status

This document records the current, repository-verifiable state of the monthly
roster surface. It is not a production capacity claim.

## Implemented

- The monthly roster uses the modular month-view and roster-domain services;
  the root `app.py` remains only the compatibility/WSGI entrypoint.
- Repeated month loads use the bounded month cache. Assignment mutations
  invalidate the affected unit/month cache entry.
- Shift assignment uses one delegated, asynchronous editor dialog rather than
  a form and select for every editable cell. The server remains authoritative:
  the flow includes CSRF, permission checks, validation, audit logging and
  optimistic-concurrency responses.
- The editor preserves roster position after a save, exposes a saved-session
  status, supports keyboard movement between editable cells, and keeps one
  inspector, command palette and readiness interface per page.
- Roster layout fitting is scheduled with `requestAnimationFrame` and observes
  relevant size changes where `ResizeObserver` is available.
- Position Monitor uses a similarly bounded live-state display and makes a
  failed refresh or disconnected event stream visibly stale rather than
  presenting last-known data as current.

## Measurement

`scripts/measure_http_performance.py` records repeatable HTTP median latency,
HTML size, gzip size and DOM-element count against local or staging URLs. Pass
an authenticated session cookie only to a controlled environment; never place
credentials in source control or benchmark artefacts.

Browser acceptance tests protect the delegated roster-editor and Position
Monitor workflows. They are functional regression tests, not a substitute for
a representative load test.

## Next operational measurements

Before a production capacity commitment, run a seeded PostgreSQL and Redis
benchmark with representative unit sizes and capture:

- month-load and edit p50/p95/p99 latency;
- SQL-query counts and database-pool saturation;
- roster HTML/DOM size at representative staffing levels;
- cache hit/invalidation behaviour;
- live-position update and reconnection latency; and
- error rates during concurrent editor activity.

The operational procedure and acceptance criteria are documented in
`docs/operations/performance-and-capacity.md`.
