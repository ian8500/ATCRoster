# Production-readiness remediation plan

Evidence baseline: 1 August 2026  
Repository: `ian8500/ATCRoster`  
Baseline revision: `d0b8a30`

This is a living, evidence-led plan. A status of **verified** means the named
repository control and automated evidence exist; it is not a certification of
the deployed service or its operating organisation.

## Second-phase baseline — 1 August 2026

Baseline revision: `35d3c54b90faef25c5564b0f6a38f6a3765c23a3`

- Application suite on Python 3.12.7: 249 passed, 3 skipped, 8 warnings in
  97.11 seconds; 71.14% coverage. The skips are the dedicated PostgreSQL/Redis
  cases.
- PostgreSQL 16 and Redis 7 integration: physical control/two-airport isolation,
  provisioning concurrency and Redis atomic-window tests passed. The generated
  backup case failed only because the local `pg_dump` 14 client refuses a
  PostgreSQL 16 server; the same backup/restore test passed against a matching
  PostgreSQL 14 server/client. CI at this exact baseline had already passed the
  PostgreSQL 16 service job, including generated backup restore.
- Blank control, operational and combined migrations reached the single Alembic
  head `20260801_34`. Representative legacy migration and backup unit tests:
  11 passed in 3.31 seconds.
- Maintained-source Ruff formatting/lint and mypy passed. Repository-wide Ruff
  formatting discovery identified 54 previously untracked files that are not
  yet governed by the maintained-source list; this is a CI coverage gap, not a
  runtime failure.
- `pip-audit`: no known production dependency vulnerabilities. Bandit High and
  Medium gate passed (two acknowledged `nosec` notices only). Whole-repository
  `compileall` passed.
- Baseline `app.py`: 9,907 lines. Docker is available locally; container build
  and High/Critical Trivy enforcement remain independently green in GitHub CI
  for the exact baseline revision.

## Baseline evidence

- Application suite: 249 passed, 3 skipped, 8 warnings (1 August 2026 local
  verification; coverage remains enforced by CI).
- The three skips require the CI PostgreSQL/Redis services; those tests run in a
  separate integration job.
- Dependency audit: no known vulnerabilities in `requirements-prod.txt`.
- Bandit: no High findings; two Low-confidence Medium migration-SQL findings
  whose identifiers are fixed code constants rather than external input.
- Alembic: one head; fresh control, operational and combined upgrades are CI
  gates, with representative legacy SQLite upgrades in the application suite.
- Existing CI also enforces Ruff, mypy, CodeQL, gitleaks, PostgreSQL/Redis
  integration, container build and High/Critical Trivy scanning.

## Second-phase verification — 1 August 2026

Code revision verified locally: `8582d50` (the following documentation-only
evidence commit does not change runtime code).

- Python 3.12.13 complete suite: **264 passed, 11 skipped, 19 warnings in
  127.45 seconds; 71.47% coverage**. The 11 skips are the opt-in PostgreSQL,
  Redis and generated-backup cases reported separately below.
- PostgreSQL 16 + Redis 7 without the locally mismatched backup client:
  **10 passed, 1 deselected, 13 warnings in 8.73 seconds**. Coverage includes
  control/two-airport isolation, tenant constraints, runtime audit grants,
  provisioning, roster/request/publication/TOIL/live-position concurrency and
  privacy-safe Redis keys.
- Matching PostgreSQL 14 server/client, Redis 7 and generated backup/restore:
  **11 passed, 14 warnings in 10.13 seconds**. The restored database reached
  Alembic revision `20260801_37` and preserved its recovery probe.
- Blank SQLite control, operational and combined upgrades reached the single
  Alembic head `20260801_37`. Representative legacy migrations: **7 passed in
  3.17 seconds**. The PostgreSQL migration test also refuses inconsistent
  cross-unit legacy data rather than rewriting ownership.
- Repository-discovered Ruff passed for every tracked Python file. Mypy passed
  for the typed package/production primitives. Bandit High/Medium gate passed
  with the two expected `B104` health-listener `nosec` notices. `compileall`
  passed. The installed pinned environment reported **no known
  vulnerabilities**; the requirement-file resolver separately crashed inside
  temporary `ensurepip`, so it is not misreported as scan evidence.
- Docker image `atcroster:limited-production` built successfully. Local Trivy
  is unavailable; exact-commit High/Critical Trivy and CycloneDX SBOM results
  require the GitHub container job and are not inferred here.
- `app.py` is 10,029 lines versus the 9,907-line phase baseline. Production
  health, metrics and structured logging logic is now isolated in the
  266-line `production_operations.py`; the net increase reflects new
  concurrency, TOIL and observability controls rather than a claimed monolith
  reduction.

## Remediation register

| ID | Severity | Component | Risk | Proposed remediation | Status | Tests/evidence | Residual risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PR-001 | Critical | Tenant routing | Browser-controlled tenant selection or stale context could expose another airport. | Keep tenant binding derived from active control-plane membership; deny platform operational binds; clear context on every request/job; expand hostile matrix tests. | Verified; expansion ongoing | Physical multi-database, tenant router, access-policy and hostile ID tests in CI. | Independent authenticated penetration testing remains required. |
| PR-002 | High | Weekly position limits | Free-form text cannot enforce slot schema, duration bounds or same-tenant position linkage. | Migrate to relational rows with composite tenant FK, unique slot and database checks; reject invalid legacy documents. | Implemented | Revision `20260801_34`; legacy conversion and invalid-data migration tests; live-position workflow tests. | Effective-dated future policy versions are not currently required; sessions preserve their applied snapshot. |
| PR-003 | High | Authentication sessions | A stolen or stale session must stop after password/MFA/account state changes. | Verify version/timestamp invalidation for every path and add missing negative tests without weakening timeouts or cookie policy. | Implemented | Session stamp covers password, role, membership and airport/platform MFA state; stale rejected principals are cleared; multi-client revocation plus idle/absolute timeout tests. | Browser/device compromise before revocation remains outside application control. |
| PR-004 | High | Authorisation | Distributed role checks may drift or permit ID tampering. | Move high-risk decisions through central policy functions and extend the parameterised role/action/resource denial matrix. | High-risk live-position boundary implemented; broader extraction ongoing | Live service independently requires an active same-unit kiosk actor, scopes idempotency evidence by unit, rejects cross-unit supporting roles and inactive controllers; `test_access_policy.py`, permission report, route and tenant-isolation tests. | Full route-to-policy extraction remains incremental while `app.py` is decomposed. |
| PR-005 | High | Database tenant integrity | Single-column foreign keys can attach related operational rows to a different unit in a combined or misrouted schema. | Add justified composite `(unit_id, id)` keys/FKs to high-risk relationships and deliberate cross-unit failure tests. | Implemented and locally verified | Revision `20260801_35`; preflight diagnostics; valid/invalid migration tests; deliberate PostgreSQL cross-unit insertion failure; tenant-integrity inventory. | Historical actor fields and physical control-to-operational identity links cannot all use same-database FKs; SQLite does not emulate the production constraint boundary. |
| PR-006 | High | Backup and restore | Provider snapshots alone do not prove recoverability of control and per-airport databases. | Add safe PostgreSQL backup, checksum/metadata verification and restore tooling plus an automated generated-backup restore test. | Implemented in repository | Daily encrypted off-Railway workflow; backup/verify/restore scripts; PostgreSQL generated-backup restore CI test. | Off-site retention approval, recovery-key custody and scheduled operator rehearsal remain deployment-owner actions. |
| PR-007 | High | Audit integrity | Inconsistent audit schemas or non-atomic writes can weaken accountability. | Enforce append-only evidence at the application and database-role boundaries; keep sensitive values out; test atomic commits. | Implemented and locally verified | Repeatable apply/verify grant tooling; PostgreSQL runtime login proves ordinary update plus audit insert/select, and denies audit update/delete and schema creation. | Grants must be applied to every managed database; provider-owner access and external immutable log retention remain external. |
| PR-008 | High | Concurrency/idempotency | Duplicate decisions, logons, handovers, TOIL or provisioning could corrupt state. | Retain unique transaction keys, row/month locking, CAS and retry-safe state machines; add PostgreSQL concurrency coverage. | Implemented and locally verified | Real separate-session PostgreSQL races cover roster stale edits, request side effects, publication/edit/unpublish/republish, TOIL, provisioning and live logon/handover/logoff; tenant-scoped retries verified. | Dedicated PostgreSQL CI must remain mandatory; new transitions require equivalent race tests. |
| PR-009 | Medium | Browser security | XSS, CSRF, open redirects or formula injection could expose workforce information. | Maintain default-deny CSRF, nonce CSP, safe redirects and export escaping; remove inline allowances. | Implemented and locally verified | CSP contains no `unsafe-inline`; inline style attributes are repository-gated; nonce/style/connect/worker/manifest policy tests pass. | Integrity-pinned Bootstrap and Font Awesome CDNs remain; vendoring is recommended. |
| PR-010 | Medium | Worker/Redis resilience | Rate-limit or job-store outage could permit abuse or lose work. | Verify fail-closed authentication limits, bounded retry/backoff, leases, stale recovery, health and privacy-safe keys. | Implemented and locally verified | Refusal/timeout/malformed/intermittent Redis tests; startup PING; namespaced hashed TTL keys; stale lease, exhaustion, safe retry and heartbeat-backed readiness tests. | Managed Redis topology, TLS, persistence and alerting are external controls. |
| PR-011 | Medium | Observability | Missing structured signals delay detection of tenant, queue, database or safety-advisory failures. | Standardise correlation-aware JSON logs, metrics, protected diagnostics and safe readiness checks. | Implemented and locally verified | Safe-field JSON formatter; request/security hooks; bearer-protected Prometheus/internal health; public live/ready; worker queue/age/success signals and tests. | Central collection, provider metrics, alert thresholds and paging tests remain deployment work. |
| PR-012 | Medium | Supply chain | Vulnerable dependencies or images could reach production. | Keep pinned dependencies/actions, dependency audit, CodeQL, secret scan, SBOM and image vulnerability gates. | Implemented locally; exact CI pending | Python 3.12/3.14 matrix, all-tracked Ruff, repository Bandit, pip-audit, CodeQL, gitleaks, image High/Critical gate and CycloneDX upload are configured. | Exact final-commit CI and GitHub security/branch settings require owner verification. |
| PR-013 | Medium | Live-position assurance | Operators may treat advisory timers/warnings as a certified safety control. | Test operational invariants and publish the explicit assurance boundary and required human procedures. | Repository boundary documented | Live-position lifecycle, kiosk restriction, qualification and report tests; `docs/safety/live-position-assurance-boundary.md`. | Independent operational safety assessment and customer procedures are mandatory. |
| PR-014 | Medium | Maintainability | The large legacy module makes security review and isolated testing harder. | Continue bounded extraction in priority order while preserving endpoints and avoiding a rewrite. | In progress | Production operations now joins auth, roster, reports, operations, training and live-position extracted boundaries with endpoint-equivalence tests. | `app.py` remains 10,029 lines and distributed legacy logic remains a defect-introduction risk. |
| PR-015 | Low | Test/runtime hygiene | Resource and upstream deprecation warnings can hide future compatibility defects. | Close test database resources and track Flask-Login/Alembic upstream migrations. | Open | 19 final-suite deprecation warnings recorded above. | No current production failure demonstrated. |

## Release decision rule

Repository changes alone cannot justify “production ready”. Once exact-commit CI
is green, this branch supports a **limited-production ready** repository
classification for a tightly controlled single-airport pilot. Actual deployment
remains blocked on managed infrastructure/grants, external penetration/privacy/
safety review, off-site backup and restore rehearsal, monitoring/paging tests and
customer acceptance recorded in the commercial-pilot checklist.
