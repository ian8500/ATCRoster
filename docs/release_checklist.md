# Production release checklist

Record owner, date and evidence for every item. An unchecked launch gate blocks
production release.

## Automated evidence

- [x] Full test suite passes: 152 passed and 2 intentionally skipped on
  30 July 2026.
- [x] Tenant-isolation and cross-unit IDOR tests pass.
- [x] Super Admin privacy tests pass.
- [x] Account-limit, request, annotation and publication tests pass.
- [x] Fresh control, operational and combined SQLite and PostgreSQL 16
  databases reach Alembic `20260730_28`.
- [x] `pip-audit -r requirements-prod.txt` reports no known vulnerabilities.
- [x] Changed-file and correctness/security lint checks pass; remaining legacy
  style debt is recorded as non-blocking.
- [x] 30-airport/108,000-assignment smoke test passes locally.
- [x] Production container builds, has no fixed HIGH/CRITICAL Trivy findings,
  and production-style PostgreSQL/Redis/private-storage configuration
  validation passes.

## Production controls

- [ ] PostgreSQL is used; SQLite and Flask development server are disabled.
- [x] Per-airport operational database routing is wired and verified in
  SQLite and PostgreSQL integration tests.
- [ ] TLS, Secure/HttpOnly/SameSite cookies and trusted proxy settings are set.
- [ ] Unique Flask, MFA encryption and database secrets are in a secret store.
- [ ] MFA and account recovery have been acceptance tested.
- [ ] Distributed rate limiting, central monitoring and alerting are active.
- [ ] Encrypted backup and isolated restore rehearsal meet approved RPO/RTO.
- [ ] Incident response and operational contingency contacts are current.

## Assurance and acceptance

- [ ] Independent penetration test findings are closed or formally accepted.
- [ ] DPIA, privacy notice, contracts, retention and subject-rights process are
  approved.
- [ ] Accessibility and supported mobile/browser testing are signed off.
- [ ] Unit configuration and imported data receive two-person verification.
- [ ] ATC operational/safety acceptance and go-live authority are recorded.
- [ ] Rollback decision point and on-call coverage are confirmed.
