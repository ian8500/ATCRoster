# Production release checklist

Record owner, date and evidence for every item. An unchecked launch gate blocks
production release.

## Automated evidence

- [ ] Full test suite passes on the release commit.
- [ ] Tenant-isolation and cross-unit IDOR tests pass.
- [ ] Super Admin privacy tests pass.
- [ ] Account-limit, request, annotation and publication tests pass.
- [ ] Clean-database `alembic upgrade head` succeeds.
- [ ] `pip-audit -r requirements.txt` reports no known vulnerabilities.
- [ ] Lint results are reviewed; new correctness/security findings are closed.
- [ ] 30-airport scale smoke test meets the agreed environment threshold.

## Production controls

- [ ] PostgreSQL is used; SQLite and Flask development server are disabled.
- [ ] Per-airport operational database routing is wired and verified.
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

