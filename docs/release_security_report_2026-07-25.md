# Release security report — 25 July 2026

## Decision

The application passes its automated tenant-isolation and Super Admin privacy
tests. It is suitable for controlled acceptance testing, but **not yet approved
for unrestricted production launch**. The required independent penetration
test, production backup/restore rehearsal and aviation/data-protection
acceptance remain external release gates.

Operational Flask-SQLAlchemy mappers now use a tenant-routed session backed by
the authenticated membership and the control-plane secret-name route.
Platform-control contexts are denied an operational bind. SQLite and
PostgreSQL integration tests prove that two airports use different physical
databases and cannot read the other airport's requests. The central membership
stores an opaque operational person ID without an impossible cross-database
foreign key.

## Controls verified

- Authenticated unit context is server-derived; tenant IDs and database names
  are not accepted from forms or query strings.
- Cross-unit request and annotation IDOR attempts are rejected.
- Tenant-keyed caches prevent cross-unit shift, settings, annotation and roster
  data reuse.
- Super Admin pages expose account/service aggregates and do not render
  personnel, roster, leave, sickness, qualification or request details.
- Active-login limits are enforced transactionally for invitation, activation
  and restoration paths.
- Secure invitations expire, are single-use and recheck capacity at acceptance.
- Authenticated mutations are default-deny CSRF protected; login is rate
  limited and safe redirects are used.
- Request transitions and annotation changes are validated and audited.
- Published rosters are immutable snapshots; rollback creates a new version.
- Passwords are hashed; MFA secrets are encrypted when the production field
  key is configured; session cookies can be Secure/HttpOnly/SameSite.
- `pip-audit` reports no known vulnerabilities for the pinned dependency set.
- The Alembic chain upgrades clean and legacy fixtures through revision
  `20260725_08`.

## Verification evidence

- Full automated suite: 73 passed and one PostgreSQL-only test skipped when
  its three integration database URLs are not supplied.
- PostgreSQL 16 three-database integration: 1 passed, covering independent
  control/operational migrations, disjoint schemas and authenticated airport
  read isolation.
- PostgreSQL 16: control plus two physically separate airport databases
  migrated and passed authenticated cross-database isolation.
- Production container image built successfully and its production
  configuration validation passed.
- Dependency audit: `pip-audit -r requirements.txt`.
- Scale smoke test: 30 fictitious airports, 40 people per airport and 90 roster
  days (108,000 assignments); seed 0.103 seconds and scoped month query 0.886
  milliseconds on the local SQLite smoke environment. These figures are
  regression evidence, not a production service-level guarantee.
- Linting identified pre-existing maintainability findings in the monolithic
  `app.py`. No lint finding is being represented as resolved merely because the
  runtime tests pass.

## Residual risk and launch gates

1. Commission an independent authenticated penetration test, including IDOR,
   privilege escalation, invitation/MFA recovery and export testing.
2. Rehearse an encrypted PostgreSQL restore and record measured RPO/RTO.
3. Complete DPIA, retention schedule, processor agreements and UK GDPR review.
4. Complete accessibility testing with keyboard, screen reader and mobile
   devices against the agreed WCAG target.
5. Obtain accountable ATC operational acceptance; ATCRoster is decision
   support and does not replace local safety management or contingency plans.

## Non-blocking improvements

- Split the monolithic application into bounded services and repositories.
- Make lint checks incremental and reduce existing broad exception handlers.
- Add Redis-backed distributed rate limiting for multi-instance deployments.
- Add email delivery and notification preference verification.
- Automate PostgreSQL load testing with production-like concurrency.
- Implement and formally test the proposed read-only auditor contract.
