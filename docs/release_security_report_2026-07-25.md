# Release security report — 25 July 2026

## Decision

The application passes its automated tenant-isolation and Super Admin privacy
tests. It is suitable for controlled acceptance testing, but **not yet approved
for unrestricted production launch**. The required independent penetration
test, production backup/restore rehearsal and aviation/data-protection
acceptance remain external release gates.

The target architecture also calls for a separate operational database per
airport. A secret-backed, authenticated-unit database router is implemented
and tested, but the Flask-SQLAlchemy operational repositories still use the
shared database with mandatory `unit_id` scoping. Physical database separation
must be wired into the production session layer, migrated and exercised before
claiming that architecture.

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
- The Alembic chain upgrades a clean database through revision `20260725_05`.

## Verification evidence

- Full automated suite: 46 passed on Python 3.12 (14 non-failing legacy
  SQLAlchemy/test-fixture warnings).
- Dependency audit: `pip-audit -r requirements.txt`.
- Scale smoke test: 30 fictitious airports, 40 people per airport and 90 roster
  days (108,000 assignments); seed 0.103 seconds and scoped month query 0.886
  milliseconds on the local SQLite smoke environment. These figures are
  regression evidence, not a production service-level guarantee.
- Linting identified pre-existing maintainability findings in the monolithic
  `app.py`. No lint finding is being represented as resolved merely because the
  runtime tests pass.

## Residual risk and launch gates

1. Integrate the operational database router with ORM session selection and
   migrate each airport to its own operational database.
2. Commission an independent authenticated penetration test, including IDOR,
   privilege escalation, invitation/MFA recovery and export testing.
3. Rehearse an encrypted PostgreSQL restore and record measured RPO/RTO.
4. Complete DPIA, retention schedule, processor agreements and UK GDPR review.
5. Complete accessibility testing with keyboard, screen reader and mobile
   devices against the agreed WCAG target.
6. Obtain accountable ATC operational acceptance; ATCRoster is decision
   support and does not replace local safety management or contingency plans.

## Non-blocking improvements

- Split the monolithic application into bounded services and repositories.
- Make lint checks incremental and reduce existing broad exception handlers.
- Add Redis-backed distributed rate limiting for multi-instance deployments.
- Add email delivery and notification preference verification.
- Automate PostgreSQL load testing with production-like concurrency.
- Implement and formally test the proposed read-only auditor contract.
