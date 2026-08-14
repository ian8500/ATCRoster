# Release security report — updated 14 August 2026

## Decision

The application passes its automated tenant-isolation, MFA, roster-concurrency,
Position Monitor and Super Admin privacy tests. The release was deployed through
staging to Railway production on 14 August 2026. It is suitable for controlled
production use, subject to the external assurance gates below; this report does
not claim unrestricted operational or regulatory approval.

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
  key is configured; session cookies can be Secure/HttpOnly/SameSite. MFA reset
  revokes prior credentials/sessions and forces re-enrolment without recovery
  codes.
- `pip-audit` reports no known vulnerabilities for the pinned dependency set.
- The Alembic chain upgrades clean and legacy fixtures through head
  `20260813_58`.

## Verification evidence

- Full automated suite: 440 passed, 11 environment-dependent integration skips,
  78.60% coverage.
- PostgreSQL control plus two physically separate airport databases, Redis and
  generated backup/restore integration: 11 passed.
- Production container image and seeded Playwright browser suite: passed.
- Production container image built successfully and its production
  configuration validation passed.
- Dependency audit: `pip-audit -r requirements.txt`.
- Scale smoke test: 30 fictitious airports, 40 people per airport and 90 roster
  days (108,000 assignments); seed 0.103 seconds and scoped month query 0.886
  milliseconds on the local SQLite smoke environment. These figures are
  regression evidence, not a production service-level guarantee.
- Root `app.py` is a small compatibility/WSGI import surface. Application
  composition is in `atcroster/application.py`, with domain-owned packages;
  no lint finding is represented as resolved merely because runtime tests pass.

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

- Continue reducing coupling in the composition root only where a clear domain
  owner exists; do not reintroduce a monolithic compatibility entrypoint.
- Reduce broad exception handlers only with a demonstrated recovery path.
- Add Redis-backed distributed rate limiting for multi-instance deployments.
- Add email delivery and notification preference verification.
- Automate PostgreSQL load testing with production-like concurrency.
- Implement and formally test the proposed read-only auditor contract.
