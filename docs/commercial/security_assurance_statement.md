# ATCRoster security and assurance summary

ATCRoster uses airport-scoped identities/memberships, role-based access and
separate operational database routing to prevent one airport from selecting or
reading another airport's roster data. Platform administration is deliberately
limited to account metadata and health information.

Current implemented controls include secure password hashing, TOTP MFA,
encrypted MFA/bootstrap secrets, secure session configuration, CSRF protection,
rate limiting, tenant-aware database access, audit trails, controlled
publication/request transitions, migration checks and automated unit,
PostgreSQL, Redis, container, dependency, static-analysis and secret-scanning
tests.

Production operation requires TLS, managed PostgreSQL/Redis, a secrets manager,
central monitoring, independent encrypted backups, restore rehearsal, incident
response, access review and controlled release/rollback. Customer data roles,
retention and subprocessors are documented in the privacy/DPA pack.

Fatigue, staffing and qualification findings are decision support. ATCRoster
does not autonomously certify a roster, replace competent management or replace
the customer's safety-management and contingency arrangements.

Evidence available under appropriate confidentiality includes architecture and
data-classification summaries, permission matrix/test report, release security
report, DPIA/DPA, backup/restore record, migration/runbooks and current
independent testing when commissioned.

Outstanding before broad commercial assurance: independent penetration test,
off-project production backup/restore evidence, confirmed provider transfer
records, approved legal contracts and customer-specific operational acceptance.
