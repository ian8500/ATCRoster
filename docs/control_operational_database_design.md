# Control-plane and operational database design

## Security boundary

The control database contains airport metadata, plan limits, global login
identities, memberships, secret references, invitations, provisioning and
signup workflow state, platform MFA credentials, and privacy-safe platform
audits. It must not contain airport personnel, qualifications, rosters,
requests, annotations, medical/licence dates, or airport audit detail.

Every airport has a separate operational database containing staff, shifts,
qualifications, rosters, assignments, requests, annotations, rules and
airport-level audit events. Operational tables retain `unit_id` as a
defence-in-depth ownership key but do not reference a control-plane `unit`
table. ORM routing is derived from the authenticated server-side membership;
a browser-supplied airport identifier is never a routing authority.

Migrations share an audited revision sequence but execute through distinct
roles:

```bash
ATCROSTER_SCHEMA_ROLE=control python -m alembic upgrade head
ATCROSTER_SCHEMA_ROLE=operational python -m alembic upgrade head
```

`scripts/migrate_all_databases.py` applies the control role first, reads only
secret references, then applies the operational role to each airport. It
prints unit IDs and revision results but never URLs or secret values.

## Platform MFA

`PlatformMfaCredential` is control-plane only. TOTP seeds are encrypted with
`ATCROSTER_FIELD_ENCRYPTION_KEY`; recovery codes are stored as SHA-256 digests
and removed after one use. Password success creates only a pre-authentication
session. A fresh authenticated session is created after MFA verification.
Login, enrolment, verification, recovery and reset create non-sensitive
central audit events and are rate limited.

Reset a credential only through a trusted administrative shell:

```bash
flask --app app reset-platform-mfa --username PLATFORM_LOGIN
```

## Airport provisioning

An airport progresses through `pending`, `database_configured`,
`migrations_complete`, `invitation_issued`, `active`, or `failed`.
Creation stores only non-personal metadata and a deployment-secret name.
Provisioning validates that reference, connects, migrates, checks the required
operational schema, and only then creates the one-time UnitAdmin invitation.
The airport becomes active only when that invitation completes.

Retries are idempotent. A failure stores an allow-listed error code and
privacy-safe audit event; connection strings and exception text are not
displayed or logged.

## Signup saga and recovery

Invitation acceptance is a durable saga:

1. `pending`
2. `identity_created` in control
3. `operational_account_created` with access disabled
4. `membership_created`
5. `completed`, enabling the account and consuming the invitation

Every invitation has one stable idempotency key. A retry resumes the last
durable stage and deterministic staff marker, so it cannot duplicate an
identity, person or membership. Intermediate identities have no membership;
intermediate staff accounts have `membership_status=pending`.

Inspect and reconcile interrupted work:

```bash
flask --app app reconcile-signups
flask --app app reconcile-signups --apply
```

## Migration and rollback

1. Take and verify independent encrypted backups of the existing database.
2. Put the application in maintenance mode.
3. Configure the control URL and every airport secret reference.
4. Run the migration tool and retain its per-database result.
5. Run readiness checks and acceptance tests before reopening.

Revision `20260726_09` and the physical split are forward-only security
boundaries. Rollback means stopping the new application and restoring the
matching control and operational backups as one recovery set. Do not run a
schema downgrade or copy operational personal data into control.
