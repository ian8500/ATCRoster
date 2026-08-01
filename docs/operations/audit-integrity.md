# Audit integrity operations

ATCRoster treats SMS, request, annotation, change, platform-security,
SuperAdmin, briefing and live-position audit rows as append-only evidence.
The ORM rejects normal application attempts to modify or delete a persisted
audit row. Business changes and their audit records must be committed in the
same database transaction; a failed audit insert rolls the business change
back.

This application control does not protect against a compromised database owner,
raw SQL executed with elevated credentials, provider administrators or backup
tampering. Production must therefore use separate PostgreSQL roles:

- the migration owner may create/alter schema but is not used by web/worker;
- the runtime role receives `SELECT` and `INSERT` on audit tables, but no
  `UPDATE`, `DELETE` or `TRUNCATE` privileges;
- support/analytics roles receive only the minimum approved read access;
- database-owner and provider access is MFA-protected, logged and reviewed;
- audit/security logs are exported to access-controlled off-database storage.

Apply grants after migrations and verify them as a release check because a
table recreated by a migration may receive default owner privileges. Never put
passwords, raw reset/invitation tokens, MFA secrets, session cookies, encryption
keys or unnecessary health details in audit summaries.

Retention and authorised erasure require a separately approved administrative
procedure using a dedicated role, with exported evidence and legal/privacy
approval. They must not be implemented as an ordinary application endpoint.
