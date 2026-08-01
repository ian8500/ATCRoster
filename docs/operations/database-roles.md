# Production PostgreSQL roles

ATCRoster production databases require separate credentials for schema changes
and application traffic.

## Roles

1. **Migration owner** — owns schema objects and is available only to the
   release/migration process. Web and worker processes must not receive this
   credential.
2. **Runtime application role** — used by web and workers. It receives schema
   `USAGE`, sequence `USAGE/SELECT`, and ordinary table
   `SELECT/INSERT/UPDATE/DELETE`. Audit tables receive only `SELECT/INSERT`.
   It receives no schema `CREATE` and no audit `UPDATE/DELETE/TRUNCATE`.
3. **Audit reader** (optional) — receives only schema `USAGE` and `SELECT` on
   existing audit tables.

Provider administrators and the database owner remain technically able to
alter evidence. Their accounts require provider MFA, access logging and an
approved break-glass procedure; this is outside application enforcement.

## Controlled deployment

Railway web and worker services contain only runtime database URLs. Migration
owner URLs are stored as protected GitHub environment secrets named
`CONTROL_DATABASE_OWNER_URL` and `OPERATIONAL_DATABASE_OWNER_URL`. The
controlled-promotion workflow migrates with those credentials, reapplies and
verifies runtime grants, and only then deploys the application services.

Do not restore Railway's per-service pre-deploy migration command: Railway
runs it in the application service environment and that would require exposing
owner credentials to web and worker processes.

## Manual application after a migration

Create the roles and passwords through the managed PostgreSQL control plane.
Do not store them in the repository. Set the migration-owner database URL in a
named environment variable, then run:

```text
ATCROSTER_RUNTIME_DATABASE_ROLE=<runtime-role>
ATCROSTER_AUDIT_READ_ROLE=<optional-reader-role>
python -m scripts.apply_runtime_database_grants \
  --database-url-env CONTROL_DATABASE_OWNER_URL
python -m scripts.apply_runtime_database_grants \
  --database-url-env ATCROSTER_UNIT_1_DATABASE_OWNER_URL
```

Use `--dry-run` to validate roles, connectivity and generated statements
without changing grants. Repeat for every routed airport database.

## Release verification

```text
python -m scripts.verify_runtime_database_grants \
  --database-url-env CONTROL_DATABASE_OWNER_URL
python -m scripts.verify_runtime_database_grants \
  --database-url-env ATCROSTER_UNIT_1_DATABASE_OWNER_URL
```

Verification fails on missing ordinary privileges, missing sequence access,
schema creation rights, or any audit mutation/truncation right. It must be a
post-migration deployment gate. A runtime credential must never run Alembic.

Managed providers may restrict role creation or database-level grants. In that
case, use the provider's owner/admin connection for the apply step or request
provider support. Do not fall back to using the owner credential in web or
worker services; record the pilot as blocked until separation is available.

## Tests

The PostgreSQL integration suite creates a real login role and proves:

- ordinary staff insert/update succeeds;
- audit insert/select succeeds;
- audit update/delete fails with `InsufficientPrivilege`;
- schema creation (representing migration authority) fails;
- programmatic privilege verification passes.
