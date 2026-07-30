# Database migration runbook

The current head is `20260730_28`. Production migration must use
`scripts/migrate_all_databases.py`, which applies the `control` schema role to
the control database and the `operational` role separately to every configured
airport database. It is safe to rerun and does not create control-plane tables
or a `unit` table in an operational database.

Back up the control database and every airport database as one labelled
recovery set. A failed airport migration stops with its unit ID and a
non-sensitive error; correct the secret/database configuration and rerun.
Never include database URLs in tickets or logs.

Interrupted invitation acceptance is handled with:

```bash
flask --app app reconcile-signups
flask --app app reconcile-signups --apply
```

Rollback is backup restoration, not Alembic downgrade. Restore control and
every airport database from the same recovery point before starting the prior
application image.

## Before deployment

1. Freeze schema-changing releases and identify the exact application commit.
2. Back up the production control and every operational database; record
   checksum, size and current Alembic revision.
3. Restore the backup into an isolated environment.
4. Install the candidate dependency lock and run
   `python scripts/migrate_all_databases.py` against the restored control and
   operational databases.
5. Run the full automated suite and tenant-isolation acceptance tests.
6. Review migration SQL, expected locks, duration and storage headroom.

## Deployment

1. Put writes into the approved maintenance state.
2. Deploy one schema-compatible application image.
3. Run `python scripts/migrate_all_databases.py` once. It upgrades control
   first, then every route in unit order and fails closed for missing,
   malformed or control-equal secrets.
4. Confirm `alembic current` reports the exact approved release head
   (`20260730_28` for this release).
5. Check `/health/live` and `/health/ready`.
6. Smoke-test login, one unit-scoped roster, requests, annotations, publication
   history and Super Admin aggregate view.
7. Re-enable writes and monitor errors, latency and connection use.

## Failure and rollback

Do not improvise a destructive Alembic downgrade for tenant-boundary changes.
Stop writes, preserve logs, and choose either a schema-compatible application
rollback or restoration of the pre-deployment backup. Verify record counts,
tenant isolation and publication history before returning service. Record the
decision, operator, timestamps and recovery result.

The legacy import command must be rerunnable, compare before/after counts and
retain its checkpoint. Never import production data into a test service.
