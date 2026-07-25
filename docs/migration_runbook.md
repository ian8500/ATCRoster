# Database migration runbook

## Before deployment

1. Freeze schema-changing releases and identify the exact application commit.
2. Back up the production control and every operational database; record
   checksum, size and current Alembic revision.
3. Restore the backup into an isolated environment.
4. Install the candidate dependency lock and run `alembic upgrade head`.
5. Run the full automated suite and tenant-isolation acceptance tests.
6. Review migration SQL, expected locks, duration and storage headroom.

## Deployment

1. Put writes into the approved maintenance state.
2. Deploy one schema-compatible application image.
3. Run `alembic upgrade head` exactly once per database.
4. Confirm `alembic current` reports `20260725_05` (or the later approved
   release revision).
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

