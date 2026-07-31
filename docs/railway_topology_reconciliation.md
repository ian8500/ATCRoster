# Railway topology reconciliation

Evidence date: 31 July 2026

## Current state

Reconciliation was completed on 31 July 2026. The Railway project
`ATCRoster Production` now has two correctly named, physically isolated
environments.

### `production`

It contains:

- `web`, serving `www.atcroster.com` and its Railway service domain;
- control PostgreSQL;
- `Postgres-IWLD`, the additional operational PostgreSQL service;
- `redis`;
- `worker`;
- the private `briefing-documents` bucket;
- configuration names for control and two operational databases, Redis,
  versioned encryption keys, trusted hosts/proxies, private briefing storage,
  SMTP and optional SMS.

### `staging`

It contains independent `staging-web`, `staging-worker`,
`staging-control-db`, `staging-airport-db`, `staging-redis` and
`staging-briefing-documents` resources. It uses fresh secrets and synthetic
acceptance data only. `pilot.atcroster.com` points to this environment.

The obsolete two-service environment was backed up and deleted. Production
data, variables and storage were not copied into staging.

## Continuing safety rules

Do not:

- copy production secret values into source control, logs or tickets;
- copy personal or operational production data into staging;
- attach `www.atcroster.com` to staging;
- share the locally retained synthetic staging credential;
- promote a staging release until database, Redis, worker, storage and
  authenticated smoke tests pass.

## Reconciliation procedure

This procedure is intentionally separate from an application-code pull
request.

### 1. Establish ownership and backups

1. Record the accountable owner, change window and rollback owner.
2. Confirm which PostgreSQL services contain live control and airport data.
3. Take encrypted custom-format backups of control and every airport database.
4. Record checksums, Alembic revisions and backup object references.
5. Verify the private briefing bucket's versioning and recovery mechanism.
6. Retain the currently running web and worker image digests.

### 2. Verify the public stack

Against the current `pilot-staging` environment:

1. Confirm every required configuration **name** from
   `docs/production_runbook.md`; do not export values.
2. Confirm control and operational database URLs are distinct.
3. Confirm Redis health and a current worker heartbeat.
4. Confirm the private bucket denies public access.
5. Run the migration command in report/compatibility mode against restored
   copies before changing the live databases.
6. Run liveness, readiness, login, MFA, tenant-routing, upload/download and
   provisioning smoke tests.
7. Confirm `ATCROSTER_ENVIRONMENT=production` and secure-cookie/trusted-host
   controls on both web and worker services.

### 3. Reconcile names without moving data

The lowest-risk target is to rename environments, not copy live databases:

1. Rename the old `production` environment to
   `legacy-production-20260724`.
2. Rename `pilot-staging` to `production`.
3. Verify that service IDs, database volumes, bucket, domains and active
   deployments did not change.
4. Verify Railway-provided environment metadata changed as expected while
   explicit application security variables remained unchanged.
5. Re-run public and authenticated smoke tests.

The exact Railway commands must be generated from fresh environment IDs at
the change window. Do not embed IDs in this document.

### 4. Create a real staging environment

Create a new `staging` environment from infrastructure definitions, not from
live data:

1. provision empty control and operational PostgreSQL services;
2. provision independent Redis, worker and private object storage;
3. create staging-only secrets and encryption keys;
4. migrate clean databases;
5. seed synthetic acceptance data only;
6. attach `pilot.atcroster.com` after validation;
7. leave `www.atcroster.com` attached only to production.

Update `.github/workflows/staging-health.yml` to the new staging service
domain after it is healthy.

### 5. Retire the legacy stack

Only after the retention period and witnessed restore:

1. confirm the legacy PostgreSQL service is not authoritative;
2. archive required logs and backup evidence;
3. remove public access;
4. stop the legacy web service;
5. delete it only through the approved retention/change process.

## Rollback

If an environment rename or domain verification fails:

1. restore the previous environment names;
2. keep `www.atcroster.com` on the previously healthy service;
3. redeploy the retained image digest only if the running deployment changed;
4. do not restore a database unless a write or migration changed it;
5. record the failure and request IDs before retrying.

Environment renaming alone must not run migrations or change application
variables. Any operation that would do so is a separate release and requires
fresh backup evidence.

## Completion evidence

Completed on 31 July 2026:

- Railway has one clearly named production environment containing web,
  worker, Redis, control PostgreSQL, per-airport PostgreSQL and private object
  storage;
- `www.atcroster.com` points only to that environment;
- a separate staging environment uses synthetic data and independent secrets;
- `pilot.atcroster.com` points only to staging;
- the staging liveness, readiness and authenticated tenant-routing smoke tests
  passed;
- the production readiness check remained healthy throughout;
- the obsolete database was exported in PostgreSQL custom format before its
  environment was deleted;
- the scheduled staging health workflow targets `pilot.atcroster.com`.
