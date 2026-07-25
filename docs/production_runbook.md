# ATCRoster production runbook

This runbook covers a self-hosted production deployment. Formal aviation
acceptance, data-protection approval and security certification remain the
operator's responsibility.

## Required services

- PostgreSQL 16 or a supported managed PostgreSQL service
- TLS reverse proxy or load balancer
- Managed secret store
- Encrypted backup destination outside the application host
- Central logging and alerting
- Organisation identity/MFA control appropriate to the exposure

SQLite and the Flask development server are prohibited in production.

## First deployment

1. Copy `.env.example` to a secure deployment-secret mechanism. Do not commit
   the resulting values.
2. Generate unique database and Flask secrets.
   Generate the MFA field-encryption key with:

   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
3. Build and inspect the container image.
4. Provide `CONTROL_DATABASE_URL`, `DATABASE_URL` and every secret referenced
   by `database_routing_metadata`, then run
   `python scripts/migrate_all_databases.py`.
5. Run:

   ```bash
   flask --app app bootstrap-platform
   ```

6. Sign in as the Super Admin, create the first airport, and transfer its
   one-time bootstrap invitation through the approved secure channel. The
   airport administrator chooses credentials and enrols MFA.
7. Configure the airport through **Onboarding** and **Operations**.
8. Verify `/health/live` and `/health/ready`.
9. Complete the acceptance plan in `docs/pilot_readiness.md`.

With Docker Compose:

```bash
docker compose build
docker compose run --rm migrate
docker compose run --rm web flask --app app bootstrap-platform
docker compose up -d web
```

### Railway

The repository includes `railway.toml` for a managed Railway deployment.
Provision one PostgreSQL service and one application service, then set these
application variables through Railway's encrypted variable store:

- `ATCROSTER_ENVIRONMENT=production`
- `ATCROSTER_SECURE_COOKIES=true`
- `FLASK_SECRET_KEY` with at least 32 random characters
- `ATCROSTER_FIELD_ENCRYPTION_KEY` generated with Fernet
- `DATABASE_URL` using the PostgreSQL service's private connection variables

The deployment runs Alembic before starting, serves through Waitress on port
8080 and uses `/health/ready` as its health check. After the first healthy
deployment, run `flask --app app bootstrap-platform` once with a unique
administrator password. Do not upload acceptance data or local environment
files.

## Reverse proxy

Terminate TLS at the proxy, redirect HTTP to HTTPS, preserve the original host
and scheme, and pass a unique `X-Request-ID`. Apply request and authentication
rate limits at the proxy. Only the proxy should reach the application port.

## Monitoring

- Poll `/health/live` for process liveness.
- Poll `/health/ready` for database/schema readiness.
- Alert on repeated HTTP 500 responses, failed logins, readiness failures,
  database saturation and backup failures.
- Retain the `X-Request-ID` across proxy, application and database audit
  records.
- Never place fatigue-report summaries, medical information, passwords,
  calendar tokens or roster comments in central logs.

## Backup

At least daily:

1. Run an encrypted PostgreSQL custom-format backup.
2. Record timestamp, database identifier, migration revision, size and
   checksum.
3. Copy it to access-controlled off-host storage.
4. Retain according to the approved retention schedule.

Example:

```bash
pg_dump --format=custom --no-owner --file=atcroster.dump "$DATABASE_URL"
sha256sum atcroster.dump > atcroster.dump.sha256
```

## Restore test

1. Provision an isolated PostgreSQL instance.
2. Verify the stored checksum.
3. Restore with `pg_restore --clean --if-exists`.
4. Run `python scripts/migrate_all_databases.py`.
5. Check `/health/ready`.
6. Run tenant-isolation and publication-history acceptance tests.
7. Record recovery time and achieved recovery point.
8. Destroy the isolated environment securely.

## Incident response

For a suspected safety-critical write failure, do not blindly repeat the
action. Use the displayed request ID to determine whether the first operation
committed. If roster integrity is uncertain, freeze publication, export the
current evidence, switch to the approved contingency process and escalate to
the operational manager.

For a suspected data breach, restrict access, preserve logs, rotate affected
secrets, assess notification obligations and follow the organisation's
incident-response and data-breach procedures.

## Upgrade and rollback

1. Back up and test the candidate migration on a restored copy.
2. Deploy the application image and run Alembic once.
3. Verify readiness and execute smoke tests.
4. Do not downgrade tenant-boundary migrations.
5. Application rollback must use a schema-compatible image. Database rollback
   requires the approved restore process, not ad-hoc SQL.
