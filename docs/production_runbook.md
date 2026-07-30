# ATCRoster production runbook

This runbook covers a self-hosted production deployment. Formal aviation
acceptance, data-protection approval and security certification remain the
operator's responsibility.

Airport creation is intentionally two-phase. Create non-personal metadata,
configure the named database secret, then use **Provision / retry**. The web
process queues the work; the worker migrates and checks the operational
database before issuing a bootstrap invitation. Its raw token is held in
Redis for one hour and can be displayed exactly once. If it is lost, revoke
and deliberately replace it. Do not send an invitation while provisioning is
`pending`, `retry_wait` or `failed`.

Platform SuperAdmins must enrol application-verified MFA. Their encrypted
credential and one-time recovery-code hashes live only in control. Use
`flask --app app reset-platform-mfa` through the trusted operator shell for
recovery; never disable the MFA requirement.

## Required services

- PostgreSQL 16 or a supported managed PostgreSQL service
- Redis 7 or a supported managed Redis service
- A continuously running `python scripts/run_provisioning_worker.py` process
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
   Control and airport URLs must be different. Additional airports use the
   exact secret name stored in control, for example
   `ATCROSTER_UNIT_2_DATABASE_URL`; never pass URLs through a web form.
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
docker compose up -d web worker
```

### Railway

The repository includes `railway.toml` for a managed Railway deployment.
Provision a control PostgreSQL service, a separate PostgreSQL service per
airport, Redis, a web service and a worker service. Both application services
receive the same encrypted variables:

- `ATCROSTER_ENVIRONMENT=production`
- `ATCROSTER_SECURE_COOKIES=true`
- `FLASK_SECRET_KEY` with at least 32 random characters
- `ATCROSTER_FIELD_ENCRYPTION_KEYS` as an ordered versioned key ring, for
  example `v2:<new-fernet-key>,v1:<old-fernet-key>`
- `ATCROSTER_TOKEN_ENCRYPTION_KEYS` as a separate ordered key ring for
  temporary bootstrap-token envelopes
- `ATCROSTER_BOOTSTRAP_TOKEN_TTL_SECONDS` (default 900)
- `DATABASE_URL` using the PostgreSQL service's private connection variables
- `CONTROL_DATABASE_URL` using the control PostgreSQL service
- `ATCROSTER_UNIT_<id>_DATABASE_URL` for every airport database
- `REDIS_URL`
- `ATCROSTER_SESSION_IDLE_MINUTES` and
  `ATCROSTER_SESSION_ABSOLUTE_MINUTES`
- `ATCROSTER_TRUSTED_PROXY_HOPS` matching the verified proxy topology
- `ATCROSTER_TRUSTED_HOSTS` as a comma-separated allowlist
- `ATCROSTER_PROVISIONING_LEASE_SECONDS` (minimum 30; default 120)
- `ATCROSTER_DB_CONNECT_TIMEOUT_SECONDS`,
  `ATCROSTER_DB_STATEMENT_TIMEOUT_MS`, `ATCROSTER_DB_POOL_TIMEOUT_SECONDS`,
  `ATCROSTER_OPERATIONAL_POOL_SIZE` and
  `ATCROSTER_OPERATIONAL_MAX_OVERFLOW`
- controlled briefing storage using either complete private S3-compatible
  `BRIEFING_STORAGE_PROVIDER=s3`, bucket, endpoint, access-key and secret-key
  configuration, or `BRIEFING_STORAGE_PROVIDER=mounted` with
  `ATCROSTER_BRIEFING_DURABLE_DIR` set to the absolute path of an explicitly
  provisioned durable volume

The web service uses the repository `railway.toml`. Configure the worker
service start command as `python scripts/run_provisioning_worker.py` and no
public endpoint. The pre-deploy command upgrades control first and then all
configured operational databases. After the first healthy deployment, run
`flask --app app bootstrap-platform` once with a unique administrator
password. Do not upload acceptance data or local environment files.
Confirm storage credentials are private, public object/container access is
disabled, and an uploaded briefing remains retrievable after a service
restart. Startup rejects incomplete S3 configuration and implicit local
instance storage, but cannot prove that an operator-supplied directory is
backed by a durable mount.

For field-key rotation, prepend the new version, retain previous decrypt keys,
take verified backups, and run:

```bash
flask --app app rotate-field-encryption \
  --confirm ROTATE-FIELD-ENCRYPTION
```

Retire old field keys only after verification. Token-envelope key versions
must overlap for at least the configured bootstrap-token TTL.

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
