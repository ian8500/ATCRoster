# Deployment profiles

## Local development

Use the default local SQLite database and in-process limiter only for development.
It does not demonstrate PostgreSQL locking, Redis failure behaviour, database
grants, durable storage, backups or availability.

## Single-node production-like pilot

`docker-compose.yml` provides split control/airport PostgreSQL databases, Redis,
one web process, one worker and a migration job. It is useful for acceptance and
recovery rehearsal, but one host is not high availability. Terminate TLS at a
trusted reverse proxy, keep proxy-hop configuration exact, and store briefing
documents on a private mounted volume or S3-compatible service.

## Recommended commercial deployment

Use managed PostgreSQL for the control and each airport database, managed Redis,
private object storage, at least two independently scheduled web instances and at
least one separately monitored worker. Use TLS termination, rolling replacement,
central JSON logs, per-instance metrics scraping, encrypted off-site backups and
provider alerts. Migration and runtime database roles must be separate. Do not run
migrations concurrently with ordinary runtime traffic without the release runbook.

Waitress remains the supported WSGI server for the pilot. A web replica defaults to
four threads (`ATCROSTER_WAITRESS_THREADS`, bounded to 1–16), a 60-second idle
channel timeout, periodic cleanup and a bounded connection limit. This keeps
per-replica concurrent database pressure modest while allowing an explicit,
capacity-tested increase. Flask rejects request bodies above 2 MiB by default. The reverse proxy should
use a request timeout below its own upstream timeout and permit at least the web
graceful-shutdown window. `ProxyFix` is enabled only by the explicit trusted-hop
count. Application correctness uses PostgreSQL/Redis/object storage, not process
memory, so multiple web instances are supported; metrics remain per instance.

## Secrets and rotation

Production requires Flask, field-encryption, token-encryption and internal-metrics
secrets; control/operational database URLs; Redis; trusted hosts/proxy hops; secure
cookies; and complete private storage credentials. Startup fails closed for missing
required values. Rotate one class at a time, retain prior versioned encryption keys
while data/token lifetime requires them, roll all instances, verify readiness and
then remove retired keys. Database and Redis credential rotation must update every
web/worker/migration service. Rotate the metrics bearer token in collectors after
the new application secret is live. Never commit or print secret values.

Railway services must set `ATCROSTER_PROCESS_TYPE=web` or `worker` explicitly and
must set `ATCROSTER_COMMIT_SHA` to the promoted commit for log correlation.
