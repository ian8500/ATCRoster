# Observability operations

ATCRoster emits JSON logs in production. Each record includes an ISO-8601 UTC
timestamp, severity, service, deployment environment and `ATCROSTER_COMMIT_SHA`.
Completed request records add the request ID, endpoint name, non-personal numeric
unit/actor identifiers, outcome, status and duration. Security-event fields are
allowlisted; passwords, cookies, tokens, MFA material, medical narrative and
uploaded content must never be passed to the logging interface.

## Endpoints

- `/health/live` proves only that the web process can serve a request.
- `/health/ready` verifies the control database and schema, and verifies Redis in
  production. It returns only `ready` or `not_ready`.
- `/internal/health` provides limited database/Redis diagnostic state.
- `/internal/metrics` serves Prometheus text for request count/latency/errors,
  active requests, readiness, login/rate-limit/Redis events and worker signals.
- `/platform/worker-health` reports queue depth, oldest queued age, active/stale
  workers and the last successful provisioning job to a platform administrator.

The two `/internal/*` endpoints require either a signed-in platform administrator
or `Authorization: Bearer $ATCROSTER_INTERNAL_METRICS_TOKEN`. Production startup
requires that token to contain at least 32 characters. Rotate it by updating the
secret on every web instance, rolling the deployment, then updating the monitoring
collector. Never place it in a URL.

Metrics are intentionally low-cardinality and per process. A commercial deployment
must scrape every web instance and aggregate centrally. Alerts should cover: no
ready web instance, repeated readiness/Redis failures, elevated 5xx responses,
login/rate-limit anomalies, no worker heartbeat, non-zero stale workers, growing
queue depth, and an oldest queued job beyond the agreed service objective.

Provider alerts for database pool exhaustion, object storage, backup age/result,
CPU/memory and external availability remain deployment responsibilities. Test
notification routing and paging escalation before pilot release.
