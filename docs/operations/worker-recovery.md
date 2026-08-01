# Provisioning worker recovery

Provisioning jobs are database-backed and move through `queued`, `running`,
`retry_wait`, `completed`, `failed` or `cancelled`. A worker claims one row with
`FOR UPDATE SKIP LOCKED`, records its lease owner/expiry and renews the lease while
processing. Airport-scoped PostgreSQL advisory locks prevent two provisioning
operations for one airport. Active-job uniqueness prevents duplicate queue rows.

Retryable failures use exponential backoff from 15 seconds, capped at 900 seconds,
and become terminal after five attempts. Operator-visible failures use fixed error
codes, not exception text. An expired running lease is moved to `retry_wait` with
`worker_interrupted`. Retrying a terminal job creates a new uniquely keyed job;
completed bootstrap and migration work remains idempotent.

Worker `/health/live` checks the child process. `/health/ready` additionally queries
the control database for a recent provisioning heartbeat; a process alone is not
ready. Platform `/platform/worker-health` shows active/stale workers, queue depth,
oldest queued age and last successful completion.

Recovery procedure:

1. Confirm database and Redis availability and capture the safe error code.
2. Check heartbeat age and queue state; do not edit job rows directly.
3. Restart the worker. Startup recovers expired leases automatically.
4. Correct the external dependency or secret, then use **Provision / retry** for
   `retry_wait` work or create a fresh operator retry after terminal failure.
5. Confirm one completed job, one bootstrap invitation and healthy routing.
6. Escalate if the same safe error reaches five attempts or queue age breaches the
   pilot service objective.

Redis loss while storing a one-time bootstrap token causes the job transaction to
roll back and retry; the token is never logged or persisted in plaintext. Database
owner intervention and deletion of job history are not normal recovery actions.
