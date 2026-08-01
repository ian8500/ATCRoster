# Redis failure behaviour

Redis is mandatory in production because authentication and account-lifecycle
rate limits must be shared across every web instance.

## Chosen safety behaviour

- Production startup verifies Redis with a two-second connection/read timeout.
  A refusal, timeout or unexpected response prevents the web process becoming
  ready.
- During a request, connection errors, timeouts and malformed pipeline results
  become `LimiterUnavailable`. Security-sensitive rate limits fail closed with
  HTTP 503 and a generic user-safe message. They never silently bypass the
  limit.
- A later healthy request recovers without process-local counter repair because
  Redis remains authoritative.
- Keys use the `atcroster:<environment>:limit` namespace and an HMAC-SHA256
  digest of scope, network source and subject. Usernames, email addresses,
  invitation/reset tokens and health information are not stored in keys.
- Fixed-window keys receive a TTL on first increment. Temporary limiter data
  therefore expires automatically.

Development uses the explicitly local in-memory limiter. It is not an accepted
multi-instance production fallback.

## Operator response

1. Treat repeated `rate_limiter_unavailable` security events or failed
   readiness as an authentication-service incident.
2. Check managed Redis availability, TLS/network policy, connection limits and
   latency without logging the URL or credentials.
3. Do not disable fail-closed behaviour or switch production to the in-memory
   limiter.
4. Restore Redis, confirm readiness, then test a failed and successful login
   rate-limit cycle.
5. Record the outage duration and whether any account workflows were delayed.

Tests cover connection refusal, timeout, malformed results, intermittent
recovery, shared counters across independent clients, key privacy and TTL.

