# Monitoring and alerting standard

## Availability signals

Monitor from outside Railway and from a different provider/region:

The repository's `production-health.yml` provides a baseline independent
scheduled probe through GitHub Actions. It is not a substitute for a dedicated
monitor with paging and a public status page; scheduled Actions can be delayed.
Failed production probes and backup jobs create or update a single labelled
GitHub incident, and a later successful check records recovery and closes it.
`backup-freshness.yml` checks every three hours and opens an incident when no
successful encrypted backup exists within the 26-hour threshold. Repository
owners must enable GitHub email/mobile notifications for monitoring issues;
GitHub issues are not a substitute for 24/7 paging.

| Signal | Check | Interval | Alert |
|---|---|---:|---|
| Public DNS/TLS | `https://atcroster.com` resolves; valid certificate | 5 min | 2 failures |
| Liveness | `GET /health/live`, HTTP 200, `status=ok` | 1 min | 3 failures |
| Readiness | `GET /health/ready`, HTTP 200, `status=ready` | 1 min | 2 failures |
| Login page | `GET /login`, expected content and latency | 5 min | 2 failures |
| Deployment | Railway deployment reaches healthy state | event | any failure |
| Worker | heartbeat/provisioning queue age | 5 min | no heartbeat 10 min or oldest ready job 10 min |
| Database | connections, storage, CPU, latency | 5 min | sustained threshold 10 min |
| Redis | reachability, memory and eviction | 5 min | unavailable or unexpected eviction |
| Backup | latest successful independent backup | daily | age exceeds 26 hours |
| Restore evidence | isolated restore rehearsal | quarterly | overdue |
| Email/SMS | provider delivery failures and spend | daily/event | abnormal failure rate or budget threshold |

Health checks must not expose database URLs, exception text, airport names,
people, rosters or counts. Synthetic monitoring must never use a real user's
credentials.

## Application/error monitoring

Capture request ID, route template, response status, release identifier,
duration and safe exception type. Alert on:

- any cross-tenant assertion or security-boundary failure;
- repeated HTTP 500s or a material rise over baseline;
- login/recovery rate-limit anomalies;
- failed database migration or provisioning job exhaustion;
- unexpected privileged-role, MFA-reset or secret-rotation events; and
- message delivery failures above the agreed unit threshold.

Do not send passwords, MFA data, tokens, message content, sickness/medical
information, request comments or roster notes to monitoring providers.

## Severity and routing

| Severity | Example | Acknowledge target | Update target |
|---|---|---:|---:|
| SEV-1 | suspected cross-unit disclosure, total outage, roster integrity uncertain | 15 min, 24/7 after commercial launch | every 30 min |
| SEV-2 | major function unavailable, one airport materially impaired | 1 hour in support hours | every 2 hours |
| SEV-3 | limited defect with workaround | 1 business day | on material change |
| SEV-4 | question, cosmetic issue, enhancement | 2 business days | as scheduled |

Targets are operational objectives until incorporated into a signed service
level agreement. SEV-1 must page the service owner and security/privacy contact.
If roster integrity is uncertain, tell the unit to use its approved contingency
process; do not describe ATCRoster as an operational fallback.

## Dashboard

One dashboard should show:

- 30-day availability and latency;
- live/readiness state and current release;
- error rate by safe route category;
- database/Redis capacity;
- worker queue age;
- backup age and last restore result;
- email/SMS delivery failures and spend; and
- open incidents and overdue operational actions.

## Go-live evidence

- External monitor URL/dashboard: `[LINK]`
- Paging destination and test date: `[DETAILS]`
- Railway deployment alert: `[LINK/DATE]`
- Error-monitoring project and scrub test: `[LINK/DATE]`
- Backup freshness alert and forced-failure test: `[LINK/DATE]`
- Status page: `[URL]`
