# Performance and capacity assurance

Performance claims must be based on repeatable measurements from a synthetic
staging environment, never from customer operational data.

## Release measurement

Seed a fresh acceptance database, then run the authenticated month roster and
Position Monitor surfaces through the HTTP measurement tool. Supply a session
cookie only from a local shell secret or CI secret; the tool never writes it.

```bash
python scripts/measure_http_performance.py \
  --cookie "__Host-atcroster=…" \
  --repetitions 10 \
  --max-median-ms 500 \
  --max-html-bytes 100000 \
  --max-dom-elements 1500 \
  https://pilot.atcroster.com/roster/2026-08 \
  https://pilot.atcroster.com/live-position/kiosk
```

The limits are release-review budgets, not promises about every network,
browser, airport configuration or database size. Record the JSON result with
the release, including dataset size, browser/device, region and timestamp.

## Capacity scenario

Run the isolated structural scale check before a release that changes roster
queries, tenant routing or assignment indexes:

```bash
python scripts/scale_assurance.py --units 30 --people 40 --days 90
```

Then perform a PostgreSQL load test with realistic synthetic rosters and
concurrent editor traffic. Capture p50/p95/p99 route latency, error rate,
database pool saturation, Redis latency, worker queue age and Position Monitor
update age. Set numeric launch thresholds only after this baseline exists.

## Response rules

- Stop promotion for an unexplained performance-budget breach, 5xx increase or
  tenant-isolation failure.
- Roll back to the retained Railway deployment if a post-deploy regression is
  confirmed.
- Do not log roster notes, medical data, request comments, tokens or session
  identifiers in benchmark artefacts.
