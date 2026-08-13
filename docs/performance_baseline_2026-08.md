# Performance baseline — August 2026

## Reproducible checks available locally

The checked-out development environment did not include a running PostgreSQL,
Redis or representative authenticated fixture, so route and browser timings were
not recorded.  This document deliberately does not substitute estimates for
measurements.

The following structural baseline was established from `main` before the roster
cell refactor:

| Surface | Baseline implementation |
| --- | --- |
| Editable monthly-roster cell | One form, two hidden inputs and one select, plus options |
| Shift editor | Repeated once for every editable cell |
| Static version lookup | `os.path.getmtime` on every `asset_url()` call |
| Versioned static response | No explicit immutable cache policy |
| Month display calculations | In the route handler |

For a measured runtime comparison, run the authenticated fixture through the
existing browser telemetry endpoint (`/roster/telemetry`) against `main` and
this branch. It records browser render, DOM-interactive and transfer values
without recording user data.

For repeatable HTTP, transfer-size and HTML/DOM-size measurements, use
`scripts/measure_http_performance.py` against a local or staging URL. Supply
an authenticated session cookie only through the command line/environment;
the utility does not persist it. Example:

```bash
python scripts/measure_http_performance.py --cookie "session=…" \
  http://127.0.0.1:8000/roster/2026-08
```
