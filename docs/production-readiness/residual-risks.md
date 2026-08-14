# Residual production risks

Status: repository controls improved; external production acceptance incomplete.

- Independent authenticated penetration testing has not been evidenced here.
- Managed production infrastructure, private networking, multi-instance scaling,
  database grants and secrets rotation must be applied and verified externally.
- Repository tests cannot prove off-site retention, recovery-key custody, restore
  rehearsal timing or Railway/provider disaster recovery.
- Logs/metrics exist, but central collection, alert thresholds, paging and tested
  incident ownership remain deployment work.
- Root `app.py` is a compatibility/WSGI entrypoint and the modular composition
  root is `atcroster/application.py`. Continued domain-boundary review remains
  important, but the historical ten-thousand-line root module is no longer the
  active architecture.
- PostgreSQL concurrency coverage must remain mandatory and expand with each new
  roster/publication/live-position transition; SQLite cannot substitute for it.
- CSP still trusts integrity-pinned third-party Bootstrap and Font Awesome CDNs.
  Vendoring is recommended before a broader commercial rollout.
- Waitress is suitable for the limited pilot profile but capacity, graceful rolling
  behaviour and reverse-proxy timeouts require load and failure testing on the real
  platform.
- Medical/absence data requires customer-specific privacy, retention and access
  review. Repository controls do not establish a lawful basis.
- Live-position timers and fatigue outputs remain advisory; independent operational
  safety assessment, local procedures and human fallback are mandatory.
- Legal terms, processor contracts, customer acceptance and support/on-call
  capability remain outside repository verification.
