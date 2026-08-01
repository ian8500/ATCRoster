# Commercial pilot release checklist

Release commit: __________  Airport: __________  Owner: __________  Date: __________

- [ ] Exact commit is on protected `main`; Quality, PostgreSQL/Redis, CodeQL,
      gitleaks, container/Trivy and SBOM jobs are green.
- [ ] `scripts/verify_release_candidate.py` is captured for the clean commit and
      does not substitute for GitHub CI evidence.
- [ ] Managed PostgreSQL and managed Redis are provisioned with TLS/private access.
- [ ] Migration-owner and runtime roles are separate; runtime grants are applied
      and `verify_runtime_database_grants.py` passes for control and airport DBs.
- [ ] Production secrets, trusted hosts/proxy hops and private object storage are
      configured; rotation owners and dates are recorded.
- [ ] TLS, HSTS, DNS and rolling deployment are verified on the customer origin.
- [ ] Encrypted off-site backup completed and a representative restore rehearsal
      passed within the agreed RPO/RTO.
- [ ] Central logs and every web/worker metric target are visible; availability,
      5xx, Redis, stale worker, queue age and backup alerts were test-fired.
- [ ] Incident contacts, escalation, maintenance window and rollback authority are
      assigned.
- [ ] Independent penetration test findings are accepted or remediated.
- [ ] Legal/privacy review covers workforce and health information, processors,
      retention, contracts and subject-rights operations.
- [ ] Independent operational safety review accepts roster/fatigue/qualification
      and live-position advisory boundaries.
- [ ] Local live-position procedures, controller briefing and fallback are signed.
- [ ] Customer acceptance test covers each role/module, multiple devices, reports,
      kiosk, concurrency conflicts and tenant isolation.
- [ ] Rollback image/config is available and application rollback plus database
      restore decision path was rehearsed.

Any unchecked security, integrity, backup, monitoring, privacy or safety item blocks
commercial pilot release unless the accountable owner records a time-bounded risk
acceptance and safe compensating control.
