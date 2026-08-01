# Disaster recovery

## Assumptions and authority

ATCRoster is currently a controlled-pilot service, not a highly available
multi-region system. The incident commander must authorise restoration and
record the recovery point, affected airports, evidence, decisions and owners.
Never restore production over the existing database.

## Recovery sequence

1. Contain the incident, revoke compromised credentials and preserve evidence.
2. Identify the last verified control and per-airport recovery set before the
   compromise or corruption time.
3. Create private, isolated empty PostgreSQL targets with new credentials.
4. Verify/decrypt each artifact and restore using the repository tooling.
5. Confirm schema versions and counts; reconcile memberships against the
   corresponding operational-person records.
6. Deploy the exact matching tested application image in isolation.
7. Run readiness, authentication, tenant-isolation, roster/publication, audit
   and live-position acceptance checks.
8. Obtain security, privacy, operational and customer go/no-go approval.
9. Rotate all affected application, database, Redis, storage, email and field-
   encryption credentials before traffic cutover.
10. Monitor closely, communicate the measured data-loss window, and complete
    the incident/post-recovery review.

## Per-airport restore

An airport operational database may be restored independently only when its
control-plane routing record and membership/person mappings are reconciled.
Keep the airport suspended until those checks pass. Do not copy rows between
airport databases as an expedient repair.

## Required rehearsals

Run an isolated restore at least quarterly and before material database changes.
Record backup age, restore duration, validation results, discrepancies and
secure destruction of the rehearsal environment. CI proves the mechanical
custom-format backup/restore path; it does not replace an operator rehearsal,
provider failover test or customer acceptance.
