# Release and change management

## Normal change

1. Define user outcome, risk, affected permissions/data and rollback boundary.
2. Work on a protected branch; require review and passing quality/security
   checks.
3. Update tests, user documentation, migrations, privacy/assurance records and
   release notes where affected.
4. Deploy the exact commit to staging; migrate a representative restored copy.
5. Run automated tests plus risk-based acceptance for every affected role/unit.
6. Confirm backup, schema-compatible rollback image, decision owner and
   monitoring coverage.
7. Approve production deployment during a communicated window.
8. Verify live/readiness, login, one safe read per role, tenant isolation
   smoke-check and affected workflow.
9. Monitor intensively for at least 30 minutes and record evidence.

## Change classes

- **Standard:** documented low-risk repeatable operation with tested runbook.
- **Normal:** reviewed release using the complete flow above.
- **Emergency:** required to contain material outage/security/integrity risk.
  The incident owner authorises minimum change; review and missing evidence are
  completed next business day.

Direct production database edits are prohibited except an authorised emergency
runbook with pre-change backup, peer check, transaction boundary, evidence and
reconciliation.

## Release authority

- Product owner confirms scope and customer communication.
- Technical release owner confirms tests, migration and rollback.
- Operational/safety reviewer approves changes to roster generation, fatigue,
  qualifications, staffing counts, publication and access control.
- Privacy/security reviewer approves changes to sensitive data, providers,
  permissions, logs, messaging and authentication.

One person may hold multiple roles during pilot, but commercial release of a
high-risk change requires an independent second reviewer.

## Rollback decision

Rollback or disable the affected feature when there is cross-unit exposure,
unreconciled roster writes, authentication bypass, sustained elevated errors,
failed migration, or no safe understanding of impact. Schema rollback is restore
from approved backup—not ad-hoc downgrade SQL.
