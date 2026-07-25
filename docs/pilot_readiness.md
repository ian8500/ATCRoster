# Pilot readiness and acceptance plan

This document is a reusable starting point for a controlled airport pilot. It
is not legal, regulatory or safety assurance advice.

## Entry criteria

- Named pilot sponsor, Unit Admin and operational safety owner
- Approved non-production or representative test dataset
- Completed data-flow map and DPIA screening
- Agreed support hours, escalation contacts and incident process
- PostgreSQL environment with TLS, managed secrets and encrypted backups
- MFA or SSO implementation appropriate to the pilot exposure

## Acceptance scenarios

1. Create an airport, securely transfer its opaque bootstrap invitation, and
   have the first Unit Admin choose credentials and enrol MFA.
2. Verify that users cannot read or change another airport's data.
3. Configure watches, shifts, qualifications and staffing requirements.
4. Import or create a representative roster and reconcile record counts.
5. Exercise leave, sickness, requests, overtime and account-limit workflows.
6. Review every Compliance Centre finding with the operational safety owner.
7. Publish a roster replacement and verify supersession history.
8. Record staff acknowledgement against the correct publication version.
9. Export audit and compliance evidence and confirm minimised content.
10. Restore the control and airport databases into an isolated environment.
11. Test loss of network, expired sessions and disabled accounts.
12. Complete keyboard, screen-reader, mobile and print checks.

## Exit criteria

- No open critical security or tenant-isolation defects
- No unexplained data-reconciliation differences
- Restore time and recovery point accepted by the sponsor
- Fatigue-rule configuration signed off by a competent operational owner
- Staff and administrator training completed
- Support, rollback and data-exit processes rehearsed
- Formal go/no-go decision recorded outside the application

## Evidence pack

Retain test results, rule configuration, screenshots, export samples,
penetration-test summary, accessibility review, restore evidence, approval
record and the version/commit used for the pilot.
