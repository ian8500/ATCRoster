# Data classification and handling

| Class | Examples | Access | Required handling |
| --- | --- | --- | --- |
| Public | Product help, generic shift-template descriptions | Everyone | Integrity review; no secrets |
| Internal service metadata | Airport code/name, plan, feature flags, aggregate account count, health | Super Admin; relevant Unit Admin subset | Audit changes; exclude personnel identifiers |
| Confidential operational | Rosters, staff number, requests/comments, watches, TOIL, leave dates | Authorised unit memberships | Tenant isolation, TLS, encryption at rest, audited export, approved retention |
| Special/restricted | Sickness detail, fatigue reports, medical and qualification data | Minimum authorised unit roles | No central-log content; enhanced retention/access review; incident escalation |
| Authentication secret | Password hashes, MFA secrets, invitation tokens, session keys, DB credentials | Application/security operators only | Secret store; encryption; rotation; never display or log; tokens single-use |
| Audit/evidence | Actor IDs, transitions, publication declarations, access/security events | Authorised audit/operations roles | Append-oriented, timestamped, protected from alteration, retained by policy |

Super Admin views may contain only internal service metadata and anonymised
health/error aggregates. Operational exports must be scoped to the current
unit. Production retention and deletion periods must be approved in the DPIA
and records-of-processing schedule.

