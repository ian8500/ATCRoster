# Customer onboarding and go-live

## 1. Discovery

- Contracting entity, unit, operating hours, watches and user count
- Current roster cycle, shift definitions and publication process
- Roles, qualifications, fatigue rules and staffing requirements
- Leave/sickness categories and retention requirements
- SMS/email needs, data migration and integrations
- Privacy, information-security, safety and acceptance owners
- Contingency process and target go-live window

Output: signed scope, order/DPA, owners, risks, success criteria and plan.

## 2. Secure provision

Create airport metadata and isolated database; configure secrets, user limit,
branding and bootstrap invitation. The named administrator accepts the
invitation, sets unique credentials and MFA. Record provisioning/migration
evidence without putting credentials in tickets.

## 3. Configuration workshop

Complete the in-app onboarding workflow: airport details, watches, base pattern,
anchors, night availability, shifts/OFF, count mappings, annotations,
leave/sickness types, fatigue rules, requirements, messaging senders and roles.

## 4. Data and validation

Import only approved minimum data. Reconcile people, watches, patterns,
qualifications, leave, assignments and published month totals against
controller-approved source records. Record exceptions and owner decisions.

## 5. Training and acceptance

Complete administrator, manager/editor and ATCO training. Run
`docs/manual_acceptance_test.md` with representative customer data across all
roles. Exercise access isolation, publication/undo, request approval, warning
explanations, exports, backup restore and contingency.

## 6. Parallel pilot

Operate alongside the approved current method for the agreed period. Review
defects weekly; do not expand scope mid-pilot without change control. Define who
may declare ATCRoster authoritative and when.

## 7. Go-live decision

Customer operational/safety, privacy/security and accountable business owners
sign the readiness checklist. Confirm support, on-call, monitoring, backups,
rollback, status page and incident contacts. Archive the accepted configuration
and release identifier.

## 8. Handover

Provide administrator/user manuals, training records, support process, security
statement, DPA/subprocessor list, data-exit method and first service review date.
Schedule 2-week and 6-week adoption reviews, then quarterly service reviews.
