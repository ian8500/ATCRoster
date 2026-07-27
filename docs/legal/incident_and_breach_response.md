# Security incident and personal-data breach response

## Reporting

Users and operators must immediately report suspected unauthorised access,
misdirected SMS/email, exposed credentials, cross-airport access, lost exports,
malware, destructive change or availability/data-integrity failure to the
designated incident contact.

## First response

1. Open a restricted incident record; note discovery time and reporter.
2. Preserve relevant logs and evidence without unnecessarily copying personal
   data.
3. Contain: revoke sessions/tokens, disable affected accounts, isolate services,
   rotate exposed secrets or stop messaging as appropriate.
4. Confirm airport(s), people, data categories, special-category involvement,
   period, recipients, integrity/availability effect and recoverability.
5. Notify the affected controller without undue delay once a personal-data
   breach affecting its data is confirmed; do not wait for a complete forensic
   report.

## Assessment and notification

The controller documents likelihood and severity of harm. It decides whether
notification to the ICO is required and, where required, submits within 72 hours
of awareness, supplementing incomplete information later. High-risk affected
people must be informed without undue delay unless a lawful exception applies.
Record every personal-data breach, including the rationale where notification is
not made.

## Recovery

- Remove persistence/root cause, patch, restore from a verified clean source and
  validate tenant isolation, authentication and data integrity.
- Use an approved rollback and communicate service status without disclosing
  exploitable detail or personal information.
- Monitor for recurrence and suspicious account activity.

## Closure

Within ten working days of stabilisation, record timeline, cause, affected
records, decisions, notifications, corrective actions, owner and due dates.
Track actions to completion and feed regression tests/change controls. Preserve
the incident record according to the approved schedule.

## Contacts to complete

- IDAviation incident lead: `[NAME / 24H METHOD]`
- Privacy lead: Ian John Dickson — privacy@atcroster.com
- Railway support/escalation: `[DETAILS]`
- Twilio support/escalation: `[DETAILS]`
- Customer incident contacts: maintained in each order form
