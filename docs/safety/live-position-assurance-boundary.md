# Live-position monitoring assurance boundary

## Intended use

ATCRoster Live Position Monitoring records who is working a configured
operational position, supporting OJTI/assessor participation, elapsed time and
locally configured advisory time limits. It provides operational awareness,
currency evidence and retrospective reports to authorised airport users.

The timers, remaining-time indications, qualification checks and warnings are
**advisory decision support**. They are not a certified operational safety
control, separation tool, surveillance system, watch alarm or substitute for
the unit's approved position-management, fatigue, competency and contingency
procedures.

## Excluded use

Do not use the module as the sole means to:

- determine whether an individual is legally or operationally fit for duty;
- guarantee position coverage, relief or sector capacity;
- authorise a rating, endorsement, medical, OJTI or assessor privilege;
- detect controller incapacity or loss of situational awareness;
- provide an authoritative time source; or
- continue operations during application, network, database or display failure.

## Configuration ownership and human checks

The airport's accountable operational owner must approve position definitions,
groups, maximum-time matrices, warning thresholds, timezone and qualification
data. Unit administrators configure the service but do not thereby approve
operational policy. Before relying on a display, staff must confirm the correct
airport, position, controller, role and local time and resolve any mismatch
using the approved local process.

The person handing over and the receiving controller remain responsible for
the operational handover. A successful click or screen update is not evidence
that an operational briefing or verbal handover was adequate.

## Data freshness and alarm limitations

The display depends on a working kiosk browser, application, network, Redis
where used, and the airport operational database. Browser refresh, clock skew,
sleeping devices, stale connections, delayed requests and configuration errors
can make indications late or incorrect. Server-side UTC session timestamps are
authoritative within the application; airport-local conversion is used only
for policy selection and display.

Warnings may be missed, delayed, obscured, muted or unavailable. No audible or
visual warning should be treated as guaranteed. Local procedures must require
an independent awareness of time-on-position and relief requirements.

## Failure and outage behaviour

On suspected stale data, contradictory state or outage:

1. Stop relying on the display and use the approved local fallback log/process.
2. Preserve operational safety through the unit's normal supervisory chain.
3. Record actual logon, handover and logoff times for later reconciliation.
4. Report the fault without placing sensitive operational/personnel details in
   unsecured channels.
5. Reconcile and, where authorised, correct records with an auditable reason
   after service restoration.

The application must fail closed for unauthorised access, but loss of the
application must not prevent the unit from safely operating under its approved
contingency arrangements.

## Required local assurance

Before activation, each customer must complete operational safety assessment,
configuration review, representative acceptance testing, outage exercise,
training and documented fallback arrangements. Material software or local-rule
changes require impact review. ATCRoster's automated tests demonstrate software
invariants; they do not certify aviation safety or replace regulator/customer
acceptance.
