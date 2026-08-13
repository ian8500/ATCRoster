# ATCRoster website product specification

No public website source is present in this repository. This specification is
the source of truth for customer-facing ATCRoster copy and must be kept aligned
with `docs/product-truth-matrix.md` and the README.

## Naming

- **Readback Correct** — the parent company proposition.
- **ATCRoster** — the product.
- **ATCO Roster** — ATCRoster's roster-management module.
- **Position Monitor** — ATCRoster's live operational position-monitoring
  module. “Live Position Monitor” may be used once as a descriptive expansion,
  but Position Monitor is the preferred customer-facing name.

## Readback Correct

Readback Correct develops focused operational software for air traffic control
teams. Avoid claims of safety assurance, regulatory approval or operational
guarantees unless separately evidenced for a particular customer deployment.

## ATCRoster proposition

ATCRoster brings ATCO roster planning, controlled publication and operational
position visibility into a tenant-isolated web application. It is suitable for
controlled customer pilots subject to local configuration, validation and the
external assurance activities described in the README.

## ATCO Roster page

Describe verified capabilities: configurable shifts, watches and repeating or
dated work patterns; staffing requirements; deterministic future population;
manual roster editing; protected future horizons; effective-dated workforce
changes; leave and requests; fatigue and qualification warnings; controlled
publication, acknowledgement, reporting and audit.

Say: “supports airport-configured fatigue and qualification checks to inform
roster decisions.” Do not say it guarantees legal, regulatory or fatigue
compliance.

## Position Monitor page

Describe an operational HMI that shows open, vacant and occupied positions;
primary and secondary controller roles; OJTI and assessor participation;
current medical/UE/endorsement checks; configurable position-time allowances;
time-of-day allowance matrices; handover; auditable lifecycle events; and
live updates with a visible stale/degraded indication when current data cannot
be confirmed.

## Supporting capabilities

Only claim: qualifications, training/competency, briefing, handover,
notifications, account/MFA workflows, reporting, audit and tenant-separated
operational data. Provider integrations should be described as configurable,
not guaranteed.

## Security and architecture

State factual controls: Flask application, PostgreSQL in production, Redis for
production services, a control-plane database plus physically separate
operational databases per airport, authenticated membership-based tenant
routing, CSRF protection, MFA workflows, rate limiting and server-side
authorisation. Do not use absolute claims such as “completely secure.”

## Product status

Use: “ATCRoster is being prepared for controlled commercial pilots.” A pilot
requires airport configuration validation, operational acceptance, production
infrastructure verification, backup/restore rehearsal, independent security
testing and any applicable safety/privacy assessment.

## Screenshots to capture from real seeded/demo data

1. Monthly ATCO Roster: requirements, totals and clearly non-sensitive sample
   staff data.
2. Roster edit state: a validation/warning example and protected-horizon cue.
3. Publication/acknowledgement view: authoritative version shown without real
   employee data.
4. Position Monitor board: Tower/Radar grouping, primary and OJTI roles,
   elapsed/remaining time and status labels.
5. Position Monitor degraded state: visible “data may be stale” warning.
6. Position configuration: time allowance matrix and role configuration.

Never publish screenshots containing real personnel, medical, qualification,
contact, roster or operational data without the relevant customer approval.

## Claims register

| Website claim | Repository evidence | Safe to publish? | Qualification/wording |
| --- | --- | --- | --- |
| ATCO roster generation and editing | `atcroster/roster`, roster population/editing tests | Yes | “Supports configurable roster generation and controlled manual editing.” |
| Future changes retain roster control | roster impact and horizon tests | Yes | “Protected horizons preserve near-term planning control.” |
| Position Monitor shows controller roles and time | `live_position_blueprint.py`, monitor tests | Yes | “Shows current configured position state, roles and elapsed/remaining session time.” |
| Position Monitor warns when data is unconfirmed | kiosk template and stale-state test | Yes | “Shows an explicit degraded state when live data cannot be confirmed.” |
| Tenant-separated airport data | PostgreSQL multi-database tests | Yes | “Uses authenticated tenant routing and separate operational databases.” |
| Fatigue/qualification compliance is guaranteed | No repository evidence | No | Say only that configured findings and warnings support decisions. |
| ATCRoster is secure or penetration-tested | No independent assurance evidence | No | Describe specific implemented controls only. |
| Zero downtime or no roster errors | No repository evidence | No | Do not publish. |
