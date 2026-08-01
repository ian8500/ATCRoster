# Tenant-integrity inventory

Date: 1 August 2026  
Baseline: `35d3c54`

ATCRoster uses a physical control database plus physically routed airport
databases. Physical separation is the primary boundary. Composite constraints
remain necessary defence in depth for combined development databases,
misrouting, imports, restores and future topology changes.

## Enforced in revision 20260801_35

The migration first validates every non-null reference and reports up to ten
inconsistent row IDs. It never changes `unit_id` or a referenced ID. PostgreSQL
then receives candidate keys on `(unit_id, id)` and composite foreign keys for:

- staff links from assignments, leave, sickness, shift requests,
  notifications, watch history, qualifications, training, roster
  acknowledgements, live-position records, break plans, achieved duty, fatigue
  reports, kiosk credentials and MFA credentials;
- shift request to resulting assignment and request-audit to request;
- qualification to qualification type and qualification history to its source;
- roster acknowledgement to publication;
- training objectives, sessions and scores to their same-unit parents;
- annotation evidence to annotation type and assignment;
- live-position status, session, participant, audit, endorsement and requirement
  rows to same-unit positions, sessions, roles, categories and people;
- achieved duty to its planned assignment.

Live-position status, session and participant idempotency keys are changed from
global uniqueness to `(unit_id, transaction_key)` uniqueness. This permits the
same client-generated key in two airports while preserving exactly-once
behaviour inside one airport.

## Migration failure and repair procedure

When preflight finds an orphan or tenant mismatch, migration stops before the
new constraints are applied. The error identifies the child table, reference
column, parent table and sample row IDs. Operators must:

1. preserve a verified backup;
2. determine the authoritative tenant and referenced record using approved
   operational evidence;
3. prepare a separately reviewed data-repair statement;
4. retain the repair approval and affected IDs in the change record;
5. rerun the migration and grant verification.

Deleting rows or silently changing tenant ownership during migration is
prohibited.

## Deliberate exclusions

- Control-plane identity relationships cross a physical database boundary and
  cannot use operational composite foreign keys. They remain protected by the
  tenant router, membership validation and physical topology tests.
- Historical actor/reference fields that deliberately preserve evidence after
  account removal remain scalar identifiers where no durable same-database
  parent is guaranteed. They are still stamped and filtered by `unit_id`.
- SQLite cannot safely add these production constraints without rebuilding many
  legacy tables. Revision `20260801_35` records the schema version but applies
  the composite constraints only on PostgreSQL. PostgreSQL integration tests are
  therefore a mandatory release gate.

## Evidence

- A valid blank PostgreSQL operational database migrates to `20260801_35`.
- Deliberate cross-unit assignment insertion fails with a database
  `IntegrityError`.
- A legacy cross-unit assignment causes migration preflight to fail with repair
  guidance rather than rewriting ownership.

