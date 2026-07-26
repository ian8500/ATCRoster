# ATCRoster production architecture review

Updated 24 July 2026. The filename is retained to preserve existing links.

The July 2026 security hardening separates the control-plane schema from every
airport operational schema. Control owns global identities, memberships,
platform MFA, provisioning and signup workflow state; airport databases own
all personnel and roster information. Role-filtered Alembic migrations and a
three-database PostgreSQL 16 CI job verify the physical boundary. See
[control_operational_database_design.md](control_operational_database_design.md).

## Current platform

ATCRoster is a Flask/SQLAlchemy application with server-rendered responsive
views, a PostgreSQL production target, Alembic migrations and a desktop SQLite
compatibility mode.

The production platform now includes:

- authenticated airport tenant context and query/write isolation;
- separate platform identities, airport memberships and operational people;
- Super Admin airport/account control without operational-data access;
- roster, request, leave, sickness, overtime, annotations and TOIL workflows;
- explainable fatigue and qualification compliance;
- operational positions, endorsements and resilient staffing demand;
- break plans and planned-versus-achieved duty records;
- controller fatigue reports and manager review;
- governed, versioned rostering-system rules;
- controlled publication, assurance declarations and acknowledgements;
- encrypted TOTP MFA and one-time recovery codes;
- production startup validation, readiness/liveness endpoints and request IDs;
- Docker/Compose packaging and GitHub CI.

## Production boundaries

The application is production-capable software, not evidence by itself of
operational or regulatory approval. A live airport deployment still requires:

- competent validation of the configured rostering rules;
- integration into the provider's Safety Management System;
- DPIA and data-controller/processor agreements;
- independent penetration and accessibility testing;
- rehearsed backup, restore, incident and contingency procedures;
- CAA notification/approval where required by the provider's change process;
- organisation-specific operating instructions and training.

## Architectural decisions

### Flask rather than a forced rewrite

The current server-rendered platform retains substantial tested operational
domain behaviour. A React/FastAPI rewrite would introduce a long parity and
safety-validation period without directly improving roster assurance. New
domain models and workflows therefore remain within the tested application
while service boundaries are progressively extracted.

### Tenant isolation

The default connection is the central control database for identities,
memberships, airport accounts, plans, feature flags, safe platform audits,
aggregate usage and secret-name routing metadata. Every operational mapper is
routed by `TenantRoutedSession` to the authenticated airport's engine.
Credentials are resolved from deployment secrets and never from browser data.
Missing or inconsistent production routes fail closed. Platform-control
contexts are explicitly forbidden from resolving an operational engine.

Every operational record also carries `unit_id` as defence in depth.
SQLAlchemy applies tenant criteria and stamps/rejects writes, while physical
database separation prevents cross-airport queries at the connection boundary.
Flask integration tests exercise distinct SQLite files and PostgreSQL 16
databases.

### Schema control

Production disables runtime `create_all`, compatibility alters and seed data.
Alembic owns production upgrades. SQLite runtime upgrades remain limited to
desktop/development compatibility.

### Authentication

Passwords use Werkzeug hashes. TOTP secrets use Fernet encryption with a
deployment-only key. Production forces users without MFA enrollment to the
setup flow. Enterprise SSO/passkeys can be added later without weakening the
current enforced MFA boundary.

### Safety assurance

Automated findings are decision support. Publication applies hard blocks for
incomplete assignments, competence failures, operational configuration,
position shortfalls and unresolved high/unfit fatigue reports. Remaining
exceptions require an accountable-manager rationale and are stored with the
immutable publication snapshot.

## Recommended next architectural work

1. Extract authentication, assurance and roster services from `app.py`.
2. Add a documented versioned REST API for approved integrations.
3. Run the existing PostgreSQL physical-isolation integration flow in CI.
4. Integrate central rate limiting, SIEM and organisation SSO.
5. Add traffic-demand/sector-opening integrations where the unit can supply an
   authoritative data source.
