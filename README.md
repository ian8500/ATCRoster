# ATCRoster

ATCRoster is a roster planning and workforce-compliance application for airport
air traffic control units. It supports day-to-day roster editing, shift
requests, fatigue checks, qualifications, annotations and TOIL, reporting,
scenario planning, and versioned publication. Airport operational data is
tenant-bound; the platform administration area exposes account and service
aggregates only.

## Contents

1. [Roles and privacy](#roles-and-privacy)
2. [Getting started](#getting-started)
3. [Unit administration workspace](#unit-administration-workspace)
4. [Airport onboarding](#airport-onboarding)
5. [Using the roster](#using-the-roster)
6. [Shift requests](#shift-requests)
7. [Annotations and TOIL](#annotations-and-toil)
8. [Qualifications and compliance](#qualifications-and-compliance)
9. [Coverage and scenario planning](#coverage-and-scenario-planning)
10. [Leave, sickness, overtime, and reports](#leave-sickness-overtime-and-reports)
11. [Accounts and platform administration](#accounts-and-platform-administration)
12. [Installation and deployment](#installation-and-deployment)
13. [Database migrations and legacy import](#database-migrations-and-legacy-import)
14. [Security operations](#security-operations)
15. [Backup, restore, and recovery](#backup-restore-and-recovery)
16. [Testing](#testing)
17. [Troubleshooting](#troubleshooting)

## Roles and privacy

| Role | Intended access |
| --- | --- |
| `SuperAdmin` | Airport account metadata, plans, feature flags, database health, migration state, storage, and aggregate activity. No personnel or roster data. |
| `UnitAdmin` | Full administration of their own airport, including people, unit configuration, request decisions, and annotation definitions. |
| `RosterEditor` | Roster editing and permitted annotation application. Cannot edit annotation definitions. |
| `WatchManager` | Explicitly granted watch and annotation actions only. |
| `StaffUser` | Their own roster, requests, notifications, and permitted self-service functions. |
| `ReadOnlyAuditor` | Product target only; not currently provisionable. Do not offer until its read-only route/export contract is implemented and tested. |

An operational `Person` does not need a login. Authentication belongs to a
`PlatformIdentity`; airport access belongs to `UnitMembership`, which may link
to a roster `Person`. The authenticated membership determines the airport
context. Forms and query strings never select a tenant database.

The platform administrator must never be given operational database
credentials, staff search, impersonation, or exports. Future support access
requires a separate airport-approved, time-limited, reason-bound,
MFA-protected, fully audited design.

## Getting started

After signing in:

- Use **Roster** for the monthly operational view.
- Use **Shift Requests** to request a future requestable shift or, as a Unit
  Admin, respond to requests.
- Use **Published** to review and acknowledge the currently released roster
  version.
- Check the airport name and code in the page header before making changes.
- Use **Admin** for staffing requirements, shift definitions, people, and
  unit tools.
- Use **Accounts** to create and deactivate login accounts if you are a Unit
  Admin.
- Use **Reference Data** to manage annotation definitions and roster code
  lists if you are a Unit Admin.
- Use **Reports** for fatigue, sickness, leave-year, overtime, swap, and
  extension information.
- Use **Qualification compliance** at `/compliance`.
- Review automatic fatigue findings directly on the relevant roster cells.
- Use **Coverage heatmap** at `/planning/coverage/YYYY-MM`.
- Use **Roster scenarios** at `/planning/scenarios`.
- Use **Airport onboarding** at `/unit/onboarding`.
- Use **Forgot username** or **Forgot password** on the login page when access
  is lost. Username reminders go to the account email; accounts without an
  email are referred to the airport administrators. Password resets use a
  two-stage process: an airport administrator approves the request, then the
  account holder receives a separate, expiring reset link. Unit Admin requests
  are approved by a Super Admin.

The shared interface includes:

- a compact mobile **Menu** that keeps the current page and airport context
  visible;
- a consistent Messages-style page banner, Inter typography, dark surface
  palette, form controls, cards and data tables across every operational,
  reporting and administration workspace (the roster retains its specialist
  dense-grid sizing);
- keyboard skip navigation, visible focus states and reduced-motion support;
- persistent roster zoom presets at 75%, 90%, 100% and **Fit width**;
- clear success/error announcements and purpose-built 400, 403, 404 and 500
  recovery pages;
- duplicate-submit protection on write forms;
- confirmation before deleting requests, leave/sickness records or
  deactivating accounts;
- CSRF protection on leave/sickness, overtime, requests, publication,
  operational assurance, accounts and platform administration workflows.

Dates are displayed using the airport configuration. Server timestamps and
audit events are recorded in UTC.

On a supported mobile browser, install ATCRoster from the browser's **Add to
Home Screen** or **Install app** action. The service worker caches only static
application assets; authenticated roster pages and personnel data are not
stored for offline viewing.

## Unit administration workspace

The airport administration page at `/admin` is written as a task-based
workspace for non-technical unit managers. Its opening checklist shows which
setup areas are ready and takes the administrator directly to anything needing
attention. It is divided into focused sections:

- **Overview** summarises the configured staffing months, shift codes, staff,
  and supporting tools.
- **Roster setup** defines the unit base cycle, the weekdays on which night
  duties operate, and the unit's named watches.
- **Staffing levels** edits monthly Morning, Day, Afternoon, and Night minimums
  across a rolling 24-month planning window. **Copy below** repeats one
  month's four values into every later row for review before saving.
- **Shifts** creates shift definitions and keeps existing definitions collapsed
  until an administrator chooses one to edit. Its **Roster count mapping**
  assigns every created shift to the Morning, Day, Afternoon, Night, or
  **Not counted** footer total.
- **Staff** creates operational ATCO records and provides a name, watch, and
  role search for existing records.
- **Tools** links to reference data, fatigue rules, manual TOIL, the change log,
  account management, and onboarding.

The selected section is retained in the browser and reflected in the URL
fragment, so returning to `/admin` resumes the last task. **Add shift** and
**Add ATCO** open compact creation panels; they do not save anything until the
form is submitted.

Operational staff records and login accounts are separate concepts. Create
controllers in **Admin → Staff**. Create users who can sign in through
**Accounts**. An operational controller does not consume an account allowance
unless they also have an active login membership.

### Patterns, watches and dated moves

Configure the repeating roster in **Admin → Roster setup**:

1. Build the unit base pattern by selecting `M`, `A`, `D`, `N`, and `OFF` in
   sequence, then choose the date represented by pattern day 1.
2. Select the weekdays on which the unit operates a night duty. An `N` that
   falls on a closed night is displayed as `OFF` without shifting the
   underlying cycle.
3. Add and name the required watches. Every watch has its own day-1 date,
   which phases the inherited unit pattern; a watch may instead define a
   different pattern while retaining that watch-specific phase.

Pattern precedence is **personal ATCO override → effective watch pattern →
unit base pattern**. Personal patterns are configured on the ATCO profile and
should only be enabled for genuine individual arrangements.

The roster footer does not infer custom codes by their first letter. In
**Admin → Shifts → Roster count mapping**, explicitly map each operational
shift to `M`, `D`, `A`, or `N`; map leave, training, standby and support codes
to **Not counted**. The same mapping is used by the on-screen totals, CSV
export, coverage heatmap and publication readiness checks. A pattern letter
does not count unless the airport has created the corresponding working shift,
and Night totals are suppressed on weekdays when night operations are closed.

To move an ATCO, open their Admin profile and use **Schedule a watch move**.
On the effective date, roster generation picks up the new watch and its
pattern automatically. Future and completed moves remain visible and
auditable on the profile.

## Airport onboarding

Unit Admins open `/unit/onboarding` to see a live readiness score based on the
airport's actual configuration. A newly created airport's Unit Admin is sent
to this workflow after login until they explicitly select **Finish setup and
open dashboard**. Normal dashboard routing resumes after that confirmation;
the workflow remains available from **Admin → Tools**. The checklist links
directly to each setup area and covers:

1. Enter airport name/code, timezone, locale, date format, and branding.
2. Configure watches or teams.
3. Create active shift definitions and reusable templates.
4. Add qualification types such as Medical, ADI, APP, APS, OJTI, UCA, and
   English Language.
5. Configure qualification warnings, commonly 180/90/60/30 days.
6. Enter staffing requirements.
7. Configure fatigue rules and the shift-request window/deadline.
8. Import CSV data after reviewing the validation preview.
9. Activate Unit Admin access.
10. Review compliance, publication, restore and go-live acceptance.

Do not go live until the timezone, request deadline, working/non-working codes,
qualifications, staffing requirements, and initial roster have been checked by
two authorised unit users.

## Using the roster

Open a month from **Roster**. Each row represents a person and each column a
date. Requirement indicators show whether coverage meets the configured
minimum for Morning, Day, Afternoon, and Night duties.

### Editing a cell

1. Select a shift code.
2. Review any fatigue, qualification, leave/sickness, roster-lock, or validity
   warning.
3. Apply only an authorised override, recording the reason where required.
4. Optionally select a permitted annotation.

Direct cell editing cannot be used for protected leave, sickness, or TOIL-use
codes; use the dedicated workflow so balances and reports stay consistent.
When a cell is changed to the exact shift requested by a pending or approved
request, the request is retained, marked `fulfilled`, and linked to the
assignment.

### Roster request badges

- Pending requests use a warning-style badge and accessible pending label.
- Approved requests use a distinct success-style badge and accessible
  approved label.
- Rejected, cancelled, and fulfilled requests do not appear as active roster
  badges.

### Monthly roster publication

Every monthly roster has one clear status directly above the roster grid:

- **Draft roster** — the roster may be subject to change.
- **Published roster** — shows the date on which the current roster was
  published.

Unit Admins, Watch Managers and Duty Watch Managers can select **Publish
roster** from the Draft banner. Publication records an immutable snapshot,
captures the publishing manager and time in the audit trail, and notifies the
unit’s other active operational staff.

The same authorised grades can select **Undo publication** beside a Published
status. The month immediately returns to Draft, operational staff are
notified, and the withdrawn publication remains in the audit history rather
than being deleted.

If any roster assignment or annotation changes after publication, the live
month automatically displays as Draft again because it no longer matches the
published snapshot. Publishing the updated month creates a new version and
marks the previous version as superseded. Publication is managed only from the
monthly roster; there is no separate publication centre or acknowledgement
workflow.

## Shift requests

### Default rule

By default, staff can request shifts in the next three calendar months. A
target month locks at the start of the **20th day of the immediately preceding
month**. Unit Admins can configure both the number of future months and lock
day. The server, UI, helper functions, and tests use the same rule.

For example, October requests lock on 20 September. Boundary dates are
inclusive: requests are allowed from the first day of the first configured
future month through the last day of the final configured month, provided the
target month has not locked.

### Creating or updating a request

1. Open **Shift Requests**.
2. Select a date in the permitted window.
3. Choose an active shift explicitly marked as requestable.
4. Add an optional requester comment of up to 500 characters.
5. Save.

A person can have one request per date. Saving again updates that pending
request. A request cannot be edited after it leaves `pending`. Users may only
create requests for themselves.

### Cancelling

Only the requester can cancel their pending, unlocked request. Cancellation
sets the status to `cancelled`; it does not delete the business record.
Approved, rejected, fulfilled, and already-cancelled requests cannot be
removed with a forged form submission.

### Admin decisions

Each pending request has two clear manager actions:

- **Approve and add to roster** creates or updates the assignment immediately.
- **Refuse** records that the request was not accepted.

When decisions are outstanding, Unit Admins see an orange count on the main
**Requests** tab and an attention banner on the Requests page. Both clear
automatically when no pending requests remain.

The manager can add an optional comment to either decision. The requester
receives an in-app notification that explains the outcome and includes that
comment. Internally, an approved-and-rostered request is retained as
`fulfilled` and a refused request as `rejected`, preserving reporting and
audit compatibility.

The requester can select **Delete from my list** for fulfilled or rejected
requests. This removes the completed item from their personal Requests view
without deleting the operational decision or audit history visible to
authorised managers.

Every transition records actor, UTC timestamp, old value, new value, and
reason in the request audit.

## Annotations and TOIL

Unit Admins manage definitions in **Admin → Reference Data**. The guided form
asks first for the code, staff-facing name, group, colour, and explanation.
Suffix, TOIL, permission, note, tag, and sorting controls are contained in the
optional **Advanced behaviour** section. Definitions are scoped to the airport
and include:

- code and display label;
- category, colour, and description/help text;
- active state and sort order;
- allowed suffixes;
- reporting tags;
- optional TOIL half-day value;
- whether a note is required;
- whether Unit Admin permission is required.

Codes must be unique within an airport; another airport may use the same code.
Suffixes are validated on the server. Once a definition has been used, its
code is immutable. “Delete” deactivates the definition so historical entries
remain readable.

Roster Editors may apply or remove permitted annotations but cannot edit
definitions. Watch and deputy watch managers receive only their explicitly
configured permissions. Unit-Admin-only annotations return a permission error
for other roles.

On the roster, the annotation selector returns to **—** after each action and
the applied code appears once as a compact yellow label beneath the shift.
Select the pencil beside that label to add or edit optional detail; the detail
is shown when the code is hovered or keyboard-focused. Use **Remove [code]**
in the selector to clear an applied annotation.

Application/removal and definition changes are audited. TOIL updates and the
annotation assignment occur in one database transaction. Optional transaction
keys make retries idempotent so TOIL is not applied twice.

For bulk changes, prepare the person/date range and preview all affected cells,
notes, permission checks, and TOIL deltas before saving. Never bypass the
preview for a live roster.

Use **Admin → Other settings → Manual TOIL** for a correction or award that is
not already generated by a roster annotation. Select the person, choose
**Add to balance** or **Deduct from balance**, enter days or hours, and record
the reason/reference. The form shows the person’s current balance before the
adjustment.

## Qualifications and compliance

The dashboard at `/compliance` categorises qualifications as:

- missing;
- expired;
- expiring within the warning window;
- valid.

Configure warning periods per qualification type; the standard defaults are
180, 90, 60, and 30 days. Assignment and request-application checks must use
the current airport’s qualification records only.

### Fatigue monitoring

Automatic fatigue findings remain visible on the affected roster cells, with
the rule explanation available from the cell warning. Rest days and any other
shift marked non-working are never displayed as fatigue-warning cells. The
standalone Compliance Centre is not part of the application navigation.

Unit Admins use **Admin → Fatigue rules** (`/admin/fatigue-rules`) to control
roster monitoring:

- review the nine system-rule families and their plain-language names;
- edit airport-specific thresholds including duty hours, rolling periods,
  minimum rest, recovery, consecutive duties/nights, early starts and morning
  duty limits;
- pause or reactivate a system warning and choose whether it is shown as a
  warning or critical finding;
- add local rules using guided checks for maximum duty length, minimum rest,
  consecutive duties, consecutive nights, early-start frequency, or maximum
  hours in a rolling period;
- edit, pause, reactivate, or remove a locally created rule.

Custom rules require a name, limit and severity. Rules measured over time also
require a review period in days. They are evaluated for every operational ATCO
and feed the roster-cell warnings. No code or JSON editing is required.
Changing a rule affects future calculations; it does not rewrite historical
roster assignments.

Fatigue configuration is stored against the authenticated airport’s unit ID.
An administrator cannot read or change another airport’s thresholds, names,
severity choices or custom rules. The dashboard displays the active airport
name and code before any edits are made.

### Production operational assurance

The Unit Admin **Operations** workspace at `/operations/YYYY-MM` provides:

- safety-critical operational positions;
- controller position endorsements with validity and restrictions;
- daily position/shift demand plus a resilience reserve;
- eligible endorsed-controller coverage and shortfalls;
- planned operational breaks;
- achieved duty start/end and planned-versus-actual variance reasons;
- manager review and closure of controller fatigue reports;
- draft, approved and superseded rostering-rule versions.

Rule approval requires a change reference, consultation summary and effective
date. Publication is blocked when the month has no operational positions,
position requirements, break plan or approved rule version; when an endorsed
position is short; or when a high/unfit fatigue report remains open.

The standalone in-app fatigue self-report form is not enabled. Controllers
should use the unit’s established fit-for-duty reporting procedure and Safety
Management System. Automatic roster warnings remain available.

## Coverage and scenario planning

Open `/planning/coverage/YYYY-MM` for a date/shift coverage heatmap. Zero
coverage is highlighted as a critical gap; low coverage is a warning.
Staffing and qualification requirements should be reviewed together before
publication.

Use `/planning/scenarios` to save proposed changes without affecting the live
roster. Give the check a meaningful name, then choose the person, date and
proposed shift. No technical JSON entry is required. The saved result records
whether the person is eligible and why; the live roster remains unchanged.
Applying any automated or scenario change requires a separate human approval
step.

## Leave, sickness, overtime, and reports

- Record leave and sickness through **Leave / Sickness**, not a roster-cell
  shortcut. Unit Admins can add or deactivate airport-specific types there.
  Deactivation preserves history while removing the type from new entries and
  reports.
- Configure Twilio before sending overtime or unit SMS messages.
- **Messages** is available only to Admin, WM and DWM users. It can target one
  person, a named watch, or all active airport users with a custom message or
  a personalised today-shift reminder.
- Reports include fatigue, sickness, leave year, overtime, swaps, extensions,
  and CSV exports.

The **Annotation Totals** report is definition-driven: it creates one column
for every active annotation available in the airport’s roster selector rather
than relying on fixed OT/Swap/EXT codes. If a retired annotation appears in
the selected date range, it remains visible as a historical column so earlier
records are not hidden. The CSV export uses the same columns as the screen.

Twilio environment variables:

```text
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_FROM_NUMBER
```

When they are absent, SMS sending is disabled.

### Calendar subscription

Each user can open **Profile → Calendar subscription** and generate a private
calendar link. Apple and Google subscription actions are shown immediately.
Generating a replacement invalidates the previous link. Calendar clients
authenticate with the unguessable token and therefore do not need an active
browser login; treat the link as confidential.

## Accounts and platform administration

Unit Admins open `/unit/accounts` or select **Accounts** in the header to:

1. Review active accounts against the airport allowance.
2. Select a fully configured roster person and create a one-time invitation
   for the required access level.
3. Deactivate an account that should no longer have access.

The currently signed-in administrator cannot deactivate their own account.
Roster profiles and login access are deliberately separate. Create an ATCO in
**Admin → Staff**, complete their watch, pattern, qualification, leave and
operational settings, and roster them normally. Issue access only when ready,
either from the ATCO profile or **Accounts**. The recipient chooses their
username and password; acceptance links the identity to the existing roster
person without replacing their staff number, watch, pattern or other setup.

Access is activated when the invitation is accepted and capacity is
available. If the allowance has been reached, activation is rolled back and
the page explains that the active-account limit has been reached.

Active-login limits are stored as integers; common plans use 10, 15, 20, 25,
or 30, and custom values are supported.

The limit is enforced transactionally while the airport row is locked.
Invited-but-not-active, disabled invitations, suspended memberships, and
operational people without login access do not count. If the limit is reached,
deactivate an account before activating or restoring another.

`/platform/admin` is restricted to `superadmin`. A Super Admin can:

1. Create an airport using a unique 2–12 character code, display name, plan,
   and active-user limit.
2. Generate an opaque, one-time bootstrap invitation without entering or
   viewing administrator identity details.
3. Change an existing airport's account limit, provided it is not reduced
   below the current active-account count.
4. Suspend or restore an airport.

Transfer the bootstrap link through the approved external secure channel. The
airport administrator enters their own identity and password. The invitation
is hashed at rest, expires, is single-use, rechecks capacity, and exposes only
unused/accepted/expired/revoked status to the Super Admin. The first Unit Admin
must enrol MFA before operational access.

The platform view displays only:

- airport name/code, status, and plan;
- active-account aggregate and limit;
- enabled features;
- database health, migration version, and storage aggregate;
- aggregate activity;
- created, trial-ending, renewal, suspended, and last-active dates.

Plan changes, feature flags, suspension actions, aggregate usage, and platform
changes have dedicated history/audit models. The portal deliberately does not
join operational people, assignments, requests, leave, sickness,
qualifications, or identifying audit content.

### Testing an account limit

Use a non-production airport:

1. Note the active count at `/unit/accounts`.
2. In `/platform/admin`, set the allowance to one more than that count.
3. Sign in as the airport Unit Admin and create one account.
4. Attempt to create another account.
5. Confirm the second account is rejected and the active count has not
   increased.
6. Deactivate the test account and restore the intended allowance.

Never publish example or production passwords in this README. Locally created
demo credentials should be distributed separately and rotated or deleted when
testing is complete.

## Installation and deployment

### Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export DATABASE_URL="sqlite:///instance/roster.db"
flask --app app.py run
```

The development server defaults to `127.0.0.1`. Never use the checked-in
fallback secret in production.

### Supported Python versions

Python 3.12 is the production and CI runtime. A newer Python major version
must not be adopted through an automated dependency update: it requires a
dedicated compatibility pull request that builds the production image and
passes the complete web, PostgreSQL, Redis, worker and desktop test matrix.

Dependency surfaces are deliberately separate:

- `requirements-prod.txt` — production web, database, security and worker;
- `requirements-dev.txt` — production dependencies plus test/security tools;
- `requirements-desktop.txt` — production dependencies plus desktop packaging;
- `requirements.txt` — compatibility entry point containing all surfaces.

Production containers install only `requirements-prod.txt`.

### PostgreSQL production

Production uses one control-plane PostgreSQL database and one physically
separate operational PostgreSQL database per airport. See
[Control-plane and operational database design](docs/control_operational_database_design.md)
for platform MFA, provisioning, signup recovery, migration and rollback.

Use a central control database for identities, memberships, plans, feature
flags, database routing metadata, and aggregate usage. Provision a separate
operational PostgreSQL database for each airport.

Database routing metadata stores a deployment-secret **name**, not a password
or editable URL. Define that secret in the deployment environment, for
example:

```text
CONTROL_DATABASE_URL=postgresql+psycopg://...
ATCROSTER_UNIT_1_DATABASE_URL=postgresql+psycopg://...
ATCROSTER_UNIT_2_DATABASE_URL=postgresql+psycopg://...
REDIS_URL=redis://...
```

The tenant-routed SQLAlchemy session sends every operational mapper to the
database selected by the authenticated membership. `OperationalDatabaseRouter`
accepts only server-side routing metadata whose value is the name of a
deployment secret. The platform-control context is explicitly denied an
operational session. Local legacy development can temporarily use the shared
database only outside production so it can be imported as the first unit.
Never accept a database name, secret name, unit ID, or connection URL from a
form or query parameter.

Run behind an HTTPS reverse proxy and set:

```text
FLASK_SECRET_KEY=<high-entropy deployment secret>
DATABASE_URL=<database URL for the selected deployment role>
ATCROSTER_SECURE_COOKIES=true
ATCROSTER_FIELD_ENCRYPTION_KEY=<Fernet key from the managed secret store>
ATCROSTER_SESSION_IDLE_MINUTES=30
ATCROSTER_SESSION_ABSOLUTE_MINUTES=720
ATCROSTER_TRUSTED_PROXY_HOPS=1
```

Production mode refuses to start with SQLite, the fallback/short Flask secret,
insecure cookies or a missing/invalid field-encryption key. Schema creation and
seed data are disabled at runtime; Alembic exclusively controls the production
schema.

Container deployment files are provided:

```bash
cp .env.example .env
docker compose build
docker compose run --rm migrate
docker compose run --rm web flask --app app bootstrap-platform
docker compose up -d web worker
```

For managed hosting, `railway.toml` configures the Docker build, the explicit
control-and-airport migration orchestrator, Waitress production server and
readiness health check. Create separate Railway services for the web process,
provisioning worker, Redis, control PostgreSQL and every airport PostgreSQL
database. The worker start command is
`python scripts/run_provisioning_worker.py`. Keep all secrets in Railway's
encrypted variables, and follow
`docs/production_runbook.md` for bootstrap and verification.

The Compose port binds to loopback. Place a maintained HTTPS reverse proxy in
front of it; do not expose the application port directly.

Health endpoints:

- `/health/live` — process liveness;
- `/health/ready` — database connectivity and required-schema readiness.

### Native desktop mode

```bash
python desktop_app.py
```

Build on the target operating system:

```bash
pyinstaller --noconfirm --onefile --windowed --name "ATCRoster" desktop_app.py
```

The desktop launcher binds locally. If the native webview is unavailable, it
falls back to the default browser.

## Database migrations and legacy import

### Alembic

Production changes use Alembic:

```bash
export CONTROL_DATABASE_URL="postgresql+psycopg://..."
export DATABASE_URL="$CONTROL_DATABASE_URL"
python scripts/migrate_all_databases.py
```

Set every routed `ATCROSTER_UNIT_<id>_DATABASE_URL` deployment secret first.
The command upgrades the control database, validates every route, refuses a
control/operational URL collision, upgrades each operational database, and
creates its local unit-boundary row.

The tenant-foundation migration creates the first airport, adds tenant keys to
legacy operational tables, adds request lifecycle fields, and extends shift
and annotation definitions. It is intentionally not automatically
downgradable because removing tenant keys would destroy security boundaries.

Older SQLite desktop databases also receive an idempotent compatibility
upgrade at startup. Back up the file first.

### Importing the existing database as the first airport

Create and migrate the target operational database, then run:

```bash
python scripts/import_first_unit.py \
  --source-url sqlite:////absolute/path/to/legacy-roster.db \
  --target-url postgresql+psycopg://user:password@host/database \
  --unit-id 1 \
  --checkpoint .import-first-unit.json
```

The importer:

- copies only columns present in the target;
- assigns every operational row to the selected airport;
- skips primary keys already present;
- checkpoints each completed table so interruption is resumable;
- prints before/after counts for reconciliation.

Review every table count and retain the report with the migration change
record before go-live.

## Security operations

### Account-recovery email

Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`,
`SMTP_FROM_ADDRESS`, and `SMTP_USE_TLS` before enabling recovery in
production. Set `ATCROSTER_SUPPORT_EMAIL` to the monitored Super Admin mailbox.
Every person should have a unique registered email in **Admin → People**.
Recovery responses deliberately do not reveal whether an account or address
exists. Approval and reset links are single-use and expire automatically.

- CSRF tokens protect shift-request and annotation-definition changes.
- IDs, dates, status values, comments, codes, and suffixes are validated on
  the server.
- Login clears prior session state before establishing identity.
- `next` redirects must be local paths.
- Tenant context comes from the authenticated membership.
- Cross-airport reads, writes, references, and admin actions return no data.
- Passwords use Werkzeug’s password hashing; never store plaintext passwords.
- Password changes require the current password, CSRF validation and a new
  password of at least 12 characters.
- Cookies are HTTP-only and SameSite=Lax; production must enable secure
  cookies.
- Responses set clickjacking, MIME-sniffing, referrer and browser-permission
  protections, with HSTS on HTTPS.
- Invitation tokens must be random, store only a digest, expire, and be
  single-use.
- Add MFA challenge enforcement at the identity layer before enabling
  privileged production access.
- Apply rate limits at the reverse proxy and application layer for login,
  invitation, password reset, export, and write endpoints.
- Emit structured security events without personnel data into the platform
  health plane.

Do not log secrets, passwords, MFA seeds, request comments, medical details,
or database URLs. Review operational audit access as personnel data.

## Backup, restore, and recovery

Back up the control database and each airport database independently using
encrypted, access-controlled storage. Record database identifier, migration
version, start/end time, size, checksum, and outcome—never row contents—in the
control-plane health record.

Test restores regularly:

1. Provision an isolated target.
2. Restore control and one selected airport database.
3. Apply the recorded migration version.
4. compare table counts and checksums;
5. run tenant-isolation and smoke tests;
6. destroy the isolated environment securely.

Do not restore one airport over another airport’s database route.

## Testing

Install dependencies and run:

```bash
python -m pytest -q
```

The suite covers physical database routing and platform-control denial, five
legacy migration fixtures, legacy helpers/routes, shared UX/error recovery, roster
zoom controls, overtime empty states, platform airport creation,
account-limit enforcement, unit account management, request persistence,
request windows and lock boundaries, requestable shifts,
one-request-per-date, pending updates, forged deletion, status validation,
approve-only, approve-and-apply, qualification conflicts, CSRF,
audit/notifications, and cross-unit read/write isolation.

### Repeatable acceptance dataset

Create a clean, date-relative test platform with Leeds Bradford, East
Midlands and Inverness airports:

```bash
python scripts/seed_acceptance_data.py --reset
export DATABASE_URL="sqlite:///instance/acceptance.db"
export FLASK_SECRET_KEY="local-acceptance-secret"
flask --app app.py run --port 5001
```

The generated `instance/acceptance.manifest.json` lists the local test
credentials and rolling month values. It is ignored by Git and must never be
used in production. The seed command creates:

- three isolated airport tenants and a platform-control account;
- 42 operational ATCOs across four watches;
- Unit Admin, Roster Editor, Watch Manager, Duty Watch Manager and Staff User
  accounts;
- four complete rolling roster months;
- leave, sickness, requests, notifications, qualifications, annotations and
  TOIL examples;
- operational positions, endorsements, position requirements, break plans,
  achieved duty, fatigue reports and governed rules;
- scenarios, a previous published roster and an acknowledgement;
- exactly one spare account at each airport for a quick account-limit test.

The full ordered acceptance procedure and result sheets are in
[docs/manual_acceptance_test.md](docs/manual_acceptance_test.md). Re-run the
seed command to return to a known baseline between test cycles.

The latest implemented-role access and isolation evidence is recorded in
[docs/permission_test_report_2026-07-24.md](docs/permission_test_report_2026-07-24.md).

Before release also run:

```bash
python -m compileall -q app.py tenancy.py saas_models.py account_limits.py scripts
python scripts/scale_assurance.py
pip-audit -r requirements-prod.txt
```

For production, add integration tests against PostgreSQL and the deployment’s
secret manager, backup service, email/SMS provider, and reverse proxy.

See [SECURITY.md](SECURITY.md) for production requirements and known gaps, and
[docs/pilot_readiness.md](docs/pilot_readiness.md) for the recommended pilot
acceptance and evidence plan.

Deployment, monitoring, backup, restore, incident and upgrade procedures are
in [docs/production_runbook.md](docs/production_runbook.md).

The accountable-manager product assessment and remaining operational roadmap
are recorded in [docs/atc_manager_review.md](docs/atc_manager_review.md).

Release assurance is collected in:

- [security report](docs/release_security_report_2026-07-25.md);
- [permission matrix](docs/permission_matrix.md);
- [data classification](docs/data_classification.md);
- [migration runbook](docs/migration_runbook.md);
- [backup/restore runbook](docs/backup_restore_runbook.md);
- [production release checklist](docs/release_checklist.md).

## Troubleshooting

### A shift is absent from the request list

Confirm it belongs to the signed-in airport and is both active and explicitly
marked `is_requestable`.

### A request month is locked earlier or later than expected

Check the airport’s request lock day. The default is the 20th of the
immediately preceding month, not two months before.

### Approve and apply shows a conflict

Review fatigue flags, required qualifications, shift state, roster lock, and
airport ownership. Confirm an override only when unit policy permits it and
record a clear reason.

### An annotation code cannot be renamed

Used codes are immutable to preserve historical meaning. Deactivate the old
definition and create a new code.

### An account cannot be activated

The active-login limit has been reached. Suspended and inactive accounts do
not count; safely deactivate an active membership before adding another.

### Database connection fails

Check that routing metadata references the correct deployment-secret name and
that the secret exists. Never copy credentials into the tenant-editable
database.

### Tests report missing packages

Activate the intended virtual environment and reinstall:

```bash
python -m pip install -r requirements-dev.txt
```
