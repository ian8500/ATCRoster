# Permission test report — 24 July 2026

## Outcome

**Pass with one documented product limitation.**

All currently implemented runtime permission levels passed the read, write,
tenant-isolation and denial tests after remediation:

- Platform Super Admin;
- Unit Admin;
- Roster Editor;
- Watch Manager;
- Duty Watch Manager;
- Staff User.

The `ReadOnlyAuditor` name in the product vision is not currently
provisionable by the account workflow and therefore was not represented as a
runtime role in this test. It must not be offered commercially until a
separate read-only permission contract, account option and regression matrix
are implemented.

## Test environment

| Item | Value |
| --- | --- |
| Database | Local rolling acceptance database |
| Airports | Leeds Bradford, East Midlands, Inverness |
| Automated suite | 38 passed |
| Role-matrix personas | 6 |
| Permission routes per persona | 18 |
| Tenant-isolation airports | 3 |
| Test date | 24 July 2026 |

## Permission matrix

`R` means readable/self-service, `W` means permitted write authority, and `—`
means server-side 403 denial.

| Capability | Super Admin | Unit Admin | Editor | WM / DWM | Staff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Platform administration | W | — | — | — | — |
| Monthly roster | — | W | W | W | R |
| Shift requests | — | W | R | R | R |
| Published rosters / acknowledgement | — | W | R | R | R |
| Personal fatigue reporting | — | R | R | R | R |
| Overtime finder | — | W | W | — | — |
| Leave / sickness management | — | W | W | — | — |
| Metrics | — | R | R | — | — |
| Qualification compliance | — | R | R | — | — |
| Fatigue compliance centre/export | — | R | — | — | — |
| Operational assurance | — | W | — | — | — |
| Coverage heatmap | — | R | R | R | — |
| Scenario planning | — | W | W | W | — |
| Account management | — | W | — | — | — |
| Airport onboarding | — | W | — | — | — |
| Unit administration/reference data | — | W | — | — | — |

## Remediated findings

### P1 — Platform control-plane escape

The Platform Super Admin could reach roster, requests, publications,
qualification and coverage URLs by typing them directly, despite those links
being hidden.

**Fix:** a central server-side control-plane allowlist now restricts the
Platform Super Admin to platform administration, own-password/MFA, logout,
static assets and health endpoints. `/` redirects to Platform Admin.

### P1 — Unit personnel-data exposure

An ordinary Staff User could directly open the unit-wide qualification
compliance page. They could also open the coverage heatmap even though it was
not part of their self-service role.

**Fix:** qualification data now requires Unit Admin or Roster Editor.
Coverage requires roster-edit authority (Admin, Editor, WM or DWM).

### P2 — Ambiguous permission denial

Restricted overtime, leave, reports and metrics URLs silently redirected an
unauthorised user to the roster, making access denial appear to be a broken
link.

**Fix:** restricted direct access now returns the consistent, recoverable 403
screen.

### P1 — Roster write CSRF gap

Direct roster-cell and monthly AI-generation posts did not validate CSRF
tokens.

**Fix:** both endpoints now require a valid token, and every roster write form
includes it.

## Write and isolation evidence

The regression suite verifies:

- Staff User cannot write roster cells;
- Watch Manager and Duty Watch Manager can write roster cells;
- Roster Editor can write roster cells and scenarios but cannot administer;
- Unit Admin can perform unit configuration, publication, operational
  assurance and account actions;
- Platform Super Admin can administer airport accounts but cannot read
  operational pages;
- Staff User can view only their own profile;
- Unit Admin can view another person in the same airport;
- an ID belonging to another airport returns 404 rather than disclosing its
  existence;
- forged cross-airport request and operational-position writes fail;
- account limits remain transactional;
- request approval/application, qualification conflicts, notifications and
  audit records remain enforced;
- missing CSRF tokens fail without changing data.

## Manual test accounts

The acceptance database uses the password recorded in
`instance/acceptance.manifest.json`.

| Permission | Leeds account |
| --- | --- |
| Platform Super Admin | `platform.admin` |
| Unit Admin | `lba.admin` |
| Roster Editor | `lba.editor` |
| Watch Manager | `lba.atco03` |
| Duty Watch Manager | `lba.atco04` |
| Staff User | `lba.atco01` |

Repeat the same checks with `ema.*` and `inv.*` accounts when conducting a
formal release acceptance cycle.

## Release decision

The implemented permission model is suitable for continued acceptance
testing. Do not enable a ReadOnly Auditor account or advertise that role until
its explicit read-only routes, export policy, privacy scope and tests have
been delivered.
