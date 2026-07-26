# ATCRoster manual acceptance test

Use this script to test a release from the perspective of platform support,
unit management, roster editors and operational ATCOs. It is deliberately
ordered so later tests can rely on changes made earlier.

## Test record

| Field | Record |
| --- | --- |
| Tester | |
| Application version / commit | |
| Browser and version | |
| Device / operating system | |
| Dataset generation date | |
| Test started | |
| Test completed | |
| Overall result | Pass / Fail / Blocked |

For every failure, record the test ID, what happened, what was expected, a
screenshot, the account used and the time. Do not include passwords, MFA
secrets, medical details or free-text fatigue content in defect reports.

## Prepare the acceptance environment

These commands create a dedicated SQLite acceptance database. The command
refuses to overwrite an existing database unless `--reset` is present.

```bash
python -m pip install -r requirements-dev.txt
python scripts/seed_acceptance_data.py --reset
export DATABASE_URL="sqlite:///instance/acceptance.db"
export FLASK_SECRET_KEY="local-acceptance-secret"
flask --app app.py run --port 5001
```

Open `http://127.0.0.1:5001`. The generated credentials and current test
months are in `instance/acceptance.manifest.json`. This manifest is local test
material and must not be committed or used in production.

The dataset contains:

- Leeds Bradford (`LBA`): 16 ATCOs, 17-account limit;
- East Midlands (`EMA`): 14 ATCOs, 15-account limit;
- Inverness (`INV`): 12 ATCOs, 13-account limit;
- a Platform Super Admin, Unit Admin, Roster Editor and Staff User login;
- four rolling months of complete assignments (previous, current, next and
  two months ahead);
- watches, shifts, leave, sickness, TOIL, overtime annotations, requirements,
  qualifications, position endorsements, break plans, achieved duty, fatigue
  reports, scenarios, requests, notifications and a previous publication.

Re-run the seed command whenever a clean baseline is needed. Use the month
values in the manifest rather than fixed dates.

## Result key

- **Pass**: behaviour and stored result match the expected outcome.
- **Fail**: behaviour is incorrect or data is lost, disclosed or corrupted.
- **Blocked**: an earlier defect prevents the test.
- **Not run**: test has not yet been attempted.

## A. Service and authentication

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| A01 | Open `/health/live`. | HTTP 200 and JSON status `ok`. | |
| A02 | Open `/health/ready`. | HTTP 200 and JSON status `ready`. | |
| A03 | Open `/login` in a private browser window. | Login page loads over the expected origin with no authenticated data visible. | |
| A04 | Submit an invalid username and password. | Generic invalid-credentials message; no account details disclosed. | |
| A05 | Sign in as the LBA Staff User from the manifest. | Roster opens and the header clearly says Leeds Bradford Airport and `LBA`. | |
| A06 | Sign out, then use the Back button. | Protected content is not usable; navigation returns to login. | |
| A07 | Sign in as the LBA Unit Admin and open `/security/mfa`. Enrol an authenticator and save the recovery codes. | Setup succeeds and the recovery codes are shown once. | |
| A08 | Sign out and sign in again using the authenticator code. | MFA challenge succeeds and the code cannot be replayed. | |
| A09 | Sign out and use one recovery code. Repeat with the same recovery code. | First use succeeds; second use is rejected. | |
| A10 | Resize to a narrow/mobile viewport and navigate the main pages. | Header, airport context, navigation and forms remain readable without hiding critical actions. | |
| A11 | At a narrow viewport, open and close Menu; press Escape while it is open. | Menu is keyboard accessible, closes on Escape and returns focus to its button. | |
| A12 | Tab from the top of a page and activate “Skip to main content”. | Focus moves directly to the page content. | |

Reset the dataset after A07–A09 if later testers need an unenrolled account.

## B. Tenant isolation and role access

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| B01 | As LBA Staff User, inspect roster, profile, requests, fatigue and published pages. | Only LBA people and records are visible. | |
| B02 | Copy an LBA staff-profile URL. Sign out, sign in as EMA Staff User and open it. | LBA personnel data is not disclosed; access is denied or not found. | |
| B03 | As EMA Staff User, attempt `/admin`, `/unit/accounts`, `/operations/<current-month>` and `/reports`. | Administrative areas are denied or safely redirected. | |
| B04 | As EMA Roster Editor, open roster, overtime, leave/sickness, reports and compliance. | Editor tools are available; account and Unit Admin configuration remain unavailable. | |
| B05 | As EMA Unit Admin, open all unit areas. | Full EMA unit administration is available; no LBA/INV data appears. | |
| B06 | As Platform Super Admin, open `/platform/admin`. | Only airport/service aggregates appear; no names, rosters, requests, fatigue or medical records are exposed. | |
| B07 | As Platform Super Admin, try operational URLs. | Operational personnel and roster functions are unavailable. | |
| B08 | In each airport account, verify the airport name/code in the header before editing. | Context always matches the login’s airport and cannot be changed by URL parameters. | |

## C. Platform administration and account limits

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| C01 | As Platform Super Admin, review LBA, EMA and INV cards. | Plan, limit, status, health, migration and aggregate usage display correctly. | |
| C02 | Change INV’s account limit, plan or status, then reload. | Change persists and a safe platform audit event is recorded. | |
| C03 | Create a new test airport with a unique code, plan and limit. Securely transfer the one-time bootstrap link. | No personal fields are requested. The recipient chooses identity/password, the link becomes accepted, and MFA is required before operational access. | |
| C04 | Attempt to create an airport using an existing code. | Validation rejects the duplicate without partial records. | |
| C05 | As LBA Unit Admin, open Accounts. | Usage reads `16 of 17`. | |
| C06 | Create one LBA account with a unique username and 12+ character password. | Account activates and usage reads `17 of 17`. | |
| C07 | Attempt to create one more LBA account. | Creation is blocked by the active-account limit and no partial account remains. | |
| C08 | Deactivate the account created in C06, then create another. | Capacity returns and replacement creation succeeds. | |
| C09 | Attempt to deactivate the account currently signed in. | Self-deactivation is refused. | |
| C10 | Reset the dataset. | All airports return to the documented baseline. | |

## D. Airport onboarding and reference data

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| D01 | As LBA Unit Admin, open Airport onboarding. | Readiness is complete and links open the correct setup areas. | |
| D02 | Edit LBA identity fields and save, then restore them. | Header and onboarding reflect saved values; other airports are unchanged. | |
| D03 | Open Admin and switch between Overview, Requirements, Shifts, Staff and Tools. | Sections are clear, retain selection and do not submit hidden forms. | |
| D04 | Add a shift `TST`, then edit its times/requestable state. | Shift appears once and saved values are used by relevant forms. | |
| D05 | Try a duplicate shift code and malformed times. | Clear validation; no duplicate or partial shift. | |
| D06 | Add an operational test ATCO, edit their profile and move them to another watch with an effective date. | Person and dated watch history persist. | |
| D07 | Edit and then delete the watch-history record created in D06. | Roster/watch view follows the current valid history; audit remains available. | |
| D08 | Open Reference Data and inspect A6, OT, TOAI and TOA8. | Labels, categories, colours, suffix and TOIL behaviour match the seed definitions. | |
| D09 | Add a note-required annotation and use it on a roster cell without a note, then with a note. | Missing note is rejected; valid use succeeds and is audited. | |
| D10 | Attempt to deactivate an annotation after it has been used. | Historical meaning is preserved and the UI explains the permitted action. | |

## E. Monthly roster and editing

Use the manifest’s `test_month`.

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| E01 | Open the LBA current-month roster. | All 16 people and every day have explicit assignments; watch grouping and M/D/A/N coverage appear. | |
| E02 | Navigate previous/next months. | Month boundaries, weekdays and data remain correct. | |
| E03 | Search/inspect several people and compare their profile assignments. | Names, watches and shifts agree across views. | |
| E04 | As LBA Roster Editor, change an unprotected future cell to a valid shift. Reload. | Change persists with manual source/audit and only the chosen cell changes. | |
| E05 | Apply an OT annotation and valid A6 suffix. | Badge/colour and annotation totals update correctly. | |
| E06 | Enter an invalid shift or annotation through a modified request. | Server rejects the value; roster is unchanged. | |
| E07 | Attempt direct overwrite of annual leave, sickness or TOIL-protected data. | Protected workflow prevents an unsafe overwrite or requires the authorised path. | |
| E08 | Create a fatigue-producing sequence and review the warning. Cancel, then repeat with an authorised override/reason if offered. | Warning is explainable; cancel preserves data; authorised override is recorded. | |
| E09 | Export roster CSV. | File downloads, contains LBA only, correct month/dates and no HTML. | |
| E10 | Open print view / print preview. | Landscape output is legible and includes airport/month context. | |
| E11 | Open the public calendar URL from a staff profile in a signed-out window. | Valid token returns only that person’s permitted roster window; altered token fails. | |
| E12 | Select 75%, 90%, 100% and Fit width, then revisit the roster. | Each preset is applied, the grid can scroll at larger sizes and the chosen preset persists. | |

## F. Shift requests

Use the manifest’s `request_month`.

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| F01 | As LBA Staff User, open Requests. | Seeded pending/approved/rejected examples display with clear status and manager response where applicable. | |
| F02 | Submit a valid requestable shift on an eligible date. | One pending request is stored with the comment and confirmation. | |
| F03 | Edit the pending request’s shift/comment. | Existing request updates; no duplicate is created. | |
| F04 | Try a second request on the same date. | One-request-per-person/date rule is enforced. | |
| F05 | Try a non-requestable/unknown shift, past date and date outside the window. | Each is rejected server-side with no stored request. | |
| F06 | Cancel the request. | Status becomes cancelled, audit is retained and active badge disappears. | |
| F07 | As LBA Unit Admin, approve a pending request without applying it. | Status becomes approved, response/audit/notification are recorded, assignment is unchanged. | |
| F08 | Approve and apply another request. | Matching assignment is created/updated, request becomes fulfilled and links to it. | |
| F09 | Attempt approve-and-apply where leave, sickness, qualification or fatigue conflicts. | Unsafe application is blocked and original data remains intact. | |
| F10 | Sign in as the requester and inspect notifications/roster. | Outcome is visible only to the correct user and airport. | |

## G. Leave, sickness, TOIL and overtime

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| G01 | Open Leave / Sickness for LBA. | Seeded leave and sickness periods appear with correct people/dates. | |
| G02 | Add, edit and remove a future annual-leave period. | Assignments update safely; reports and profile agree; removal restores expected roster behaviour. | |
| G03 | Add sickness with self-cert/certified code and inspect the sickness report. | Period, roster code and report agree without affecting other staff. | |
| G04 | Attempt overlapping or end-before-start leave/sickness. | Validation rejects invalid range with no partial changes. | |
| G05 | Add half-day and full-day TOIL through the dedicated workflow. | Balance changes by the correct half-day units and audit/report reflect it. | |
| G06 | Inspect the seeded OT annotation in metrics. | Overtime total includes the seeded entry exactly once. | |
| G07 | Run Overtime Finder for an operational shift. | Candidates exclude unavailable, opted-out, unqualified or fatigue-conflicted people as applicable. | |
| G08 | If SMS is unconfigured, attempt notification. | Clear safe configuration message; no false success and no secret disclosure. | |

## H. Qualifications and compliance

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| H01 | Open Qualification compliance. | MED, ADI and OJTI rows show; one medical is within 30 days and states are clear. | |
| H02 | Inspect the seeded expired legacy radar validation on a profile. | Expired state is prominent and does not appear valid. | |
| H03 | Add/edit a qualification expiry and reload. | New state and warning band calculate from the actual date. | |
| H04 | Open Compliance Centre for current month. | Fatigue findings are grouped/explained and scoped to LBA. | |
| H05 | Export compliance evidence CSV. | Correct airport/month, headers and findings; no other airport data. | |
| H06 | Compare a finding against the person’s roster sequence. | Dates and duties support the explanation. | |

## I. Operational assurance and fatigue

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| I01 | Open Operations for current month. | TWR/GMC/APP, endorsements, position requirements, break, achieved duty, reports and rule v1 display. | |
| I02 | Add a position and grant an endorsement with valid dates. | Both persist and are airport-scoped. | |
| I03 | Add/update a position requirement for a date/shift. | Assurance recalculates eligible count and shortfall. | |
| I04 | Set a requirement above endorsed staffing. | Shortfall becomes visible and blocks publication. Restore the original value. | |
| I05 | Add a break with end before start, then a valid break. | Invalid period is rejected; valid period persists. | |
| I06 | Record achieved duty with a variance reason. | Actual times and variance persist; invalid negative duration is rejected. | |
| I07 | As Staff User, submit low, high and unfit fatigue reports. | Reports are stored with clear immediate-reporting guidance. | |
| I08 | As Unit Admin, review/close the reports with a meaningful response. | Reviewer/time/status persist; high/unfit remains a publication blocker until closed. | |
| I09 | Create draft rule v2 with valid JSON and governance evidence. | Draft is created without changing approved v1. | |
| I10 | Try malformed JSON and approval without evidence. | Both are rejected safely. | |
| I11 | Approve rule v2 with effective date. | v2 is approved, v1 superseded and audit evidence retained. | |

## J. Coverage, scenarios and publication

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| J01 | Open Coverage heatmap for the current month. | M/D/A/N coverage and shortfall colours agree with roster requirements. | |
| J02 | Open Scenarios and inspect “Summer traffic uplift”. | Scenario changes display without modifying the live roster. | |
| J03 | Create a scenario with two changes, then apply/approve only if the UI supports the required authority. | Preview is isolated; authorised application is audited and explicit. | |
| J04 | Open Published for the previous month. | Version 1 and release information display; one seeded acknowledgement exists. | |
| J05 | As an unacknowledged Staff User, acknowledge previous version. | Acknowledgement records once; repeat does not duplicate it. | |
| J06 | Open current-month publication centre. | Preflight reports configuration, competence, coverage, position, break, fatigue and acknowledgement information. | |
| J07 | Introduce an unassigned cell, position shortfall or open high fatigue report and attempt publication. | Publication is hard-blocked. Restore the baseline. | |
| J08 | Attempt publication without release declaration. | Publication is refused. | |
| J09 | If only soft exceptions remain, use a short rationale then a 20+ character rationale. | Short rationale is refused; adequate rationale plus declaration publishes version 1. | |
| J10 | Change a roster cell and publish again. | Prior version becomes superseded; version 2 is immutable and current. | |
| J11 | As Staff User, acknowledge current version and inspect notification. | Acknowledgement and notification state persist for that person only. | |

## K. Reports, audit and data quality

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| K01 | Open Reports, leave-year, sickness and metrics as LBA Unit Admin. | Totals reconcile with seeded roster, leave, sickness, TOIL and OT records. | |
| K02 | Compare one person’s profile, calendar, roster and leave-year report. | Identity and dates agree in every view. | |
| K03 | Open the change log after the preceding edits. | Actor, entity, change, month and reason are clear and correctly ordered. | |
| K04 | Mark notifications read and reload. | Read state persists and another user’s notifications are unchanged. | |
| K05 | Repeat selected report/export checks in EMA and INV. | Counts differ as expected (14 and 12 people) and tenant data never mixes. | |
| K06 | Enter text containing `<script>`, quotes and a very long value in safe test fields. | Output is escaped, lengths enforced and the page remains functional. | |
| K07 | Submit a stale/missing CSRF token on a state-changing form. | Request is rejected and no data changes. | |

## L. Resilience and release checks

| ID | Action | Expected result | Result |
| --- | --- | --- | --- |
| L01 | Restart the server and sign in again. | All committed test changes persist; sessions behave safely. | |
| L02 | Open two different airport sessions in separate browser profiles and make independent edits. | Each remains bound to its own airport. | |
| L03 | Rapidly double-submit a request/account/annotation form. | No duplicate business records or 500 error. | |
| L04 | Test a 28/29/30/31-day month and year boundary using roster navigation. | All days and month links are correct. | |
| L05 | Run `python -m pytest -q`. | Entire automated suite passes. | |
| L06 | Run `python -m compileall -q app.py tenancy.py saas_models.py account_limits.py scripts`. | No compile errors. | |
| L07 | Rebuild a fresh acceptance database and compare manifest counts. | 3 airports, 42 operational staff, 9 requests and four rolling assignment months are recreated. | |
| L08 | Review browser console and server log during the script. | No uncaught errors, secrets, passwords or sensitive free text in logs. | |
| L09 | Verify backup/restore and HTTPS/reverse-proxy procedures in a production-like environment. | Runbook evidence is captured; restored service passes health, login and isolation smoke tests. | |

## Exit criteria

Release only when:

- every priority safety, security, tenant-isolation and data-integrity test
  passes;
- no open defect can create an incorrect published roster, conceal a
  competence/fatigue/coverage issue or disclose another airport’s data;
- all automated tests and production migration checks pass;
- Unit Manager, operational safety representative and technical owner sign
  the test record;
- remaining lower-priority defects have an owner, risk decision and target
  date.

Acceptance of this script is product validation, not regulatory approval. The
unit remains responsible for its approved rostering rules, fatigue risk
management, competence scheme, operational change process and local safety
assurance.
