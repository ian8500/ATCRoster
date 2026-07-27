# Initial Data Protection Impact Assessment — ATCRoster

**Status:** draft for controller and privacy-adviser approval

**Owner:** Ian John Dickson, IDAviation

**Date:** 27 July 2026

**Review trigger:** new data category, automated decision, provider/region,
large customer, integration, security incident or material workflow change

## 1. Processing

ATCRoster is a hosted multi-airport workforce rostering service. Airport units
create users, watches, patterns and shifts; manage leave/sickness,
qualifications, requests, overtime and communications; and review fatigue and
staffing warnings. Data is entered by authorised users and processed by the
application, airport-specific databases and approved infrastructure providers.

## 2. Necessity and proportionality

The processing supports safe staffing and workforce administration. Tenant and
role boundaries, configurable types, limited platform metadata and absence of
clinical diagnosis fields reduce scope. Fatigue and staffing outputs are
decision support; accountable humans retain decisions. Customers must document
lawful bases, special-category conditions, workforce transparency and why less
intrusive records would not meet their obligations.

## 3. People affected

ATCOs, trainees, watch/duty managers, roster editors, administrators and other
authorised unit staff. Employment imbalance, safety-sensitive roles and
health-related records increase potential impact.

## 4. Risk assessment

| Risk | Initial | Controls | Residual |
|---|---|---|---|
| Cross-airport disclosure | High | authenticated tenant binding, airport databases, tests, privacy-safe super-admin | Medium |
| Excess access to sickness/medical data | High | role limits, minimal status data, reports, special-category policy, audit | Medium |
| Account takeover | High | strong password hashing, MFA, secure recovery, rate limits, secure cookies | Medium |
| Incorrect roster/fatigue result affects decisions | High | visible warnings/reasons, human review, publication status, audit, tests | Medium |
| SMS exposes sensitive or wrong-recipient data | High | authorised roles, verified contacts, sender control, audit, content policy | Medium |
| Data loss/outage | High | database separation, backups, restore rehearsal, monitoring/runbooks | Medium pending independent backup |
| Excess retention/free text | High | retention schedule, field limits, minimisation guidance, deletion process | Medium pending automation |
| Provider/international transfer | Medium | DPA, subprocessor register, regions, transfer mechanism/assessment | Medium pending contract evidence |
| Privileged misuse | High | restricted platform view, MFA, audit, access review, leaver process | Medium |
| Workers unaware/unable to exercise rights | Medium | public notice, controller workforce notice, rights procedure | Low/medium |

## 5. Outstanding measures before wider launch

- Approve customer-specific retention and implement deletion automation.
- Establish independently recoverable backups and record a successful restore.
- Confirm Railway/email/backup regions, DPAs and transfer safeguards.
- Complete external penetration/security testing and remediate material issues.
- Make privileged MFA mandatory and complete quarterly access reviews.
- Validate fatigue/qualification logic through operational assurance.
- Obtain controller sign-off, including Article 6/9 bases and workforce notice.
- Record consultation with representative users/worker representatives or
  explain why it was not appropriate.

## 6. Decision

Pilot processing may continue only within documented scope and controls.
Commercial expansion should not be approved until the outstanding high-impact
measures have owners and evidence. Any residual high risk that cannot be reduced
requires specialist advice and potential prior consultation with the ICO.

Approval — Controller privacy lead: __________ Date: ____

Approval — Controller operational/safety lead: ______ Date: ____

Approval — IDAviation product/security owner: _______ Date: ____
