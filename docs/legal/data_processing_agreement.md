# ATCRoster Data Processing Agreement

**Parties:** `[CUSTOMER LEGAL NAME]` (Controller) and Ian John Dickson trading
as IDAviation, of Flat 0/2, 24 Caird Drive, Glasgow, Scotland, G11 5DT
(Processor).

**Effective date:** `[DATE]`

**Main agreement:** `[CUSTOMER AGREEMENT / ORDER FORM]`

## 1. Scope and precedence

This DPA applies where Processor processes personal data for Controller in
providing ATCRoster. It forms part of the Main Agreement. Data protection terms
take precedence for that processing. Defined UK GDPR terms have their statutory
meaning.

## 2. Controller instructions

Processor shall process personal data only on Controller's documented
instructions, including the Main Agreement, configuration and authorised use of
the service, unless UK law requires otherwise. Processor shall notify Controller
before legally required processing unless prohibited by law and shall inform
Controller if an instruction appears to infringe applicable data protection law.

Controller is responsible for its lawful bases, special-category condition,
workforce transparency, data accuracy, authorised users, retention decisions and
lawfulness of instructions and communications.

## 3. Confidentiality and security

Processor shall ensure authorised personnel are bound by confidentiality and
shall maintain measures appropriate to risk, including:

- logical separation of airport tenants and physically separate operational
  databases where configured;
- least-privilege role controls and controlled privileged administration;
- TLS in transit, managed encryption at rest, password hashing and encrypted MFA
  secrets;
- MFA for privileged access, secure sessions, CSRF protection and rate limits;
- audit trails for material roster, request, account and SMS activity;
- protected secrets, vulnerability/dependency management and change control;
- resilient backups, restore testing, monitoring and incident procedures; and
- staff access removal and periodic access review.

Security measures may evolve provided protection is not materially reduced.

## 4. Subprocessors

Controller gives general written authorisation for the subprocessors published
at `/subprocessors`. Processor shall:

1. bind each subprocessor to materially equivalent data-protection duties;
2. remain responsible for its subprocessor's performance;
3. give at least 30 days' notice of an intended new subprocessor where
   practicable; and
4. allow Controller to object on reasonable data-protection grounds.

If the parties cannot resolve an objection, Controller may stop the affected
optional feature or terminate the affected service under the Main Agreement.

## 5. International transfers

Processor shall not make a restricted transfer without a lawful mechanism,
including the UK International Data Transfer Agreement/Addendum where
appropriate, and any required transfer-risk assessment and supplementary
measures. Controller authorises transfers inherent in approved subprocessors,
subject to these safeguards.

## 6. Data-subject requests

Taking account of the nature of processing, Processor shall provide reasonable
technical and organisational assistance for Controller to respond to rights
requests. Processor shall promptly forward requests received directly and shall
not respond substantively unless instructed or legally required.

## 7. Assistance and incidents

Processor shall assist Controller, taking account of available information, with
security, breach assessment/notification, DPIAs and regulatory consultation.
Processor shall notify Controller without undue delay after confirming a
personal-data breach affecting Controller data and provide known:

- nature, affected data/people and likely consequences;
- containment and remediation measures;
- incident contact and continuing material updates.

Notification is not an admission of fault. Controller is responsible for
regulatory and data-subject notifications.

## 8. Return and deletion

On termination or Controller's written request, Processor shall provide an
agreed export and delete or return personal data, unless law requires retention.
Residual backup data will be isolated from ordinary use and expire through the
documented backup cycle. Controller must request its export before the agreed
closure deadline.

## 9. Evidence and audit

Processor shall provide information reasonably necessary to demonstrate Article
28 compliance. No more than annually, unless following a material incident or
regulatory requirement, Controller may audit through current independent
reports, questionnaires and then a scoped inspection if those are insufficient.
Audits require reasonable notice, confidentiality, security safeguards and
minimal disruption. Controller bears its costs unless material non-compliance is
found.

## Schedule 1 — processing details

| Item | Description |
|---|---|
| Subject matter | Hosted multi-tenant ATC roster, staffing, availability, communications and reporting service |
| Duration | Main Agreement plus export/closure and backup-expiry period |
| Nature/purpose | Store, organise, display, calculate, communicate, audit, secure, back up and support Controller's roster operations |
| Data subjects | ATCOs, managers, administrators, trainees, other authorised unit personnel and support contacts |
| Ordinary data | Identity, staff number, username, contact details, unit/watch/role, rosters, requests, annotations, leave, TOIL, overtime, communications, audit and technical records |
| Special-category data | Sickness/absence information and medical validity or related health information entered by Controller |
| Frequency | Continuous during service use |
| Controller rights | Configuration, role management, access, correction, export, retention instructions and termination/deletion |

## Signatures

Controller: __________________ Name/title: __________________ Date: ______

Processor: ___________________ Name/title: __________________ Date: ______
