# ATCRoster baseline retention schedule

This is a proposed service baseline, not an automatic legal answer. Each
airport/ATS controller must approve periods against employment, aviation,
medical, safety-management, insurance, limitation and public-record duties.
Health information must not be retained merely because storage is available.

| Record | Proposed active retention | Disposal/action | Owner |
|---|---:|---|---|
| Active identity, role and contact | Account life | Delete/anonymise after closure and export window | Controller |
| Disabled account security stub | 12 months | Delete identity; retain minimal audit reference if necessary | Joint operational process |
| Rosters and assignments | 7 years after roster month | Delete/anonymise unless safety/legal hold applies | Controller to approve |
| Shift requests and decisions | 3 years after decision | Delete comments/audit unless dispute requires longer | Controller |
| Leave records | Current leave year + 3 years | Delete/anonymise | Controller |
| Sickness/health-related absence | 3 years after record/return, subject to legal need | Secure deletion; periodic necessity review | Controller |
| Medical/qualification validity | Active engagement + 3 years | Delete/anonymise unless regulatory duty requires longer | Controller |
| SMS audit content | 12 months | Delete content; aggregated delivery totals may be anonymised | Controller |
| In-app notifications | Read + 90 days; unread + 12 months | Delete | Controller |
| Change/security audit records | 24 months | Delete or irreversibly anonymise | Processor/controller by record |
| Recovery/invitation tokens | Expiry or use + 30 days metadata | Delete token; retain outcome/security event only | Processor |
| Application logs | 30–90 days | Rolling deletion; never intentionally log health/comments/secrets | Processor |
| Support cases | Closure + 3 years | Delete/anonymise attachments and personal content | IDAviation controller |
| Customer contract/billing | Contract + 7 years | Secure deletion subject to tax/legal duties | IDAviation controller |
| Production backups | 30 daily + 12 monthly proposed | Cryptographic/secure deletion by lifecycle | Processor |

## Controls

- Apply legal holds narrowly, document the reason and review at least quarterly.
- Review special-category data at least annually and on staff departure.
- Remove data from live systems promptly; allow documented backup expiry.
- Record deletion jobs and investigate failures without copying deleted content
  into logs.
- Customer-specific schedules must be configuration/documented instructions,
  not informal support requests.
