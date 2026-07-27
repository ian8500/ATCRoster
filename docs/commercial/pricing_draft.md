# Pricing and packaging — decision draft

The safest early model is a fixed monthly fee per airport with an included user
allowance. It is predictable for customers and aligns with the existing
airport-level account limit. Avoid per-SMS inclusion: pass messaging through at
cost or sell a separately capped allowance.

## Proposed validation prices (excluding VAT)

| Package | Suitable for | Included active users | Proposed monthly | Proposed setup |
|---|---|---:|---:|---:|
| Unit | Small non-24-hour unit | 25 | £195 | £750 |
| Unit Plus | Medium/24-hour unit | 75 | £395 | £1,500 |
| Enterprise | Large/multi-function unit | custom | from £695 | scoped |

Annual prepayment could receive up to a 10% discount after cashflow and churn
are understood. Pilot pricing should be a time-limited written offer, not a
permanent hidden tier.

## Included baseline

- isolated airport account and configured user limit;
- roster, requests, leave/sickness, reports and authorised messaging;
- standard onboarding materials and UK business-hours support;
- routine updates, security fixes and customer data export;
- published security/privacy information.

## Separately priced

- data migration/cleansing;
- bespoke integrations or reports;
- on-site training and travel;
- 24/7 P1 on-call;
- additional storage/retention;
- SMS/provider consumption;
- custom contractual/security assessment; and
- multi-airport group reporting when implemented and assured.

## Cost validation before approval

For each tier calculate Railway databases/web/worker/Redis, independent backup,
monitoring, email, support time, Twilio use, payment fees, insurance, tax,
security testing and legal/accounting overhead. Target a gross margin that
funds support and assurance; do not price only against Railway's current hobby
bill.

Owner decisions: VAT status, currency/term, minimum contract, setup/refund
rules, overage price, annual uplift, pilot conversion and public/private price.
