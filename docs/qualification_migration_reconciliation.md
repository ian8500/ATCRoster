# Qualification migration reconciliation

Revision `20260725_06` establishes `QualificationType`,
`PersonQualification` and append-oriented qualification history as the
authoritative competence source.

| Legacy field | New code | Reconciliation rule |
| --- | --- | --- |
| `medical_expiry` | `MEDICAL` | Create valid record with the same expiry when populated |
| `tower_ut` / `tower_ue_expiry` | `ADI` | Create valid record when either legacy value demonstrates the qualification |
| `radar_ut` / `radar_ue_expiry` | `APS` | Create valid record when either legacy value demonstrates the qualification |
| `met_ut` / `met_ue_expiry` | `MET` | Create valid record when either legacy value demonstrates the qualification |
| `has_ojti` | `OJTI` | Create non-expiring valid record when true |
| `has_assessor` | `ASSESSOR` | Create non-expiring valid record when true |

The migration also creates inactive-empty-ready definitions for `APP`, `UCA`
and `ENGLISH_LANGUAGE` without inventing person qualifications. Inserts use
unit/person/type existence guards, so a resumed import cannot duplicate a
record. Existing assignments and requests are not rewritten.

Reconciliation evidence:

1. Capture per-table counts before migration.
2. Run `python scripts/migrate_all_databases.py`.
3. Confirm staff, assignment and request counts are unchanged.
4. Count each non-null/true legacy field and compare with the corresponding
   new records, accounting for people represented by both flag and expiry.
5. Review any difference before enabling qualification-based shifts.

The automated minimal, full-original, partial, clean and historical fixtures
all preserve their input counts and reach revision `20260725_08`.
