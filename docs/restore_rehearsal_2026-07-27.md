# Pilot staging restore rehearsal — 27 July 2026

## Outcome

The control-plane and airport operational databases in `pilot-staging` were
streamed as PostgreSQL custom-format backups and restored into separate
databases in a disposable local PostgreSQL 18 container. The rehearsal passed.
No production database was accessed and no persistent backup file was written
to the workstation.

## Evidence

| Check | Result |
| --- | --- |
| Control backup stream checksum | SHA-256 calculated |
| Operational backup stream checksum | SHA-256 calculated |
| Control restore | Pass |
| Operational restore | Pass |
| Control public tables | 16 |
| Operational public tables | 31 |
| Control Alembic revision | `20260727_15` |
| Operational Alembic revision | `20260727_15` |
| Measured rehearsal duration | 40 seconds |
| Temporary restore target | Destroyed after verification |

Checksums were verified as part of the in-memory stream and are deliberately
not retained in the repository. They identify an ephemeral test backup, not a
retained recovery set.

## Finding

The workstation's PostgreSQL 14 client cannot dump the PostgreSQL 18 Railway
databases. The successful rehearsal used the official `postgres:18-alpine`
client image. Recovery procedures must pin a client major version equal to or
newer than the database server.

## Remaining launch controls

- Enable Railway daily, weekly and monthly volume backup schedules for both
  database volumes. The current CLI login can inspect the volumes but was not
  authorised to alter their backup schedules.
- Approve numeric production RPO and RTO targets.
- Repeat the rehearsal from a retained Railway backup before production use.
- Configure an off-platform encrypted backup if the approved recovery plan
  requires recovery outside the same Railway project and environment.
