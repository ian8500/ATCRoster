# Backup and restore runbook

## Backup

1. Use an automated PostgreSQL custom-format backup for the control database
   and every airport operational database.
2. Encrypt in transit and at rest; store off-host with least-privilege access.
3. Record database identifier, UTC timestamp, Alembic revision, byte size and
   SHA-256 checksum without personnel content in the job log.
4. Alert on failure or missed recovery-point objective.
5. Retain and securely erase backups according to the approved retention plan.

Before launch, the accountable operator must approve numeric targets. Template:

- RPO: no more than ___ minutes of committed control or airport data loss.
- RTO: control access restored within ___ hours; each airport restored within
  ___ hours in the documented priority order.
- Backup frequency: control every ___; each airport every ___.
- Restore rehearsal owner and frequency: ___ (at least quarterly).

Example:

```bash
pg_dump --format=custom --no-owner --file=atcroster.dump "$DATABASE_URL"
sha256sum atcroster.dump > atcroster.dump.sha256
```

Use a `pg_dump` client whose major version is equal to or newer than the
database server. Railway currently provisions PostgreSQL 18 for this project;
the official `postgres:18-alpine` image provides a reproducible recovery
client when the operator workstation has an older client installed.

### Current automated recovery set

The **Encrypted database backup** GitHub Actions workflow runs daily at 02:30
UTC and can also be started manually. It:

- retrieves production database connection details at runtime from Railway;
- streams PostgreSQL 18 custom-format dumps without writing plaintext dumps;
- encrypts each dump to the checked-in recovery public key;
- records the database label, timestamp, Alembic revision, encrypted byte size,
  SHA-256 checksum and recovery-key fingerprint; and
- retains the encrypted GitHub artifact for 30 days.

The private recovery key is not stored in GitHub, Railway, the repository or
the workflow artifact. The accountable operator must keep at least two
access-controlled copies in separate locations and test decryption quarterly.
Rotating the key requires retaining the old private key until every artifact
encrypted to it has expired.

This provides an off-Railway logical recovery copy. It does not provide
point-in-time recovery. Railway volume snapshots and PITR remain unavailable
on the current plan; reassess a Pro-plan upgrade before committing to a
customer RPO that requires either capability. GitHub artifact retention covers
the proposed 30 daily copies, not the proposed 12 monthly copies. Configure an
approved long-term object store before representing monthly retention as met.

## Restore rehearsal

1. Authorise an isolated PostgreSQL target and restrict network access.
2. Verify the backup checksum.
3. Restore with `pg_restore --clean --if-exists`.
4. Set the restored control and airport secret references, then run
   `python scripts/migrate_all_databases.py` using the matching release image.
5. Check health endpoints and compare control totals.
6. Run tenant-isolation, authentication, roster, publication and audit-history
   acceptance checks.
7. Record achieved RPO/RTO and discrepancies.
8. Securely destroy the rehearsal environment.

Restore tests must occur at least quarterly and before a migration with
material tenant or publication impact. A successful backup job is not evidence
of recoverability until this rehearsal passes.
