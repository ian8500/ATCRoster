# Backup and restore operations

ATCRoster requires a separate logical backup for the control database and
every airport operational database. Docker volumes and a successful backup job
are not proof of recoverability.

The scheduled `database-backup.yml` workflow is the production mechanism. It
streams custom-format dumps directly into GPG encryption, stores an off-Railway
artifact, records checksums/schema versions and triggers freshness incidents.
The private recovery key must remain outside GitHub, Railway and this repository.

## Operator tooling

Use environment variables for URLs so credentials are not written into command
history or metadata:

```bash
python scripts/backup_databases.py --output /secure/off-host/staging \
  --database control control CONTROL_DATABASE_URL \
  --database airport-iwld operational AIRPORT_IWLD_DATABASE_URL
python scripts/verify_backup.py control-YYYYMMDDTHHMMSSZ.dump \
  control-YYYYMMDDTHHMMSSZ.json
```

The staging directory must be encrypted and access controlled. Move the
verified recovery set to approved encrypted off-host storage, then securely
remove the staging copy according to the retention policy.

Restore only into a newly created, isolated PostgreSQL database:

```bash
python scripts/restore_database.py airport-iwld-YYYYMMDDTHHMMSSZ.dump \
  airport-iwld-YYYYMMDDTHHMMSSZ.json \
  --target-url-env RESTORE_DATABASE_URL \
  --confirm RESTORE-INTO-EMPTY-DATABASE
```

The tool refuses a non-empty target, verifies size and SHA-256, validates the
archive with `pg_restore --list`, restores with owner/ACL portability, and
confirms the restored Alembic revision. Afterward, use the matching release to
run health, tenant-isolation and customer acceptance checks before any cutover.

## Retention and recovery objectives

The accountable operator must approve numeric RPO/RTO and retention periods.
Current daily GitHub artifacts do not provide point-in-time recovery or the
proposed long-term monthly archive. Recovery-key custody requires two separate,
access-controlled copies and a quarterly decryption/restore rehearsal.

See `docs/backup_restore_runbook.md` for the current production workflow and
`docs/operations/disaster-recovery.md` for the recovery sequence.
