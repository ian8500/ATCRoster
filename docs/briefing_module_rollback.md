# Briefing module rollback

The briefing feature is isolated from roster assignments, staff records and
roster publications.

## Immediate rollback

Disable the `briefing_module` feature flag for the airport in Platform Admin.
This removes the module selector, navigation links and access to all briefing
routes. Roster behaviour and data are unchanged.

## Code and schema rollback

1. Back up the airport operational database and inventory the private briefing
   object store or durable mounted directory.
2. Disable `briefing_module`.
3. Prefer a schema-compatible application rollback. Do not run an improvised
   Alembic downgrade: later migrations contain briefing schema and data changes
   beyond the original module tables.
4. If schema rollback is required, restore the control and all airport
   databases from the same verified pre-deployment recovery set, then deploy
   the matching earlier application revision as described in the database
   migration runbook.

Briefing objects are outside the database backup. Retain and inventory them
through rollback so controlled documents are not silently lost. After
confirming that the module will not be restored, the airport prefix may be
archived or deliberately removed only under the approved retention process.
Development may use `ATCROSTER_BRIEFING_UPLOAD_DIR`; production requires
complete private S3-compatible configuration or
`ATCROSTER_BRIEFING_DURABLE_DIR` on an explicitly provisioned durable mount.
