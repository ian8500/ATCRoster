# Briefing module rollback

The briefing feature is isolated from roster assignments, staff records and
roster publications.

## Immediate rollback

Disable the `briefing_module` feature flag for the airport in Platform Admin.
This removes the module selector, navigation links and access to all briefing
routes. Roster behaviour and data are unchanged.

## Code and schema rollback

1. Back up the airport operational database and inventory the private briefing
   bucket.
2. Disable `briefing_module`.
3. Downgrade Alembic from `20260728_21` to `20260727_20`.
4. Deploy the earlier application revision.

The downgrade removes only:

- `briefing_item`
- `briefing_delivery`
- `briefing_audit`
- `briefing_assurance_run`

Uploaded objects are intentionally retained during database downgrade so an
accidental rollback does not destroy controlled documents. After confirming
that the module will not be restored, the airport prefix in the configured
private bucket may be archived or deliberately removed under the approved
retention process. Local development uses the directory beneath
`ATCROSTER_BRIEFING_UPLOAD_DIR`.
