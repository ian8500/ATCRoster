# Assignment baseline/override migration assumptions

Migration `20260803_42` preserves the legacy `assignment.code` value and
classifies every existing displayed assignment conservatively.

- `source=auto` with note `pattern` or `generated watch coverage` becomes the
  generated baseline.
- Manual changes and approved shift requests become editor overrides.
- Leave and sickness values become absence overrides.
- Any unrecognised provenance becomes an override marked
  `MIGRATED_UNCERTAIN`; it is never treated as disposable generated data.

The legacy `code` column remains populated during the compatibility rollout.
Run `python scripts/report_assignment_migration.py --details` after migration
to review uncertain assignment identifiers without displaying staff names or
shift details. Use `--fail-on-uncertain` in assurance automation when required.
