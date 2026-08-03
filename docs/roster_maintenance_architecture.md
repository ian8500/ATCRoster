# Deterministic roster maintenance architecture

## Baseline, override and effective assignment

`Assignment.generated_code` is the deterministic baseline derived from dated
workforce records. `Assignment.override_code` records editor intent and is
never changed by ordinary recalculation. `effective_code` resolves override,
then baseline, then the legacy materialised value during the compatibility
transition. Roster screens, totals, exports, coverage and fatigue logic consume
the effective value.

Recalculation is idempotent. Explicit `OFF` rows are generated, generation
metadata identifies the pattern and event, and unchanged baselines are not
rewritten. Overrides are classified as valid, redundant, after leaving,
outside employment, conflicting with a hard restriction, or requiring review.
The default `Unit.preserve_redundant_overrides` policy retains redundant values
until operational evidence supports safe cleanup.

## Whole-month protection and periods

`Unit.protected_roster_months_ahead` defaults to two. The current month plus
that number of following whole calendar months are protected; automatic
maintenance begins on day one of the next month. The calculation is based on
the airport timezone and works across year and leap-year boundaries.

`RosterPeriod` records generation method/version and a derived status of
`CURRENT`, `PROTECTED`, `FUTURE_AUTOMATIC` or `HISTORICAL`. A persisted
`CLOSED` status overrides the calculation. Automatic population skips closed
periods. Only the explicit admin rebuild can deliberately enter the protected
range.

## Effective-dated workflows

- Joiners store employment, unit-join and roster-start dates, watch history,
  pattern alignment, hours and initial qualifications. Protected dates create
  exceptions; automatic dates receive baseline rows.
- First and additional UEs are dated qualifications. First UE changes coverage
  contribution from its true date without rewriting assignment placement.
- Leavers retain history, stop operational contribution after their dated
  boundary, generate `OFF` baselines in the automatic horizon and flag duties,
  leave, swaps and overrides that need protected-period review.
- Watch transfers close the prior history row and create a dated destination
  record with destination-watch or selected-cycle alignment.
- Full-time, part-time and pattern changes close the prior pattern period and
  create a new dated arrangement containing weekly hours, anchor and reason.

`OperationalCapabilityService` deliberately separates roster placement from
coverage contribution. It evaluates unit/employment dates, roster-active
state, medical validity, independent UEs and qualification suspensions for the
date being counted.

## Events, exceptions and transactions

Every automatic or manual recalculation creates a `RosterImpactEvent` with its
scope, protected/automatic ranges, counts, warnings and completion state.
`RosterImpactException` is the editor queue with `OPEN`, `ACKNOWLEDGED`,
`RESOLVED` and `NOT_APPLICABLE` states. Failed critical processing rolls back
the workforce and partial baseline transaction, then records a clean `FAILED`
audit row; success is never fabricated.

The preview at `/roster-impact/preview` performs a dry-run calculation and
shows the effective boundary, calculated baseline changes, existing assignments
and overrides to preserve. `/roster-impact/exceptions` provides the queue and
recent recalculation history. Protected rebuild requires Unit Admin permission,
a reason and the explicit `REBUILD` confirmation.

## Commands and Railway scheduling

Run future generation locally with:

```bash
flask roster ensure-future-periods
```

Set `ROSTER_GENERATION_MONTHS_AHEAD=18` (the default) to control the horizon.
The command may be scoped during support/testing with `--unit-code CODE` and
`--months-ahead N`. It creates missing periods, populates only new automatic
periods, records `FUTURE_PERIOD_CREATED`, preserves overrides and is idempotent.

For Railway, create a scheduled service from the same repository/image and use
the command above. Supply the same database and encryption secrets as the web
service, set the generation horizon variable, and schedule it monthly before
the unit's planning cycle. Do not run a second migration command in the job;
the normal pre-deploy migration remains authoritative. Alert on a non-zero exit
and inspect failed events in the impact queue.

## Migration assumptions and limitations

Alembic migrations remain role-aware: operational tables are not created in a
control-only database. Legacy assignment values remain materialised while all
readers complete the effective-code transition. Existing CSV/watch patterns
remain a fallback for airports not yet migrated to normalised patterns.

Current limitations:

- future generation is synchronous; large estates should run it as the
  scheduled Railway job rather than from a request;
- the queue is intentionally editor-managed and does not yet send escalation
  notifications;
- redundant overrides are preserved by default and require evidence before a
  unit opts into automatic cleanup;
- a closed period can be changed only through an explicit admin process; there
  is no bulk reopen workflow.
