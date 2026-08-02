# Flexible roster engine design

## Stage 1 foundation

The normalised pattern engine is an additive operational-domain capability.
Existing unit, watch and staff CSV patterns remain unchanged and continue to
drive roster generation until an integration stage explicitly opts a workflow
into the new resolver. Creating the new tables therefore cannot rewrite an
assignment, change a published snapshot or alter historical roster output.

All pattern and staff-rule rows are airport scoped. Composite foreign keys bind
staff, shift, pattern and pattern-day references to the same airport. The tables
are migrated only in operational databases; the control database stores no
staff restrictions or working-pattern details.

## Pattern-day calculation

`StaffPatternAssignment` is effective dated. The applicable row is the latest
assignment whose inclusive date range contains the requested date. Its cycle
index is calculated as:

```text
(anchor_day_index + requested_date - anchor_date) modulo cycle_length_days
```

This supports dates before and after the anchor and prevents later pattern
changes from altering earlier resolutions. Overlapping assignments are rejected
by the service before persistence. PostgreSQL exclusion constraints are not used
because the project supports SQLite for development and tests; the same service
validation must be used by every write interface.

Every pattern day uses a controlled day-type value. Allowed shift sets are rows
in `WorkPatternDayAllowedShift`, not free text. `weekdays_mask` on a staff rule
is a seven-bit integer where bit zero is Monday and bit six is Sunday.

## Hard and soft rules

Hard rules return an ineligible result with stable reason codes and human-readable
explanations. Soft rules never make an otherwise legal assignment eligible or
ineligible; they add their configured penalty and structured explanation. Leave
is treated as a hard blocker. Existing fatigue, medical, qualification and
endorsement checks remain authoritative and will be composed with this service
in the roster-validation stage rather than duplicated here.

Count and minutes rules use `rolling_period_days`. `maximum_count` stores either
the maximum duty count or maximum minutes according to `rule_type`. This shared
typed representation avoids adding a new staff column for each future rule.

## Compatibility and next integration boundary

The first consumer will be the roster-validation service. CSV migration, pattern
administration, fairness calculations and automatic proposals are separate
stages. No optimiser may write live assignments directly; generated duties will
remain proposal records until an authorised user accepts them.

## Phase 2 administration

Unit administrators manage normalised patterns from **Administration → Flexible
work patterns**. The standard seed is idempotent: it creates missing 6-on/4-off
and part-time 4-on/6-off examples but never overwrites a unit's existing rows.
Units must configure active working `M`, `A`, and `N` shift types before using
the seed, keeping all foreign keys local to that operational database.

Staff profiles link to a separate effective-dated configuration screen. A new
assignment is rejected when it overlaps another assignment for that person.
Rules retain start and optional end dates; deactivation preserves their history.
The 28-day preview uses the same resolver used by eligibility decisions rather
than duplicating cycle arithmetic in the template.

Once a pattern has been assigned, its cycle definition is immutable. It may be
retired from future assignment, but structural changes require a replacement
pattern and a new effective-dated staff assignment. This prevents an edit today
from changing historical roster interpretation. The legacy CSV editor remains
available as a labelled fallback until a normalised assignment is effective.

## Phase 3 roster validation

The monthly roster now evaluates existing working assignments against effective
normalised patterns and staff rules. Pattern mismatches and hard-rule breaches
are publication blockers. Soft-rule breaches are clearly marked preferences:
they remain advisory and never prevent publication or alter an assignment.

Validation is read-only. It adds cell-level explanations and a pre-publication
summary, but does not move, replace or generate duties. The publication handler
locks the month before validating and refuses to create a snapshot while any
blocking finding remains, so bypassing the disabled browser button cannot
publish a roster that fails these checks.

Count and contracted-minute checks distinguish a proposed duty from an existing
assignment. Existing duties are measured once: reaching a configured maximum is
valid, while exceeding it is a blocker. Proposal eligibility continues to
include the candidate duty when testing the limit.
