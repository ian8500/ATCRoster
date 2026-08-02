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
