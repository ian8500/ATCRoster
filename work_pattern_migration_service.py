"""Safe, explicit migration from legacy CSV cycles to normalised patterns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable


@dataclass(frozen=True)
class LegacyPatternMigrationRow:
    staff_id: int
    staff_name: str
    source: str
    legacy_cycle: tuple[str, ...]
    legacy_anchor: date
    status: str
    explanation: str
    pattern_id: int | None = None
    pattern_name: str | None = None

    @property
    def can_migrate(self) -> bool:
        return self.status == "exact_match"


@dataclass(frozen=True)
class LegacyPatternMigrationResult:
    effective_from: date
    rows: tuple[LegacyPatternMigrationRow, ...]

    @property
    def exact_count(self) -> int:
        return sum(row.can_migrate for row in self.rows)


@dataclass(frozen=True)
class WorkPatternMigrationDependencies:
    db: Any
    Staff: Any
    WorkPattern: Any
    WorkPatternDay: Any
    ShiftType: Any
    StaffPatternAssignment: Any
    pattern_context: Callable[[Any, date], tuple[list[str], date]]
    pattern_service: Any


class WorkPatternMigrationService:
    def __init__(self, dependencies: WorkPatternMigrationDependencies) -> None:
        self.dependencies = dependencies

    def analyse(self, unit_id: int, effective_from: date) -> LegacyPatternMigrationResult:
        signatures = self._normalised_signatures(unit_id)
        rows = []
        staff_rows = self.dependencies.Staff.query.filter_by(
            unit_id=unit_id
        ).order_by(self.dependencies.Staff.name, self.dependencies.Staff.id).all()
        for staff in staff_rows:
            existing = self._effective_assignment(unit_id, staff.id, effective_from)
            if existing:
                rows.append(LegacyPatternMigrationRow(
                    staff.id, staff.name, "normalised", (), effective_from,
                    "already_migrated", "A normalised pattern already applies on this date.",
                ))
                continue
            cycle, anchor = self.dependencies.pattern_context(staff, effective_from)
            signature = tuple(str(code).strip().upper() for code in cycle if str(code).strip())
            source = "personal override" if staff.pattern_override else (
                "watch pattern" if staff.watch_id else "unit pattern"
            )
            matches = signatures.get(signature, ())
            if not signature:
                status, explanation = "invalid_legacy", "The effective legacy cycle is empty or invalid."
                pattern = None
            elif len(matches) == 1:
                status, explanation = "exact_match", "Exact ordered cycle match; safe to migrate."
                pattern = matches[0]
            elif len(matches) > 1:
                status = "ambiguous"
                explanation = "More than one normalised pattern has this cycle. Select manually."
                pattern = None
            else:
                status = "no_match"
                explanation = "No exact normalised cycle exists; legacy fallback will remain active."
                pattern = None
            rows.append(LegacyPatternMigrationRow(
                staff.id, staff.name, source, signature, anchor, status, explanation,
                getattr(pattern, "id", None), getattr(pattern, "name", None),
            ))
        return LegacyPatternMigrationResult(effective_from, tuple(rows))

    def migrate_exact(
        self, unit_id: int, effective_from: date, staff_ids: set[int]
    ) -> tuple[Any, ...]:
        report = self.analyse(unit_id, effective_from)
        eligible = {
            row.staff_id: row for row in report.rows
            if row.can_migrate and row.staff_id in staff_ids
        }
        if eligible.keys() != staff_ids:
            raise ValueError(
                "Selection changed or includes a non-exact match. Run the dry-run again."
            )
        created = []
        for staff_id in sorted(eligible):
            row = eligible[staff_id]
            assignment = self.dependencies.StaffPatternAssignment(
                unit_id=unit_id,
                staff_id=staff_id,
                work_pattern_id=row.pattern_id,
                effective_from=effective_from,
                anchor_date=row.legacy_anchor,
                anchor_day_index=0,
                notes="Migrated from exact legacy CSV cycle match.",
            )
            self.dependencies.pattern_service.validate_staff_pattern_assignment(assignment)
            self.dependencies.db.session.add(assignment)
            created.append(assignment)
        self.dependencies.db.session.flush()
        return tuple(created)

    def _effective_assignment(
        self, unit_id: int, staff_id: int, on_date: date
    ) -> Any | None:
        model = self.dependencies.StaffPatternAssignment
        return model.query.filter(
            model.unit_id == unit_id,
            model.staff_id == staff_id,
            model.effective_from <= on_date,
            (model.effective_to.is_(None) | (model.effective_to >= on_date)),
        ).first()

    def _normalised_signatures(self, unit_id: int) -> dict[tuple[str, ...], tuple[Any, ...]]:
        shifts = {
            row.id: row.code.upper()
            for row in self.dependencies.ShiftType.query.filter_by(unit_id=unit_id).all()
        }
        grouped: dict[tuple[str, ...], list[Any]] = {}
        patterns = self.dependencies.WorkPattern.query.filter_by(
            unit_id=unit_id, is_active=True
        ).all()
        for pattern in patterns:
            days = self.dependencies.WorkPatternDay.query.filter_by(
                unit_id=unit_id, work_pattern_id=pattern.id
            ).order_by(self.dependencies.WorkPatternDay.day_index).all()
            if len(days) != pattern.cycle_length_days:
                continue
            signature = []
            for day in days:
                if day.day_type == "OFF":
                    signature.append("OFF")
                elif day.day_type == "FIXED_SHIFT" and day.fixed_shift_type_id in shifts:
                    signature.append(shifts[day.fixed_shift_type_id])
                else:
                    break
            else:
                grouped.setdefault(tuple(signature), []).append(pattern)
        return {key: tuple(value) for key, value in grouped.items()}
