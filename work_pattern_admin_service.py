"""Administrative mutations and standard seeds for flexible work patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from work_pattern_service import WorkPatternService


STANDARD_SIX_ON_FOUR_OFF = "Standard 6-on/4-off"
STANDARD_FOUR_ON_SIX_OFF = "Part-time 4-on/6-off"


@dataclass(frozen=True)
class WorkPatternAdminDependencies:
    db: Any
    WorkPattern: Any
    WorkPatternDay: Any
    WorkPatternDayAllowedShift: Any
    ShiftType: Any
    pattern_service: WorkPatternService


class WorkPatternAdminService:
    def __init__(self, dependencies: WorkPatternAdminDependencies) -> None:
        self.dependencies = dependencies

    def seed_standard_patterns(self, unit_id: int) -> tuple[Any, ...]:
        """Create missing standard patterns without altering existing rows."""
        shifts = {
            row.code.upper(): row
            for row in self.dependencies.ShiftType.query.filter_by(
                unit_id=unit_id, is_active=True, is_working=True
            ).all()
        }
        missing = [code for code in ("M", "A", "N") if code not in shifts]
        if missing:
            raise ValueError(
                "Create active working shift types for "
                + ", ".join(missing)
                + " before adding the standard patterns."
            )
        created: list[Any] = []
        if not self._pattern_named(unit_id, STANDARD_SIX_ON_FOUR_OFF):
            created.append(self._create_pattern(
                unit_id,
                STANDARD_SIX_ON_FOUR_OFF,
                "M, M, A, A, N, N followed by four protected days off.",
                [
                    ("FIXED_SHIFT", shifts[code], ())
                    for code in ("M", "M", "A", "A", "N", "N")
                ] + [("OFF", None, ())] * 4,
            ))
        if not self._pattern_named(unit_id, STANDARD_FOUR_ON_SIX_OFF):
            allowed = tuple(
                shift for code, shift in shifts.items() if code in {"M", "D", "A"}
            )
            if not allowed:
                allowed = (shifts["M"], shifts["A"])
            created.append(self._create_pattern(
                unit_id,
                STANDARD_FOUR_ON_SIX_OFF,
                "Four flexible working days followed by six protected days off.",
                [("WORK_ALLOWED_SET", None, allowed)] * 4
                + [("OFF", None, ())] * 6,
                contracted_minutes=4 * 8 * 60,
            ))
        return tuple(created)

    def replace_pattern_days(
        self,
        pattern: Any,
        day_specs: Iterable[dict[str, Any]],
    ) -> None:
        specs = list(day_specs)
        pattern.cycle_length_days = len(specs)
        rows = []
        for index, spec in enumerate(specs):
            row = self.dependencies.WorkPatternDay(
                unit_id=pattern.unit_id,
                work_pattern_id=pattern.id,
                day_index=index,
                day_type=spec["day_type"],
                fixed_shift_type_id=spec.get("fixed_shift_type_id"),
                required_work=bool(spec.get("required_work")),
                notes=str(spec.get("notes") or "")[:500],
            )
            rows.append(row)
        self.dependencies.pattern_service.validate_pattern(pattern, rows)
        existing_ids = [
            row.id for row in self.dependencies.WorkPatternDay.query.filter_by(
                unit_id=pattern.unit_id, work_pattern_id=pattern.id
            ).all()
        ]
        if existing_ids:
            self.dependencies.WorkPatternDayAllowedShift.query.filter(
                self.dependencies.WorkPatternDayAllowedShift.unit_id == pattern.unit_id,
                self.dependencies.WorkPatternDayAllowedShift.work_pattern_day_id.in_(
                    existing_ids
                ),
            ).delete(synchronize_session=False)
        self.dependencies.WorkPatternDay.query.filter_by(
            unit_id=pattern.unit_id, work_pattern_id=pattern.id
        ).delete(synchronize_session=False)
        self.dependencies.db.session.flush()
        for row, spec in zip(rows, specs, strict=True):
            self.dependencies.db.session.add(row)
            self.dependencies.db.session.flush()
            for shift_type_id in spec.get("allowed_shift_type_ids", ()):
                self.dependencies.db.session.add(
                    self.dependencies.WorkPatternDayAllowedShift(
                        unit_id=pattern.unit_id,
                        work_pattern_day_id=row.id,
                        shift_type_id=int(shift_type_id),
                    )
                )

    def _pattern_named(self, unit_id: int, name: str) -> Any | None:
        return self.dependencies.WorkPattern.query.filter_by(
            unit_id=unit_id, name=name
        ).first()

    def _create_pattern(
        self,
        unit_id: int,
        name: str,
        description: str,
        specs: list[tuple[str, Any | None, tuple[Any, ...]]],
        *,
        contracted_minutes: int | None = None,
    ) -> Any:
        fixed_minutes = sum(
            _shift_minutes(shift) for day_type, shift, _ in specs
            if day_type == "FIXED_SHIFT"
        )
        pattern = self.dependencies.WorkPattern(
            unit_id=unit_id,
            name=name,
            description=description,
            cycle_length_days=len(specs),
            contracted_minutes_per_cycle=(
                fixed_minutes if contracted_minutes is None else contracted_minutes
            ),
            is_active=True,
        )
        self.dependencies.db.session.add(pattern)
        self.dependencies.db.session.flush()
        self.replace_pattern_days(pattern, [
            {
                "day_type": day_type,
                "fixed_shift_type_id": fixed.id if fixed else None,
                "allowed_shift_type_ids": tuple(shift.id for shift in allowed),
                "required_work": day_type not in {"OFF", "OPTIONAL_WORK"},
            }
            for day_type, fixed, allowed in specs
        ])
        return pattern


def _shift_minutes(shift: Any | None) -> int:
    if not shift or not shift.start_time or not shift.end_time:
        return 8 * 60
    start = shift.start_time.hour * 60 + shift.start_time.minute
    end = shift.end_time.hour * 60 + shift.end_time.minute
    if end <= start:
        end += 24 * 60
    return end - start
