from dataclasses import replace
from datetime import date, time

import pytest

import app
from roster_population_service import (
    GENERATION_VERSION,
    DeterministicRosterPopulationService,
    PopulationDependencies,
)


@pytest.fixture()
def population_service():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit.query.filter_by(code="RPOP").first()
        if not unit:
            unit = app.Unit(
                code="RPOP",
                name="Roster Population Test",
                protected_roster_months_ahead=2,
                onboarding_step=100,
            )
            app.db.session.add(unit)
            app.db.session.flush()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.StaffPatternAssignment.query.filter_by(unit_id=unit.id).delete()
        day_ids = [
            row.id for row in app.WorkPatternDay.query.filter_by(unit_id=unit.id)
        ]
        if day_ids:
            app.WorkPatternDayAllowedShift.query.filter(
                app.WorkPatternDayAllowedShift.work_pattern_day_id.in_(day_ids)
            ).delete(synchronize_session=False)
        app.WorkPatternDay.query.filter_by(unit_id=unit.id).delete()
        app.WorkPattern.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        app.ShiftType.query.filter_by(unit_id=unit.id).delete()

        shifts = {}
        for code, start, end, working in (
            ("M", time(6), time(14), True),
            ("A", time(14), time(22), True),
            ("OFF", None, None, False),
        ):
            shift = app.ShiftType(
                unit_id=unit.id,
                code=code,
                name=code,
                start_time=start,
                end_time=end,
                is_working=working,
            )
            app.db.session.add(shift)
            app.db.session.flush()
            shifts[code] = shift
        person = app.Staff(
            unit_id=unit.id,
            username="roster_population_person",
            password_hash="unused",
            name="Roster Population Person",
            staff_no="RP-001",
            role="user",
            membership_status="active",
            is_operational=True,
        )
        app.db.session.add(person)
        app.db.session.flush()
        app.db.session.commit()

        service = DeterministicRosterPopulationService(PopulationDependencies(
            db=app.db,
            Unit=app.Unit,
            Staff=app.Staff,
            Assignment=app.Assignment,
            ShiftType=app.ShiftType,
            WorkPattern=app.WorkPattern,
            WorkPatternDay=app.WorkPatternDay,
            WorkPatternDayAllowedShift=app.WorkPatternDayAllowedShift,
            StaffPatternAssignment=app.StaffPatternAssignment,
            utcnow=app.utcnow,
            legacy_code_resolver=lambda _person, _day: "A",
        ))
        yield service, unit, person, shifts

        app.db.session.rollback()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.StaffPatternAssignment.query.filter_by(unit_id=unit.id).delete()
        day_ids = [
            row.id for row in app.WorkPatternDay.query.filter_by(unit_id=unit.id)
        ]
        if day_ids:
            app.WorkPatternDayAllowedShift.query.filter(
                app.WorkPatternDayAllowedShift.work_pattern_day_id.in_(day_ids)
            ).delete(synchronize_session=False)
        app.WorkPatternDay.query.filter_by(unit_id=unit.id).delete()
        app.WorkPattern.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        app.ShiftType.query.filter_by(unit_id=unit.id).delete()
        app.db.session.delete(unit)
        app.db.session.commit()


def _assign_pattern(unit, person, shifts, specs, anchor=date(2026, 11, 1)):
    pattern = app.WorkPattern(
        unit_id=unit.id,
        name="Deterministic pattern",
        cycle_length_days=len(specs),
        contracted_minutes_per_cycle=480,
    )
    app.db.session.add(pattern)
    app.db.session.flush()
    for index, (day_type, code) in enumerate(specs):
        row = app.WorkPatternDay(
            unit_id=unit.id,
            work_pattern_id=pattern.id,
            day_index=index,
            day_type=day_type,
            fixed_shift_type_id=(shifts[code].id if day_type == "FIXED_SHIFT" else None),
            required_work=day_type not in {"OFF", "OPTIONAL_WORK"},
        )
        app.db.session.add(row)
        app.db.session.flush()
        if day_type == "WORK_ALLOWED_SET" and code:
            for allowed_code in code.split(","):
                app.db.session.add(app.WorkPatternDayAllowedShift(
                    unit_id=unit.id,
                    work_pattern_day_id=row.id,
                    shift_type_id=shifts[allowed_code].id,
                ))
    app.db.session.add(app.StaffPatternAssignment(
        unit_id=unit.id,
        staff_id=person.id,
        work_pattern_id=pattern.id,
        effective_from=anchor,
        anchor_date=anchor,
        anchor_day_index=0,
    ))
    app.db.session.commit()
    return pattern


def test_fixed_pattern_population_is_idempotent(population_service):
    service, unit, person, shifts = population_service
    pattern = _assign_pattern(unit, person, shifts, [
        ("FIXED_SHIFT", "M"), ("OFF", None),
    ])
    first = service.populate_or_recalculate_baseline(
        unit.id, date(2026, 11, 1), date(2026, 11, 2), mode="manual"
    )
    assert first.created == 2
    assert [change.generated_code for change in first.changes] == ["M", "OFF"]
    rows = app.Assignment.query.filter_by(unit_id=unit.id).order_by(
        app.Assignment.day
    ).all()
    generated_at = [row.generated_at for row in rows]
    assert [row.generated_code for row in rows] == ["M", "OFF"]
    assert all(row.generation_version == GENERATION_VERSION for row in rows)
    assert all(row.generated_from_pattern_id == pattern.id for row in rows)

    second = service.populate_or_recalculate_baseline(
        unit.id, date(2026, 11, 1), date(2026, 11, 2), mode="manual"
    )
    assert second.changed == 0
    assert second.unchanged == 2
    assert [row.generated_at for row in rows] == generated_at


def test_recalculation_preserves_editor_override(population_service):
    service, unit, person, shifts = population_service
    _assign_pattern(unit, person, shifts, [("FIXED_SHIFT", "M")])
    row = app.Assignment(
        unit_id=unit.id,
        staff_id=person.id,
        day=date(2026, 11, 1),
        code="A",
        generated_code="OFF",
        override_code="A",
        override_type="MANUAL",
    )
    app.db.session.add(row)
    app.db.session.commit()

    result = service.populate_or_recalculate_baseline(
        unit.id, row.day, row.day, mode="manual"
    )
    assert result.updated == 1
    assert row.generated_code == "M"
    assert row.override_code == "A"
    assert row.effective_code == "A"
    assert row.code == "A"


def test_automatic_mode_skips_whole_protected_months(population_service):
    service, unit, person, shifts = population_service
    _assign_pattern(unit, person, shifts, [("FIXED_SHIFT", "M")])
    result = service.populate_or_recalculate_baseline(
        unit.id,
        date(2026, 10, 30),
        date(2026, 11, 2),
        mode="automatic",
        reference_date=date(2026, 8, 3),
    )
    assert result.protected_dates == 2
    assert result.populated_from == date(2026, 11, 1)
    assert result.created == 2
    assert app.Assignment.query.filter_by(
        unit_id=unit.id, day=date(2026, 10, 31)
    ).first() is None


def test_automatic_mode_never_modifies_a_closed_roster_period(population_service):
    service, unit, person, shifts = population_service
    _assign_pattern(unit, person, shifts, [("FIXED_SHIFT", "M")])
    app.db.session.add(app.RosterPeriod(
        unit_id=unit.id, year=2026, month=11, status="CLOSED",
        generated_at=app.utcnow(), generation_method="MANUAL",
    ))
    app.db.session.commit()
    service.dependencies = replace(service.dependencies, RosterPeriod=app.RosterPeriod)
    result = service.populate_or_recalculate_baseline(
        unit.id, date(2026, 11, 1), date(2026, 11, 2), mode="automatic",
        reference_date=date(2026, 8, 3),
    )
    assert result.changed == 0
    assert result.protected_dates == 2
    assert app.Assignment.query.filter_by(unit_id=unit.id).count() == 0


def test_dry_run_reports_without_writing(population_service):
    service, unit, person, shifts = population_service
    _assign_pattern(unit, person, shifts, [("FIXED_SHIFT", "M")])
    result = service.populate_or_recalculate_baseline(
        unit.id,
        date(2026, 11, 1),
        date(2026, 11, 2),
        mode="manual",
        dry_run=True,
    )
    assert result.dry_run
    assert result.created == 2
    assert app.Assignment.query.filter_by(unit_id=unit.id).count() == 0


def test_flexible_pattern_is_explicitly_unresolved(population_service):
    service, unit, person, shifts = population_service
    _assign_pattern(unit, person, shifts, [("WORK_ALLOWED_SET", "M,A")])
    result = service.populate_or_recalculate_baseline(
        unit.id, date(2026, 11, 1), date(2026, 11, 1), mode="manual"
    )
    row = app.Assignment.query.filter_by(unit_id=unit.id).one()
    assert result.unresolved_flexible_dates == 1
    assert row.generated_code == "OFF"


def test_legacy_pattern_fallback_and_inactive_staff(population_service):
    service, unit, person, _shifts = population_service
    active = service.populate_or_recalculate_baseline(
        unit.id, date(2026, 11, 1), date(2026, 11, 1), mode="manual"
    )
    assert active.created == 1
    row = app.Assignment.query.filter_by(unit_id=unit.id).one()
    assert row.generated_code == "A"
    assert row.generated_from_pattern_id is None

    person.is_operational = False
    app.db.session.commit()
    inactive = service.populate_or_recalculate_baseline(
        unit.id, row.day, row.day, mode="manual"
    )
    assert inactive.updated == 1
    assert row.generated_code == "OFF"
