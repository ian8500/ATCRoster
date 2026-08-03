from datetime import date

import pytest

import app
from roster_impact_service import (
    RosterImpactDependencies,
    RosterImpactEventType,
    RosterImpactService,
)
from roster_population_service import (
    DeterministicRosterPopulationService,
    PopulationDependencies,
)


@pytest.fixture()
def impact_context():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit.query.filter_by(code="RIMP").first()
        if not unit:
            unit = app.Unit(
                code="RIMP", name="Roster Impact Test",
                protected_roster_months_ahead=2, onboarding_step=100,
            )
            app.db.session.add(unit)
            app.db.session.flush()
        app.RosterImpactException.query.filter_by(unit_id=unit.id).delete()
        app.RosterImpactEvent.query.filter_by(unit_id=unit.id).delete()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.StaffPatternAssignment.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        person = app.Staff(
            unit_id=unit.id, username="roster_impact_person",
            password_hash="unused", name="Roster Impact Person",
            staff_no="RI-001", role="user", membership_status="active",
            is_operational=True,
        )
        app.db.session.add(person)
        app.db.session.commit()

        population = DeterministicRosterPopulationService(PopulationDependencies(
            db=app.db, Unit=app.Unit, Staff=app.Staff,
            Assignment=app.Assignment, ShiftType=app.ShiftType,
            WorkPattern=app.WorkPattern, WorkPatternDay=app.WorkPatternDay,
            WorkPatternDayAllowedShift=app.WorkPatternDayAllowedShift,
            StaffPatternAssignment=app.StaffPatternAssignment,
            utcnow=app.utcnow, legacy_code_resolver=lambda _person, _day: "M",
        ))
        coverage_calls = []
        service = RosterImpactService(RosterImpactDependencies(
            db=app.db, Unit=app.Unit,
            RosterImpactEvent=app.RosterImpactEvent,
            RosterImpactException=app.RosterImpactException,
            population_service=population,
            generated_horizon_end=lambda _unit_id, _start: date(2026, 11, 2),
            recalculate_coverage=lambda *args: coverage_calls.append(args),
            utcnow=app.utcnow,
        ))
        yield service, unit, person, coverage_calls

        app.db.session.rollback()
        app.RosterImpactException.query.filter_by(unit_id=unit.id).delete()
        app.RosterImpactEvent.query.filter_by(unit_id=unit.id).delete()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.StaffPatternAssignment.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        app.db.session.delete(unit)
        app.db.session.commit()


def test_event_splits_protected_and_automatic_ranges(impact_context):
    service, unit, person, coverage_calls = impact_context
    override = app.Assignment(
        unit_id=unit.id, staff_id=person.id, day=date(2026, 11, 2),
        code="A", generated_code="OFF", override_code="A",
        override_type="MANUAL",
    )
    app.db.session.add(override)
    app.db.session.commit()

    result = service.handle_roster_impact_event(
        unit.id, RosterImpactEventType.WORK_PATTERN_CHANGE,
        date(2026, 9, 15), staff_ids=[person.id], rebuild_baseline=True,
        reason="Dated work-pattern change", reference_date=date(2026, 8, 3),
    )
    app.db.session.commit()

    assert (result.protected_from, result.protected_to) == (
        date(2026, 9, 15), date(2026, 10, 31)
    )
    assert (result.automatic_from, result.automatic_to) == (
        date(2026, 11, 1), date(2026, 11, 2)
    )
    assert result.exception_count == 1
    assert result.population_result.changed == 2
    assert len(coverage_calls) == 1
    assert coverage_calls[0][1:3] == (date(2026, 9, 15), date(2026, 11, 2))
    assert override.generated_code == "M"
    assert override.override_code == "A"
    assert override.effective_code == "A"
    assert override.generation_event_id == result.event_id
    event = app.db.session.get(app.RosterImpactEvent, result.event_id)
    assert event.status == "COMPLETED"
    assert app.RosterImpactException.query.filter_by(event_id=event.id).count() == 1


def test_coverage_only_event_does_not_create_manual_exception(impact_context):
    service, unit, person, coverage_calls = impact_context
    result = service.handle_roster_impact_event(
        unit.id, RosterImpactEventType.FIRST_UE_ACHIEVED,
        date(2026, 9, 15), staff_ids=[person.id],
        rebuild_baseline=False, reference_date=date(2026, 8, 3),
    )
    app.db.session.commit()
    assert result.exception_count == 0
    assert result.population_result is None
    assert len(coverage_calls) == 1
    assert app.Assignment.query.filter_by(unit_id=unit.id).count() == 0


def test_handler_rejects_override_destruction_and_unknown_events(impact_context):
    service, unit, _person, _coverage_calls = impact_context
    with pytest.raises(ValueError, match="preserve editor overrides"):
        service.handle_roster_impact_event(
            unit.id, RosterImpactEventType.MANUAL_RECALCULATION,
            date(2026, 11, 1), preserve_overrides=False,
        )
    with pytest.raises(ValueError, match="Unknown roster-impact"):
        service.handle_roster_impact_event(
            unit.id, "NOT_A_REAL_EVENT", date(2026, 11, 1)
        )
    with pytest.raises(ValueError, match="end date"):
        service.handle_roster_impact_event(
            unit.id, RosterImpactEventType.MANUAL_RECALCULATION,
            date(2026, 11, 2), effective_to=date(2026, 11, 1),
        )
    assert app.RosterImpactEvent.query.filter_by(unit_id=unit.id).count() == 0


def test_failure_rolls_back_event_and_exceptions(impact_context):
    service, unit, person, _coverage_calls = impact_context
    service.dependencies.population_service.populate_or_recalculate_baseline = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("generation failed"))
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        service.handle_roster_impact_event(
            unit.id, RosterImpactEventType.UNIT_JOINER,
            date(2026, 9, 1), staff_ids=[person.id], rebuild_baseline=True,
            reference_date=date(2026, 8, 3),
        )
    assert app.RosterImpactEvent.query.filter_by(unit_id=unit.id).count() == 0
    assert app.RosterImpactException.query.filter_by(unit_id=unit.id).count() == 0


@pytest.mark.parametrize(
    ("code", "old_status", "new_status", "expected"),
    [
        ("MEDICAL", None, "valid", "MEDICAL_RESTORED"),
        ("MEDICAL", "valid", "suspended", "MEDICAL_EXPIRED"),
        ("ADI", None, "valid", "FIRST_UE_ACHIEVED"),
        ("APS", "valid", "suspended", "UE_SUSPENDED"),
        ("APS", "suspended", "valid", "UE_RESTORED"),
        ("OJTI", None, "valid", "OJTI_ACHIEVED"),
        ("ASSESSOR", None, "valid", "ASSESSOR_ACHIEVED"),
    ],
)
def test_qualification_transitions_map_to_required_events(
    code, old_status, new_status, expected
):
    event_type, effective = app._qualification_impact_type(
        code, old_status, None, None,
        new_status, date(2026, 12, 1), date(2027, 12, 1),
    )
    assert event_type.value == expected
    assert effective == date(2026, 12, 1)
