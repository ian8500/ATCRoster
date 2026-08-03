from datetime import date

import app
from roster_period_service import RosterPeriodDependencies, RosterPeriodService


def test_period_status_boundaries_and_closed_override():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit(code="RPER", name="Roster Period Test", protected_roster_months_ahead=2)
        app.db.session.add(unit)
        app.db.session.flush()
        service = RosterPeriodService(RosterPeriodDependencies(
            db=app.db, RosterPeriod=app.RosterPeriod, utcnow=app.utcnow,
        ))
        reference = date(2026, 12, 15)
        assert service.status_for(unit, 2026, 11, reference) == "HISTORICAL"
        assert service.status_for(unit, 2026, 12, reference) == "CURRENT"
        assert service.status_for(unit, 2027, 1, reference) == "PROTECTED"
        assert service.status_for(unit, 2027, 2, reference) == "PROTECTED"
        assert service.status_for(unit, 2027, 3, reference) == "FUTURE_AUTOMATIC"
        row, created = service.ensure_period(unit, 2027, 3, reference_date=reference)
        assert created and row.status == "FUTURE_AUTOMATIC"
        row.status = "CLOSED"
        same, created = service.ensure_period(
            unit, 2027, 3, reference_date=date(2027, 3, 1)
        )
        assert not created and same.status == "CLOSED"
        app.db.session.rollback()


def test_future_period_command_is_idempotent_and_records_generation_event():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit.query.filter_by(code="RPCLI").first()
        if not unit:
            unit = app.Unit(
                code="RPCLI", name="Roster CLI Test",
                protected_roster_months_ahead=0, onboarding_step=100,
            )
            app.db.session.add(unit)
            app.db.session.flush()
        app.RosterImpactException.query.filter_by(unit_id=unit.id).delete()
        app.RosterImpactEvent.query.filter_by(unit_id=unit.id).delete()
        app.RosterPeriod.query.filter_by(unit_id=unit.id).delete()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        person = app.Staff(
            unit_id=unit.id, username="period_cli_person", password_hash="unused",
            name="Period CLI Person", staff_no="RP-1", role="user",
            membership_status="active", is_operational=False, pattern_csv="M,OFF",
            pattern_anchor=date.today().replace(day=1),
        )
        app.db.session.add(person)
        app.db.session.commit()
        unit_id = unit.id

    runner = app.app.test_cli_runner()
    first = runner.invoke(args=[
        "roster", "ensure-future-periods", "--months-ahead", "2",
        "--unit-code", "RPCLI",
    ])
    assert first.exit_code == 0, first.output
    second = runner.invoke(args=[
        "roster", "ensure-future-periods", "--months-ahead", "2",
        "--unit-code", "RPCLI",
    ])
    assert second.exit_code == 0, second.output
    with app.app.app_context():
        assert app.RosterPeriod.query.filter_by(unit_id=unit_id).count() == 3
        events = app.RosterImpactEvent.query.filter_by(
            unit_id=unit_id, event_type="FUTURE_PERIOD_CREATED"
        ).all()
        assert len(events) == 2
        assert all(event.status in {"COMPLETED", "COMPLETED_WITH_WARNINGS"} for event in events)
        assert app.Assignment.query.filter_by(unit_id=unit_id).count() > 0
        app.RosterImpactException.query.filter_by(unit_id=unit_id).delete()
        app.RosterImpactEvent.query.filter_by(unit_id=unit_id).delete()
        app.RosterPeriod.query.filter_by(unit_id=unit_id).delete()
        app.Assignment.query.filter_by(unit_id=unit_id).delete()
        app.Staff.query.filter_by(unit_id=unit_id).delete()
        app.db.session.delete(app.db.session.get(app.Unit, unit_id))
        app.db.session.commit()
