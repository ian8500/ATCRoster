from datetime import date

import app


def test_effective_dated_capability_controls_operational_contribution():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit.query.filter_by(code="CAP").first()
        if not unit:
            unit = app.Unit(code="CAP", name="Capability Test", onboarding_step=100)
            app.db.session.add(unit)
            app.db.session.flush()
        person = app.Staff(
            unit_id=unit.id, username="capability_person",
            password_hash="unused", name="Capability Person", staff_no="CAP-1",
            role="user", membership_status="active", is_operational=True,
            employment_start_date=date(2026, 9, 1),
            unit_join_date=date(2026, 9, 15),
            roster_start_date=date(2026, 9, 15),
        )
        app.db.session.add(person)
        app.db.session.flush()
        medical_type = app.QualificationType(
            unit_id=unit.id, code="MEDICAL", label="Medical",
            expiry_required=True,
        )
        ue_type = app.QualificationType(
            unit_id=unit.id, code="ADI", label="Tower UE",
            expiry_required=True,
        )
        app.db.session.add_all((medical_type, ue_type))
        app.db.session.flush()
        medical = app.PersonQualification(
            unit_id=unit.id, person_id=person.id,
            qualification_type_id=medical_type.id, status="valid",
            valid_from=date(2026, 9, 1), expires_on=date(2027, 9, 30),
        )
        ue = app.PersonQualification(
            unit_id=unit.id, person_id=person.id,
            qualification_type_id=ue_type.id, status="valid",
            valid_from=date(2026, 10, 1), expires_on=date(2027, 9, 30),
        )
        app.db.session.add_all((medical, ue))
        app.db.session.commit()

        service = app.operational_capability_service()
        before_join = service.get_staff_operational_capability(
            person.id, date(2026, 9, 10)
        )
        assert not before_join.in_unit
        assert not before_join.counts_as_operational
        before_ue = service.get_staff_operational_capability(
            person.id, date(2026, 9, 20)
        )
        assert before_ue.medically_valid
        assert not before_ue.counts_as_operational
        qualified = service.get_staff_operational_capability(
            person.id, date(2026, 10, 1)
        )
        assert qualified.counts_as_operational
        assert qualified.independent_competencies == frozenset({"ADI"})

        ue.suspended_from = date(2026, 11, 1)
        ue.suspended_to = date(2026, 11, 30)
        app.db.session.commit()
        suspended = service.get_staff_operational_capability(
            person.id, date(2026, 11, 15)
        )
        assert not suspended.counts_as_operational

        app.PersonQualificationHistory.query.filter(
            app.PersonQualificationHistory.unit_id == unit.id
        ).delete(synchronize_session=False)
        app.PersonQualification.query.filter_by(unit_id=unit.id).delete()
        app.QualificationType.query.filter_by(unit_id=unit.id).delete()
        app.Staff.query.filter_by(unit_id=unit.id).delete()
        app.db.session.delete(unit)
        app.db.session.commit()
