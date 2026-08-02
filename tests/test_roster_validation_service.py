from datetime import date, time

import app


def test_roster_validation_finds_blockers_and_soft_preferences():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit(
            code="RVAL", name="Roster Validation Test", onboarding_step=100
        )
        app.db.session.add(unit)
        app.db.session.flush()
        morning = app.ShiftType(
            unit_id=unit.id, code="M", name="Morning",
            start_time=time(6), end_time=time(14), is_working=True,
        )
        night = app.ShiftType(
            unit_id=unit.id, code="N", name="Night",
            start_time=time(22), end_time=time(6), is_working=True,
        )
        app.db.session.add_all([morning, night])
        app.db.session.flush()
        patterned = _person(unit.id, "patterned", "RV-1")
        preferred = _person(unit.id, "preferred", "RV-2")
        restricted = _person(unit.id, "restricted", "RV-3")
        irrelevant = _person(unit.id, "irrelevant", "RV-4")
        app.db.session.add_all([patterned, preferred, restricted, irrelevant])
        app.db.session.flush()
        pattern = app.WorkPattern(
            unit_id=unit.id, name="One on one off", cycle_length_days=2,
            contracted_minutes_per_cycle=480,
        )
        app.db.session.add(pattern)
        app.db.session.flush()
        app.db.session.add_all([
            app.WorkPatternDay(
                unit_id=unit.id, work_pattern_id=pattern.id, day_index=0,
                day_type="FIXED_SHIFT", fixed_shift_type_id=morning.id,
                required_work=True,
            ),
            app.WorkPatternDay(
                unit_id=unit.id, work_pattern_id=pattern.id, day_index=1,
                day_type="OFF", required_work=False,
            ),
            app.StaffPatternAssignment(
                unit_id=unit.id, staff_id=patterned.id,
                work_pattern_id=pattern.id, effective_from=date(2026, 9, 1),
                anchor_date=date(2026, 9, 1), anchor_day_index=0,
            ),
            app.StaffRule(
                unit_id=unit.id, staff_id=preferred.id,
                rule_type="AVOID_NIGHT", hardness="SOFT",
                effective_from=date(2026, 9, 1), penalty_weight=3,
            ),
            app.StaffRule(
                unit_id=unit.id, staff_id=restricted.id,
                rule_type="NO_NIGHT", hardness="HARD",
                effective_from=date(2026, 9, 1), penalty_weight=0,
            ),
        ])
        app.db.session.add_all([
            app.Assignment(
                unit_id=unit.id, staff_id=patterned.id,
                day=date(2026, 9, 1), code=morning.code,
            ),
            app.Assignment(
                unit_id=unit.id, staff_id=patterned.id,
                day=date(2026, 9, 2), code=morning.code,
            ),
            app.Assignment(
                unit_id=unit.id, staff_id=preferred.id,
                day=date(2026, 9, 1), code=night.code,
            ),
            app.Assignment(
                unit_id=unit.id, staff_id=restricted.id,
                day=date(2026, 9, 1), code=night.code,
            ),
            app.Assignment(
                unit_id=unit.id, staff_id=irrelevant.id,
                day=date(2026, 9, 1), code=morning.code,
            ),
        ])
        app.db.session.commit()

        result = app.roster_validation_service.validate_range(
            unit.id, date(2026, 9, 1), date(2026, 9, 2)
        )

        assert result.blocking_count == 2
        assert result.advisory_count == 1
        assert not result.can_publish
        facts = {
            (row.staff_id, row.day, row.reason_code, row.severity)
            for row in result.findings
        }
        assert (
            patterned.id, date(2026, 9, 2),
            "PATTERN_NON_WORKING_DAY", "blocking",
        ) in facts
        assert (
            restricted.id, date(2026, 9, 1), "NO_NIGHT_RULE", "blocking",
        ) in facts
        assert (
            preferred.id, date(2026, 9, 1), "SOFT_AVOID_NIGHT", "advisory",
        ) in facts
        assert all(row.staff_id != irrelevant.id for row in result.findings)


def _person(unit_id: int, username: str, staff_no: str):
    return app.Staff(
        unit_id=unit_id, username=f"roster_validation_{username}",
        password_hash="unused", name=username.title(), staff_no=staff_no,
        role="user", is_operational=True,
    )
