from datetime import date, time

import pytest

import app
from work_pattern_service import WorkPatternDependencies, WorkPatternService


@pytest.fixture()
def pattern_service():
    with app.app.app_context():
        app.db.create_all()
        unit = app.Unit.query.filter_by(code="WPAT").first()
        if not unit:
            unit = app.Unit(
                code="WPAT", name="Pattern Test Airport", onboarding_step=100
            )
            app.db.session.add(unit)
            app.db.session.flush()
        shifts = {}
        for code, start, end in (
            ("M", time(6), time(14)),
            ("D", time(8), time(16)),
            ("A", time(14), time(22)),
            ("N", time(22), time(6)),
        ):
            shift = app.ShiftType.query.filter_by(unit_id=unit.id, code=code).first()
            if not shift:
                shift = app.ShiftType(
                    unit_id=unit.id, code=code, name=code,
                    start_time=start, end_time=end, is_working=True,
                )
                app.db.session.add(shift)
                app.db.session.flush()
            shifts[code] = shift
        person = app.Staff.query.filter_by(
            unit_id=unit.id, staff_no="WP-001"
        ).first()
        if not person:
            person = app.Staff(
                unit_id=unit.id,
                username="work_pattern_test_person",
                password_hash="unused",
                name="Pattern Test Person",
                staff_no="WP-001",
                role="user",
            )
            app.db.session.add(person)
            app.db.session.flush()
        app.db.session.commit()

        service = WorkPatternService(WorkPatternDependencies(
            Staff=app.Staff,
            ShiftType=app.ShiftType,
            Leave=app.Leave,
            Assignment=app.Assignment,
            WorkPattern=app.WorkPattern,
            WorkPatternDay=app.WorkPatternDay,
            WorkPatternDayAllowedShift=app.WorkPatternDayAllowedShift,
            StaffPatternAssignment=app.StaffPatternAssignment,
            StaffRule=app.StaffRule,
            shift_group=lambda shift: shift.code[0],
        ))
        yield service, unit, person, shifts

        app.StaffRule.query.filter_by(unit_id=unit.id).delete()
        app.StaffPatternAssignment.query.filter_by(unit_id=unit.id).delete()
        allowed_day_ids = [
            row.id for row in app.WorkPatternDay.query.filter_by(unit_id=unit.id)
        ]
        if allowed_day_ids:
            app.WorkPatternDayAllowedShift.query.filter(
                app.WorkPatternDayAllowedShift.work_pattern_day_id.in_(allowed_day_ids)
            ).delete(synchronize_session=False)
        app.WorkPatternDay.query.filter_by(unit_id=unit.id).delete()
        app.WorkPattern.query.filter_by(unit_id=unit.id).delete()
        app.Leave.query.filter_by(unit_id=unit.id).delete()
        app.Assignment.query.filter_by(unit_id=unit.id).delete()
        app.db.session.commit()


def _add_pattern(
    unit, shifts, name: str, day_specs: list[tuple[str, str | None]],
):
    pattern = app.WorkPattern(
        unit_id=unit.id,
        name=name,
        cycle_length_days=len(day_specs),
        contracted_minutes_per_cycle=sum(
            480 for day_type, _ in day_specs if day_type != "OFF"
        ),
    )
    app.db.session.add(pattern)
    app.db.session.flush()
    rows = []
    for index, (day_type, shift_code) in enumerate(day_specs):
        row = app.WorkPatternDay(
            unit_id=unit.id,
            work_pattern_id=pattern.id,
            day_index=index,
            day_type=day_type,
            fixed_shift_type_id=(
                shifts[shift_code].id if day_type == "FIXED_SHIFT" else None
            ),
            required_work=day_type not in {"OFF", "OPTIONAL_WORK"},
        )
        app.db.session.add(row)
        app.db.session.flush()
        if day_type == "WORK_ALLOWED_SET" and shift_code:
            for code in shift_code.split(","):
                app.db.session.add(app.WorkPatternDayAllowedShift(
                    unit_id=unit.id,
                    work_pattern_day_id=row.id,
                    shift_type_id=shifts[code].id,
                ))
        rows.append(row)
    app.db.session.commit()
    return pattern, rows


def _assign(person, pattern, effective_from, anchor_date, effective_to=None):
    row = app.StaffPatternAssignment(
        unit_id=person.unit_id,
        staff_id=person.id,
        work_pattern_id=pattern.id,
        effective_from=effective_from,
        effective_to=effective_to,
        anchor_date=anchor_date,
        anchor_day_index=0,
    )
    app.db.session.add(row)
    app.db.session.commit()
    return row


def test_six_on_four_off_resolves_anchor_and_wraparound(pattern_service):
    service, unit, person, shifts = pattern_service
    pattern, _ = _add_pattern(unit, shifts, "6 on 4 off", [
        ("FIXED_SHIFT", "M"), ("FIXED_SHIFT", "M"),
        ("FIXED_SHIFT", "A"), ("FIXED_SHIFT", "A"),
        ("FIXED_SHIFT", "N"), ("FIXED_SHIFT", "N"),
        ("OFF", None), ("OFF", None), ("OFF", None), ("OFF", None),
    ])
    anchor = date(2026, 1, 1)
    _assign(person, pattern, anchor, anchor)

    assert service.get_pattern_day_for_staff(person.id, anchor).cycle_index == 0
    assert service.get_pattern_day_for_staff(
        person.id, date(2026, 1, 11)
    ).cycle_index == 0
    assert service.get_pattern_day_for_staff(
        person.id, date(2026, 1, 7)
    ).pattern_day.day_type == "OFF"


def test_historical_assignment_and_dated_pattern_change(pattern_service):
    service, unit, person, shifts = pattern_service
    old, _ = _add_pattern(unit, shifts, "Historical days", [
        ("FIXED_SHIFT", "M"), ("OFF", None),
    ])
    new, _ = _add_pattern(unit, shifts, "Future days", [
        ("FIXED_SHIFT", "A"), ("OFF", None),
    ])
    _assign(person, old, date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 31))
    _assign(person, new, date(2026, 2, 1), date(2026, 2, 1))

    january = service.get_pattern_day_for_staff(person.id, date(2026, 1, 1))
    february = service.get_pattern_day_for_staff(person.id, date(2026, 2, 1))
    assert january.pattern.id == old.id
    assert january.pattern_day.fixed_shift_type_id == shifts["M"].id
    assert february.pattern.id == new.id
    assert february.pattern_day.fixed_shift_type_id == shifts["A"].id

    old.is_active = False
    app.db.session.commit()
    assert service.get_pattern_day_for_staff(
        person.id, date(2026, 1, 1)
    ).pattern.id == old.id


def test_four_on_six_off_allowed_set_and_off_day(pattern_service):
    service, unit, person, shifts = pattern_service
    pattern, _ = _add_pattern(unit, shifts, "4 on 6 off", [
        ("WORK_ALLOWED_SET", "M,D"), ("WORK_ALLOWED_SET", "M,D"),
        ("FIXED_SHIFT", "A"), ("FIXED_SHIFT", "A"),
        ("OFF", None), ("OFF", None), ("OFF", None), ("OFF", None),
        ("OFF", None), ("OFF", None),
    ])
    anchor = date(2026, 3, 1)
    _assign(person, pattern, anchor, anchor)

    assert service.is_staff_eligible_for_shift(
        person.id, anchor, shifts["D"].id
    ).eligible
    wrong = service.is_staff_eligible_for_shift(
        person.id, anchor, shifts["A"].id
    )
    assert not wrong.eligible
    assert wrong.reason_code == "PATTERN_SHIFT_NOT_ALLOWED"
    off = service.is_staff_eligible_for_shift(
        person.id, date(2026, 3, 5), shifts["M"].id
    )
    assert not off.eligible
    assert off.reason_code == "PATTERN_NON_WORKING_DAY"


def test_hard_no_night_blocks_and_soft_avoid_night_penalises(pattern_service):
    service, unit, person, shifts = pattern_service
    on_date = date(2026, 4, 1)
    hard = app.StaffRule(
        unit_id=unit.id, staff_id=person.id, rule_type="NO_NIGHT",
        hardness="HARD", effective_from=on_date, reason="Medical restriction",
    )
    app.db.session.add(hard)
    app.db.session.commit()
    result = service.is_staff_eligible_for_shift(person.id, on_date, shifts["N"].id)
    assert not result.eligible
    assert result.reason_code == "NO_NIGHT_RULE"

    app.db.session.delete(hard)
    app.db.session.add(app.StaffRule(
        unit_id=unit.id, staff_id=person.id, rule_type="AVOID_NIGHT",
        hardness="SOFT", effective_from=on_date, penalty_weight=17,
    ))
    app.db.session.commit()
    preferred = service.is_staff_eligible_for_shift(
        person.id, on_date, shifts["N"].id
    )
    assert preferred.eligible
    assert preferred.soft_penalty == 17
    assert preferred.reasons[0].code == "SOFT_AVOID_NIGHT"


def test_expired_and_future_rules_do_not_apply_early(pattern_service):
    service, unit, person, shifts = pattern_service
    app.db.session.add_all([
        app.StaffRule(
            unit_id=unit.id, staff_id=person.id, rule_type="NO_NIGHT",
            hardness="HARD", effective_from=date(2025, 1, 1),
            effective_to=date(2025, 12, 31),
        ),
        app.StaffRule(
            unit_id=unit.id, staff_id=person.id, rule_type="NO_NIGHT",
            hardness="HARD", effective_from=date(2027, 1, 1),
        ),
    ])
    app.db.session.commit()
    result = service.is_staff_eligible_for_shift(
        person.id, date(2026, 6, 1), shifts["N"].id
    )
    assert result.eligible


def test_approved_leave_and_allowed_shift_rule_are_hard_constraints(pattern_service):
    service, unit, person, shifts = pattern_service
    on_date = date(2026, 7, 1)
    app.db.session.add(app.StaffRule(
        unit_id=unit.id, staff_id=person.id, rule_type="ALLOWED_SHIFT",
        hardness="HARD", effective_from=on_date, shift_type_id=shifts["M"].id,
    ))
    app.db.session.commit()
    disallowed = service.is_staff_eligible_for_shift(
        person.id, on_date, shifts["A"].id
    )
    assert not disallowed.eligible
    assert disallowed.reason_code == "SHIFT_NOT_ALLOWED_RULE"

    app.db.session.add(app.Leave(
        unit_id=unit.id, staff_id=person.id, leave_type="AL",
        start=on_date, end=on_date,
    ))
    app.db.session.commit()
    leave = service.is_staff_eligible_for_shift(
        person.id, on_date, shifts["M"].id
    )
    assert not leave.eligible
    assert leave.reason_code == "APPROVED_LEAVE"


def test_overlapping_effective_pattern_ranges_are_rejected(pattern_service):
    service, unit, person, shifts = pattern_service
    pattern, _ = _add_pattern(unit, shifts, "Overlap test", [
        ("FIXED_SHIFT", "M"), ("OFF", None),
    ])
    _assign(
        person, pattern, date(2026, 1, 1), date(2026, 1, 1), date(2026, 6, 30)
    )
    candidate = app.StaffPatternAssignment(
        unit_id=unit.id, staff_id=person.id, work_pattern_id=pattern.id,
        effective_from=date(2026, 6, 1), anchor_date=date(2026, 6, 1),
        anchor_day_index=0,
    )
    with pytest.raises(ValueError, match="overlaps"):
        service.validate_staff_pattern_assignment(candidate)
