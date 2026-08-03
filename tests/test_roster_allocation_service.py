from datetime import date

import pytest

from roster_allocation_service import (
    AllocationShift,
    AllocationStaff,
    ExistingAllocation,
    HardConstraintResult,
    ProposalStatus,
    SoftConstraintResult,
    StaffingNeed,
    generate_roster_proposal,
)


DAY = date(2026, 9, 1)
NIGHT = AllocationShift(4, "N", 600, is_night=True)


def allowed(_staff_id, _day, _shift_id):
    return HardConstraintResult(True)


def proposal(staff, *, existing=(), hard=allowed, soft=None, **options):
    return generate_roster_proposal(
        DAY,
        DAY,
        staff=staff,
        shifts=[NIGHT],
        staffing_needs=[StaffingNeed(DAY, NIGHT.shift_type_id, 1)],
        existing_assignments=existing,
        hard_constraint=hard,
        soft_constraint=soft,
        **options,
    )


def test_fills_uncovered_shift_without_writing_input_records():
    people = [AllocationStaff(1, "Alex", 1200)]

    result = proposal(people)

    assert result.status == ProposalStatus.FEASIBLE
    assert result.proposed_assignments[0].staff_id == 1
    assert result.retained_assignments == ()
    assert people[0].actual_minutes == 0


def test_hard_constraint_can_block_no_night_employee():
    def no_night(staff_id, _day, _shift_id):
        if staff_id == 1:
            return HardConstraintResult(
                False, "NO_NIGHT_RULE", "Employee cannot work nights."
            )
        return HardConstraintResult(True)

    result = proposal(
        [AllocationStaff(1, "Alex", 1200), AllocationStaff(2, "Blair", 1200)],
        hard=no_night,
    )

    assert result.proposed_assignments[0].staff_id == 2


def test_leave_constraint_is_never_traded_for_fairness():
    def on_leave(staff_id, _day, _shift_id):
        if staff_id == 1:
            return HardConstraintResult(
                False, "APPROVED_LEAVE", "Employee is on approved leave."
            )
        return HardConstraintResult(True)

    result = proposal(
        [
            AllocationStaff(1, "Under target", 2000),
            AllocationStaff(2, "Available", 1000),
        ],
        hard=on_leave,
    )

    assert result.proposed_assignments[0].staff_id == 2


def test_existing_hard_locked_assignment_is_retained_and_covers_need():
    locked = ExistingAllocation(10, 1, DAY, 4, "N", "HARD_LOCKED")

    result = proposal([AllocationStaff(1, "Alex", 1200)], existing=[locked])

    assert result.status == ProposalStatus.FEASIBLE
    assert result.proposed_assignments == ()
    assert result.retained_assignments == (locked,)


def test_one_duty_per_day_is_a_hard_constraint():
    existing = ExistingAllocation(10, 1, DAY, 2, "M")

    result = proposal([AllocationStaff(1, "Alex", 1200)], existing=[existing])

    assert result.status == ProposalStatus.INFEASIBLE
    assert "ONE_DUTY_PER_DAY" in result.uncovered_shifts[0].reason_codes


def test_avoids_overtime_when_another_legal_person_exists():
    result = proposal([
        AllocationStaff(1, "At target", 600, actual_minutes=600),
        AllocationStaff(2, "Below target", 1200, actual_minutes=0),
    ])

    assert result.proposed_assignments[0].staff_id == 2


def test_selects_fairer_night_candidate_when_other_costs_equal():
    result = proposal([
        AllocationStaff(1, "More nights", 1200, night_count=3, target_night_count=2),
        AllocationStaff(2, "Fewer nights", 1200, night_count=0, target_night_count=2),
    ])

    assert result.proposed_assignments[0].staff_id == 2


def test_soft_preference_penalty_selects_non_breaching_candidate():
    def preferences(staff_id, _day, _shift_id):
        if staff_id == 1:
            return SoftConstraintResult(
                penalty=5,
                reason_codes=("SOFT_AVOID_NIGHT",),
                explanations=("Employee prefers to avoid night duties.",),
            )
        return SoftConstraintResult()

    result = proposal(
        [AllocationStaff(1, "Avoids nights", 1200), AllocationStaff(2, "Alex", 1200)],
        soft=preferences,
    )

    assert result.proposed_assignments[0].staff_id == 2


def test_reports_structured_reasons_when_no_legal_candidate_exists():
    def blocked(_staff_id, _day, _shift_id):
        return HardConstraintResult(False, "MEDICAL_EXPIRED", "Medical is expired.")

    result = proposal([AllocationStaff(1, "Alex", 1200)], hard=blocked)

    assert result.status == ProposalStatus.INFEASIBLE
    assert result.uncovered_shifts[0].reason_codes == ("MEDICAL_EXPIRED",)
    assert result.uncovered_shifts[0].missing_count == 1


def test_deterministic_tie_break_uses_staff_id():
    people = [AllocationStaff(2, "Blair", 1200), AllocationStaff(1, "Alex", 1200)]

    first = proposal(people)
    second = proposal(people)

    assert first.proposed_assignments[0].staff_id == 1
    assert first.proposed_assignments == second.proposed_assignments


def test_initial_service_refuses_regeneration_of_existing_assignments():
    with pytest.raises(ValueError, match="preserve_existing=True"):
        proposal(
            [AllocationStaff(1, "Alex", 1200)],
            existing=[ExistingAllocation(10, 1, DAY, 4, "N")],
            preserve_existing=False,
        )


def test_later_candidates_can_validate_earlier_proposed_duties():
    second_day = date(2026, 9, 2)
    planned = set()

    def sequence_constraint(staff_id, day, _shift_id):
        if staff_id == 1 and day == second_day and (1, DAY) in planned:
            return HardConstraintResult(
                False, "FATIGUE_RULE", "Proposed sequence breaches fatigue rules."
            )
        return HardConstraintResult(True)

    result = generate_roster_proposal(
        DAY,
        second_day,
        staff=[
            AllocationStaff(1, "Alex", 2400),
            AllocationStaff(2, "Blair", 2400),
        ],
        shifts=[NIGHT],
        staffing_needs=[StaffingNeed(DAY, 4, 1), StaffingNeed(second_day, 4, 1)],
        hard_constraint=sequence_constraint,
        on_assignment_selected=lambda staff_id, day, _shift_id: planned.add(
            (staff_id, day)
        ),
    )

    assert [row.staff_id for row in result.proposed_assignments] == [1, 2]
