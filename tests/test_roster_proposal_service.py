from datetime import date
from types import SimpleNamespace

from roster_proposal_service import RosterProposalDependencies, index_fairness_rows


def test_fairness_dependency_contract_returns_rows_and_totals():
    row = SimpleNamespace(staff_id=17)
    dependencies = RosterProposalDependencies(
        db=None,
        Staff=None,
        ShiftType=None,
        Assignment=None,
        Sickness=None,
        Requirement=None,
        SpecialRequirement=None,
        RosterProposal=None,
        RosterProposalAssignment=None,
        ChangeLog=None,
        work_pattern_service=None,
        requirements_for_day=lambda *_: {},
        shift_group_for_day=lambda *_: None,
        shift_minutes=lambda *_: 0,
        staff_is_countable_on=lambda *_: True,
        staff_has_qualification=lambda *_: True,
        would_trigger_fatigue=lambda *_: [],
        compute_fairness_range=lambda *_: ([row], {"actual_minutes": 0}),
        utcnow=lambda: None,
    )

    rows, totals = dependencies.compute_fairness_range(
        date(2026, 8, 1), date(2026, 8, 31)
    )

    assert rows == [row]
    assert totals == {"actual_minutes": 0}
    assert index_fairness_rows((rows, totals)) == {17: row}
