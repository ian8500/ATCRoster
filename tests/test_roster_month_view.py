"""Focused safety checks for the monthly roster data projection."""

from datetime import date
from types import SimpleNamespace

import pytest

from atcroster.roster.month_view import (
    MonthDisplayDependencies,
    MonthRosterLoadDependencies,
    RosterMonthViewService,
    load_month_roster,
)


def test_month_view_builds_counts_rag_and_watch_order_once():
    first_day, second_day = date(2026, 8, 1), date(2026, 8, 2)
    staff = [
        SimpleNamespace(id=2, name="Bravo", is_operational=True, is_wm=False, is_dwm=False, medical_expiry=None, tower_ue_expiry=None, radar_ue_expiry=None, met_ue_expiry=None, tower_ut=False, radar_ut=False, met_ut=False),
        SimpleNamespace(id=1, name="Alpha", is_operational=True, is_wm=True, is_dwm=False, medical_expiry=None, tower_ue_expiry=None, radar_ue_expiry=None, met_ue_expiry=None, tower_ut=False, radar_ut=False, met_ut=False),
    ]
    service = RosterMonthViewService(MonthDisplayDependencies(
        staff_is_countable_on=lambda *_: True,
        shift_counter_group_for_day=lambda code, *_: code,
    ))
    state = service.build(
        staff=staff, days=[first_day, second_day],
        assignment_map={1: {first_day: "M"}, 2: {first_day: "M"}},
        capability_matrix={}, excluded=set(), training_codes=set(),
        requirements={first_day: {"M": 2, "D": 0, "A": 0, "N": 0}, second_day: {"M": 1, "D": 0, "A": 0, "N": 0}},
        night_active={first_day: False, second_day: False}, unit_id=1,
        display_watch_by_staff={1: 10, 2: 20}, watch_order={10: 0, 20: 1}, today=first_day,
    )

    assert [person.id for person in staff] == [1, 2]
    assert state["counters"][first_day]["M"] == 2
    assert state["rag"][first_day]["M"] == "green"
    assert state["watch_break_after_ids"] == [1]


def test_month_roster_loader_does_not_mask_a_query_failure_as_an_empty_roster():
    class FailingQuery:
        def outerjoin(self, *_args):
            return self

        def filter(self, *_args):
            raise RuntimeError("operational database unavailable")

    dependencies = MonthRosterLoadDependencies(
        db=SimpleNamespace(),
        Assignment=SimpleNamespace(),
        Requirement=SimpleNamespace(),
        Staff=SimpleNamespace(
            query=FailingQuery(), unit_id=1, role="role", watch_id=1, id=1
        ),
        Watch=SimpleNamespace(id=1, order_index=1),
        ensure_month_requirement=lambda *_args: pytest.fail(
            "a failed query must not create a misleading fallback requirement"
        ),
        log_exception=lambda *_args: None,
    )

    with pytest.raises(RuntimeError, match="operational database unavailable"):
        load_month_roster(1, 2025, 4, dependencies)
