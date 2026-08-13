from datetime import date
from types import SimpleNamespace

from atcroster.roster.month_view import MonthDisplayDependencies, RosterMonthViewService


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
