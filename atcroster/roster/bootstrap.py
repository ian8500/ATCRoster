"""Legacy operational-data bootstrap for local desktop installations."""

from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any


def ensure_shift(
    code: str,
    name: str,
    *,
    db: Any,
    ShiftType: Any,
    start: time | None = None,
    end: time | None = None,
    is_working: bool = False,
    is_training: bool = False,
) -> Any:
    shift = ShiftType.query.filter_by(code=code).first()
    if shift is None:
        shift = ShiftType(
            code=code,
            name=name,
            start_time=start,
            end_time=end,
            is_working=is_working,
            is_training=is_training,
        )
        db.session.add(shift)
        db.session.commit()
    return shift


def ensure_watch(name: str, order_index: int, *, db: Any, Watch: Any) -> Any:
    watch = Watch.query.filter_by(name=name).first()
    if watch is None:
        watch = Watch(name=name, order_index=order_index)
        db.session.add(watch)
        db.session.commit()
    return watch


def seed_legacy_operational_data(
    *, db: Any, Unit: Any, Watch: Any, ShiftType: Any, Staff: Any
) -> None:
    """Seed historical demo data, never the platform-control tenant."""
    if Unit.query.filter_by(status="platform_control").first():
        return
    if Watch.query.count() > 0:
        for code, name in (
            ("TOUI", "TOIL (UI)"),
            ("TOU8", "TOIL (U8)"),
            ("OSS", "Operational Support"),
        ):
            ensure_shift(code, name, db=db, ShiftType=ShiftType)
        return

    watches = [
        Watch(name=f"Watch {letter}", order_index=index)
        for index, letter in enumerate("ABCDE", start=1)
    ]
    watches.append(Watch(name="Watch NOPS", order_index=6))
    db.session.add_all(watches)

    definitions = (
        ("M", "Morning", time(6), time(14), True, False, True),
        ("D", "Day", time(8), time(16), True, False, True),
        ("A", "Afternoon", time(14), time(22), True, False, True),
        ("N", "Night", time(22), time(6), True, False, True),
        ("OFF", "Rest Day", None, None, False, False, False),
        ("AL", "Annual Leave", None, None, False, False, False),
        ("PL", "Parental Leave", None, None, False, False, False),
        ("SPL", "Special Leave", None, None, False, False, False),
        ("SC", "Sick Cert", time(9), time(17), True, True, False),
        ("SSC", "Sick Self Cert", time(9), time(17), True, True, False),
        ("SBY", "Standby", time(8), time(16), True, False, False),
        ("TOUI", "TOIL (UI)", None, None, False, False, False),
        ("TOU8", "TOIL (U8)", None, None, False, False, False),
        ("OSS", "Operational Support", None, None, False, False, False),
        ("OFFICE", "Office", None, None, False, False, False),
        ("WFH", "Work from home", None, None, False, False, False),
        ("MTG", "Meeting", None, None, False, False, False),
    )
    db.session.add_all(
        ShiftType(
            code=code,
            name=name,
            start_time=start,
            end_time=end,
            is_working=working,
            is_training=training,
            is_requestable=requestable,
        )
        for code, name, start, end, working, training, requestable in definitions
    )

    names = (
        ("Alex McLean", "Bethany Kerr", "Callum Reid", "Donna Fraser", "Euan Boyd"),
        ("Fiona Watt", "Gordon Bryce", "Harris Quinn", "Isla Morton", "Jamie Lindsay"),
        ("Kara Drummond", "Lewis Pratt", "Maya Allan", "Noah Cairns", "Orla McAdam"),
        ("Poppy Neill", "Quinn Murray", "Robbie Hogg", "Sophie Duff", "Tommy Craig"),
        ("Una McKay", "Viktor Shaw", "Will Findlay", "Xander Kerr", "Yasmin Doyle"),
    )
    cycle_days = {"A": 6, "B": 4, "C": 2, "D": 10, "E": 8}
    anchor = date(2025, 9, 1)
    staff_members = []
    staff_number = 2001
    for watch_index, watch in enumerate(watches[:-1]):
        label = watch.name.removeprefix("Watch ")
        for name in names[watch_index]:
            member = Staff(
                username="admin" if staff_number == 2001 else f"user{staff_number}",
                name=name,
                staff_no=str(staff_number),
                watch=watch,
                is_operational=True,
                has_ojti=staff_number % 3 == 0,
                is_trainee=staff_number % 7 == 0,
                role="admin" if staff_number == 2001 else "user",
                leave_year_start_month=4,
                leave_entitlement_days=25,
                leave_public_holidays=8,
                leave_carryover_days=0,
            )
            member.set_password("password")
            member.pattern_anchor = anchor - timedelta(days=cycle_days[label] - 1)
            staff_members.append(member)
            staff_number += 1
    db.session.add_all(staff_members)
    db.session.commit()
