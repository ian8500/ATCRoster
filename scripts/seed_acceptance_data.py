#!/usr/bin/env python3
"""Create a deterministic, non-production ATCRoster acceptance database.

The dataset is intentionally rebuilt from scratch. It uses dates relative to
the day it is generated so request windows, expiry warnings and current-month
views remain useful whenever the script is run.
"""
from __future__ import annotations

import argparse
import json
import os
from calendar import monthrange
from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PASSWORD = "Test-ATCRoster-2026!"
AIRPORTS = (
    # Each unit has exactly one spare account, making the limit test quick.
    ("LBA", "Leeds Bradford Airport", 16, 17),
    ("EMA", "East Midlands Airport", 14, 15),
    ("INV", "Inverness Airport", 12, 13),
)
FIRST_NAMES = (
    "Alex", "Beth", "Callum", "Deepa", "Euan", "Fiona", "Gareth", "Hannah",
    "Imran", "Jenny", "Kieran", "Lucy", "Martin", "Nadia", "Owen", "Priya",
)
SURNAMES = (
    "Taylor", "Singh", "Murray", "Campbell", "Roberts", "Wilson", "Fraser",
    "Ahmed", "Davies", "Brown", "Khan", "Evans", "Reid", "Morgan", "Clark",
    "Stewart",
)


def add_months(day: date, count: int) -> date:
    total = day.year * 12 + day.month - 1 + count
    return date(total // 12, total % 12 + 1, 1)


def month_days(first: date) -> list[date]:
    return [
        date(first.year, first.month, day)
        for day in range(1, monthrange(first.year, first.month)[1] + 1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=str(REPO_ROOT / "instance" / "acceptance.db"),
        help="SQLite file to create (default: instance/acceptance.db)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Replace the target database if it already exists.",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Dataset reference date in YYYY-MM-DD format (default: today).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.database).expanduser().resolve()
    if db_path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        raise SystemExit("Acceptance data must target an explicit SQLite database file.")
    if db_path.exists() and not args.reset:
        raise SystemExit(f"{db_path} already exists; use --reset to replace it.")
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["ATCROSTER_ENV"] = "development"
    os.environ["ATCROSTER_SKIP_RUNTIME_SCHEMA"] = "1"
    os.environ.setdefault("FLASK_SECRET_KEY", "acceptance-only-secret-never-deploy")

    import app as roster

    current_month = args.as_of.replace(day=1)
    previous_month = add_months(current_month, -1)
    next_month = add_months(current_month, 1)
    request_month = add_months(current_month, 2)
    months = (previous_month, current_month, next_month, request_month)

    with roster.app.app_context():
        roster.db.drop_all()
        roster.db.create_all()

        control = roster.Unit(
            code="PLATFORM",
            name="ATCRoster Platform Control",
            status="platform_control",
            plan="internal",
            active_user_limit=5,
        )
        roster.db.session.add(control)
        roster.db.session.flush()
        platform_user = roster.Staff(
            unit_id=control.id,
            username="platform.admin",
            name="Platform Super Administrator",
            staff_no="PLATFORM-001",
            role="superadmin",
            is_operational=False,
        )
        platform_user.set_password(PASSWORD)
        roster.db.session.add(platform_user)
        roster.db.session.flush()
        platform_identity = roster.PlatformIdentity(
            public_id="platform-acceptance-admin",
            username=platform_user.username,
            password_hash=platform_user.password_hash,
        )
        roster.db.session.add(platform_identity)
        roster.db.session.flush()

        credentials = [{
            "airport": "Platform control",
            "role": "Super Admin",
            "username": platform_user.username,
            "password": PASSWORD,
        }]
        summaries = []

        for airport_index, (code, name, staff_count, account_limit) in enumerate(AIRPORTS):
            unit = roster.Unit(
                code=code,
                name=name,
                status="active",
                plan="professional",
                active_user_limit=account_limit,
                request_months_ahead=3,
                request_lock_day=20,
                onboarding_step=10,
                branding_json=json.dumps({"short_name": code}),
                last_active_at=roster.utcnow(),
            )
            roster.db.session.add(unit)
            roster.db.session.flush()

            watches = []
            for watch_index, watch_name in enumerate(("Blue", "Red", "Green", "Gold"), 1):
                watch = roster.Watch(
                    unit_id=unit.id,
                    name=f"{watch_name} Watch",
                    order_index=watch_index,
                )
                roster.db.session.add(watch)
                watches.append(watch)
            roster.db.session.flush()

            shift_specs = (
                ("M", "Morning", time(6), time(14), True, True),
                ("D", "Day", time(8), time(16), True, True),
                ("A", "Afternoon", time(14), time(22), True, True),
                ("N", "Night", time(22), time(6), True, True),
                ("SBY", "Standby", time(8), time(16), True, True),
                ("TRG", "Training", time(9), time(17), True, False),
                ("OFF", "Rest day", None, None, False, False),
                ("AL", "Annual leave", None, None, False, False),
                ("SC", "Sickness", None, None, False, False),
                ("TOUI", "TOIL half day", None, None, False, False),
                ("TOU8", "TOIL full day", None, None, False, False),
            )
            for shift_code, label, start, end, working, requestable in shift_specs:
                roster.db.session.add(roster.ShiftType(
                    unit_id=unit.id,
                    code=shift_code,
                    name=label,
                    start_time=start,
                    end_time=end,
                    is_working=working,
                    is_training=shift_code == "TRG",
                    is_requestable=requestable,
                    required_qualification="medical" if working else "",
                ))

            for order, (annotation_code, label, category, colour) in enumerate((
                ("A6", "Six-hour extension", "Extension", "#7c3aed"),
                ("OT", "Overtime", "Overtime", "#b45309"),
                ("TOAI", "TOIL accrued half day", "TOIL Accrual", "#047857"),
                ("TOA8", "TOIL accrued full day", "TOIL Accrual", "#065f46"),
            )):
                roster.db.session.add(roster.AnnotationType(
                    unit_id=unit.id,
                    code=annotation_code,
                    label=label,
                    category=category,
                    colour=colour,
                    description=f"Acceptance-test {label.lower()} annotation.",
                    allow_suffix=annotation_code == "A6",
                    suffixes="M,D,A,N" if annotation_code == "A6" else "",
                    toil_half_days={"TOAI": 1, "TOA8": 2}.get(annotation_code, 0),
                    tags="acceptance",
                    sort_order=order,
                ))

            qualification_types = {}
            for qual_code, label in (
                ("MED", "Class 3 Medical"),
                ("ADI", "Aerodrome Control Instrument"),
                ("OJTI", "On-the-job Training Instructor"),
            ):
                qtype = roster.QualificationType(
                    unit_id=unit.id,
                    code=qual_code,
                    label=label,
                    warning_days_csv="180,90,60,30",
                )
                roster.db.session.add(qtype)
                qualification_types[qual_code] = qtype
            roster.db.session.flush()

            people = []
            for index in range(staff_count):
                role = "admin" if index == 0 else "editor" if index == 1 else "user"
                username = (
                    f"{code.lower()}.admin" if index == 0
                    else f"{code.lower()}.editor" if index == 1
                    else f"{code.lower()}.atco{index - 1:02d}"
                )
                person = roster.Staff(
                    unit_id=unit.id,
                    username=username,
                    name=f"{FIRST_NAMES[index]} {SURNAMES[(index + airport_index * 3) % len(SURNAMES)]}",
                    staff_no=f"{code}-{100 + index}",
                    role=role,
                    membership_status="active",
                    watch=watches[index % len(watches)],
                    pattern_csv="M,M,A,A,N,N,OFF,OFF,OFF,OFF",
                    pattern_anchor=current_month - timedelta(days=index % 10),
                    medical_expiry=args.as_of + timedelta(days=(20 if index == 2 else 400)),
                    tower_ue_expiry=args.as_of + timedelta(days=(80 if index == 3 else 500)),
                    radar_ue_expiry=args.as_of - timedelta(days=5) if index == 4 else None,
                    has_ojti=index % 4 == 0,
                    has_assessor=index % 6 == 0,
                    is_wm=index in (0, 4, 8),
                    is_dwm=index in (1, 5, 9),
                    is_trainee=index == staff_count - 1,
                    is_operational=True,
                    leave_entitlement_days=25,
                    leave_public_holidays=8,
                    leave_carryover_days=index % 3,
                    toil_half_days=index % 5,
                )
                person.set_password(PASSWORD)
                roster.db.session.add(person)
                people.append(person)
            roster.db.session.flush()

            for person_index, person in enumerate(people):
                identity = roster.PlatformIdentity(
                    public_id=f"acceptance-{code.lower()}-{person_index:02d}",
                    username=person.username,
                    password_hash=person.password_hash,
                )
                roster.db.session.add(identity)
                roster.db.session.flush()
                roster.db.session.add(roster.UnitMembership(
                    identity_id=identity.id,
                    unit_id=unit.id,
                    person_id=person.id,
                    role={
                        "admin": "UnitAdmin",
                        "editor": "RosterEditor",
                        "user": "StaffUser",
                    }[person.role],
                    status="active",
                    activated_at=roster.utcnow(),
                ))
                for qual_code in ("MED", "ADI"):
                    expires = args.as_of + timedelta(days=400 + person_index)
                    if qual_code == "MED" and person_index == 2:
                        expires = args.as_of + timedelta(days=20)
                    roster.db.session.add(roster.PersonQualification(
                        unit_id=unit.id,
                        person_id=person.id,
                        qualification_type_id=qualification_types[qual_code].id,
                        expires_on=expires,
                        status="valid",
                    ))
                if person.has_ojti:
                    roster.db.session.add(roster.PersonQualification(
                        unit_id=unit.id,
                        person_id=person.id,
                        qualification_type_id=qualification_types["OJTI"].id,
                        expires_on=args.as_of + timedelta(days=500),
                        status="valid",
                    ))

            positions = []
            for position_code, label in (
                ("TWR", "Aerodrome Control"),
                ("GMC", "Ground Movement Control"),
                ("APP", "Approach Control"),
            ):
                position = roster.OperationalPosition(
                    unit_id=unit.id,
                    code=position_code,
                    label=label,
                    description=f"{label} operational position for {code}.",
                )
                roster.db.session.add(position)
                positions.append(position)
            roster.db.session.flush()
            for person in people:
                for position in positions[:2]:
                    roster.db.session.add(roster.PositionEndorsement(
                        unit_id=unit.id,
                        person_id=person.id,
                        position_id=position.id,
                        valid_from=previous_month,
                        valid_until=args.as_of + timedelta(days=500),
                        status="valid",
                    ))
            for person in people[::2]:
                roster.db.session.add(roster.PositionEndorsement(
                    unit_id=unit.id,
                    person_id=person.id,
                    position_id=positions[2].id,
                    valid_from=previous_month,
                    valid_until=args.as_of + timedelta(days=500),
                    status="valid",
                ))

            for month in months:
                roster.db.session.add(roster.Requirement(
                    unit_id=unit.id,
                    year=month.year,
                    month=month.month,
                    req_m=1,
                    req_d=1,
                    req_a=1,
                    req_n=1,
                ))
                for day_index, duty_day in enumerate(month_days(month)):
                    for person_index, person in enumerate(people):
                        pattern = ("M", "M", "A", "A", "N", "N",
                                   "OFF", "OFF", "OFF", "OFF")
                        code_index = (
                            duty_day - person.pattern_anchor
                        ).days % len(pattern)
                        duty_code = pattern[code_index]
                        roster.db.session.add(roster.Assignment(
                            unit_id=unit.id,
                            staff_id=person.id,
                            day=duty_day,
                            code=duty_code,
                            source="acceptance",
                            annotation="OT" if day_index == 4 and person_index == 5 else "",
                        ))
                    for shift_code, position in (("M", positions[0]), ("A", positions[1])):
                        roster.db.session.add(roster.PositionRequirement(
                            unit_id=unit.id,
                            day=duty_day,
                            shift_code=shift_code,
                            position_id=position.id,
                            required_count=1,
                            contingency_count=0,
                        ))

            leave_start = current_month + timedelta(days=9)
            sickness_day = current_month + timedelta(days=4)
            roster.db.session.add(roster.Leave(
                unit_id=unit.id,
                staff_id=people[6].id,
                leave_type="AL",
                start=leave_start,
                end=leave_start + timedelta(days=2),
            ))
            roster.db.session.add(roster.Sickness(
                unit_id=unit.id,
                staff_id=people[7].id,
                start=sickness_day,
                end=sickness_day + timedelta(days=1),
                code="SC",
            ))
            for person, duty_day, code_value in (
                (people[6], leave_start, "AL"),
                (people[6], leave_start + timedelta(days=1), "AL"),
                (people[6], leave_start + timedelta(days=2), "AL"),
                (people[7], sickness_day, "SC"),
                (people[7], sickness_day + timedelta(days=1), "SC"),
            ):
                assignment = roster.Assignment.query.execution_options(
                    skip_tenant_scope=True
                ).filter_by(unit_id=unit.id, staff_id=person.id, day=duty_day).one()
                assignment.code = code_value
                assignment.source = "acceptance"

            request_day = request_month + timedelta(days=6 + airport_index)
            for request_index, status in enumerate(("pending", "approved", "rejected")):
                roster.db.session.add(roster.ShiftRequest(
                    unit_id=unit.id,
                    staff_id=people[2 + request_index].id,
                    day=request_day + timedelta(days=request_index),
                    code=("D", "A", "M")[request_index],
                    requester_comment=f"Acceptance {status} request example.",
                    status=status,
                    admin_response=(
                        "" if status == "pending"
                        else f"Acceptance manager response: {status}."
                    ),
                    responded_by_id=None if status == "pending" else people[0].id,
                    responded_at=None if status == "pending" else roster.utcnow(),
                ))

            roster.db.session.add(roster.BreakPlan(
                unit_id=unit.id,
                day=args.as_of,
                person_id=people[2].id,
                position_id=positions[0].id,
                start_time=time(10, 0),
                end_time=time(10, 30),
                recorded_by_id=people[0].id,
            ))
            roster.db.session.add(roster.AchievedDuty(
                unit_id=unit.id,
                person_id=people[3].id,
                day=args.as_of - timedelta(days=1),
                actual_start=datetime.combine(args.as_of - timedelta(days=1), time(6)),
                actual_end=datetime.combine(args.as_of - timedelta(days=1), time(14, 15)),
                duty_type="operational",
                variance_reason="Handover extended by fifteen minutes.",
                recorded_by_id=people[0].id,
            ))
            roster.db.session.add_all([
                roster.FatigueReport(
                    unit_id=unit.id,
                    person_id=people[4].id,
                    duty_day=args.as_of,
                    severity="medium",
                    summary="Reduced sleep following an unexpected domestic disturbance.",
                    status="open",
                ),
                roster.FatigueReport(
                    unit_id=unit.id,
                    person_id=people[5].id,
                    duty_day=args.as_of - timedelta(days=3),
                    severity="low",
                    summary="Tired near the end of a sequence; discussed with supervisor.",
                    status="closed",
                    manager_response="Reviewed; restorative rest confirmed before next duty.",
                    reviewed_by_id=people[0].id,
                    reviewed_at=roster.utcnow(),
                    closed_at=roster.utcnow(),
                ),
            ])
            roster.db.session.add(roster.RosterRuleVersion(
                unit_id=unit.id,
                version=1,
                name="Acceptance rostering rules",
                rules_json=json.dumps({
                    "maximum_consecutive_duties": 6,
                    "minimum_rest_hours": 12,
                    "night_duty_limit": 3,
                }),
                state="approved",
                effective_from=previous_month,
                change_reference=f"{code}-ACCEPT-001",
                consultation_summary=(
                    "Acceptance dataset rule version approved for repeatable "
                    "workflow and publication testing."
                ),
                approved_by_id=people[0].id,
                approved_at=roster.utcnow(),
            ))
            roster.db.session.add(roster.Scenario(
                unit_id=unit.id,
                name="Summer traffic uplift",
                changes_json=json.dumps([
                    {"date": next_month.isoformat(), "shift": "D", "delta": 1},
                    {"date": (next_month + timedelta(days=1)).isoformat(), "shift": "A", "delta": 1},
                ]),
                created_by_id=people[1].id,
            ))
            publication = roster.RosterPublication(
                unit_id=unit.id,
                year=previous_month.year,
                month=previous_month.month,
                version=1,
                state="published",
                snapshot_json=json.dumps({
                    "airport": code,
                    "month": previous_month.strftime("%Y-%m"),
                    "release_assurance": {"acceptance_dataset": True},
                }),
                published_at=roster.utcnow(),
            )
            roster.db.session.add(publication)
            roster.db.session.flush()
            roster.db.session.add(roster.RosterAcknowledgement(
                unit_id=unit.id,
                publication_id=publication.id,
                person_id=people[2].id,
            ))
            roster.db.session.add_all([
                roster.Notification(
                    unit_id=unit.id,
                    recipient_id=people[2].id,
                    kind="roster_published",
                    message="Previous-month roster version 1 is ready to review.",
                ),
                roster.Notification(
                    unit_id=unit.id,
                    recipient_id=people[3].id,
                    kind="request_update",
                    message="Your shift request has been approved.",
                ),
            ])
            roster.db.session.add(roster.DatabaseRoutingMetadata(
                unit_id=unit.id,
                secret_name=f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL",
                health="healthy",
                migration_version="20260724_02",
                storage_bytes=2_000_000 + airport_index * 500_000,
            ))
            for feature in ("operational_assurance", "scenario_planning", "sms_notifications"):
                roster.db.session.add(roster.FeatureFlag(
                    unit_id=unit.id,
                    key=feature,
                    enabled=feature != "sms_notifications",
                ))
            roster.db.session.add(roster.PlanHistory(
                unit_id=unit.id,
                plan=unit.plan,
                active_user_limit=unit.active_user_limit,
                changed_by_identity_id=platform_identity.id,
            ))
            roster.db.session.add(roster.AggregateUsageEvent(
                unit_id=unit.id,
                event_type="monthly_active_users",
                count=staff_count,
            ))
            roster.db.session.add(roster.SuperAdminAudit(
                actor_identity_id=platform_identity.id,
                unit_id=unit.id,
                action="acceptance_dataset_created",
                safe_summary=f"{code} acceptance account prepared.",
            ))

            credentials.extend([
                {
                    "airport": name,
                    "role": "Unit Admin",
                    "username": people[0].username,
                    "password": PASSWORD,
                },
                {
                    "airport": name,
                    "role": "Roster Editor",
                    "username": people[1].username,
                    "password": PASSWORD,
                },
                {
                    "airport": name,
                    "role": "Staff User",
                    "username": people[2].username,
                    "password": PASSWORD,
                },
                {
                    "airport": name,
                    "role": "Watch Manager",
                    "username": people[4].username,
                    "password": PASSWORD,
                },
                {
                    "airport": name,
                    "role": "Duty Watch Manager",
                    "username": people[5].username,
                    "password": PASSWORD,
                },
            ])
            summaries.append({
                "airport": code,
                "staff": staff_count,
                "account_limit": account_limit,
                "test_month": current_month.strftime("%Y-%m"),
                "request_month": request_month.strftime("%Y-%m"),
            })

        roster.db.session.commit()

        manifest = {
            "generated_as_of": args.as_of.isoformat(),
            "database": str(db_path),
            "password_notice": "Acceptance use only. Never deploy these credentials.",
            "credentials": credentials,
            "airports": summaries,
        }
        manifest_path = db_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        counts = {
            "airports": roster.Unit.query.filter(
                roster.Unit.status != "platform_control"
            ).count(),
            "staff": roster.Staff.query.execution_options(
                skip_tenant_scope=True
            ).filter_by(is_operational=True).count(),
            "assignments": roster.Assignment.query.execution_options(
                skip_tenant_scope=True
            ).count(),
            "requests": roster.ShiftRequest.query.execution_options(
                skip_tenant_scope=True
            ).count(),
        }
        print(json.dumps({
            "status": "created",
            "database": str(db_path),
            "manifest": str(manifest_path),
            "months": [month.strftime("%Y-%m") for month in months],
            "counts": counts,
        }, indent=2))


if __name__ == "__main__":
    main()
