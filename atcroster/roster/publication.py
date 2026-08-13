"""Roster-publication snapshot, preflight, and notification service."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Callable

from flask import url_for


@dataclass(frozen=True)
class PublicationDependencies:
    db: Any
    Assignment: Any
    RosterPublication: Any
    Staff: Any
    Requirement: Any
    RosterRuleVersion: Any
    FatigueReport: Any
    OperationalPosition: Any
    PositionRequirement: Any
    BreakPlan: Any
    Unit: Any
    current_unit_id: Callable[[], int]
    now: Callable[[], Any]
    month_add: Callable[[int, int, int], tuple[int, int]]
    month_range: Callable[[int, int], tuple[Any, list[date]]]
    is_admin_user: Callable[[Any], bool]
    normalise_snapshot: Callable[[list[dict[str, Any]]], Any]
    get_shift: Callable[[str], Any]
    staff_has_shift_qualification: Callable[[Any, Any, date], bool]
    excluded_codes: Callable[[], set[str]]
    staff_is_countable_on: Callable[[Any, date], bool]
    shift_counter_group_for_day: Callable[..., str | None]
    night_active_on: Callable[[int, date], bool]
    compliance_findings: Callable[[int, int], dict[str, Any]]
    position_assurance: Callable[[int, int], list[dict[str, Any]]]
    valid_email: Callable[[str | None], str]
    send_account_email: Callable[[str, str, str], bool]


def create_publication_service(
    dependencies: PublicationDependencies,
) -> SimpleNamespace:
    """Build publication operations with explicit roster-domain dependencies."""
    db = dependencies.db
    Assignment = dependencies.Assignment
    RosterPublication = dependencies.RosterPublication
    Staff = dependencies.Staff
    Requirement = dependencies.Requirement
    RosterRuleVersion = dependencies.RosterRuleVersion
    FatigueReport = dependencies.FatigueReport
    OperationalPosition = dependencies.OperationalPosition
    PositionRequirement = dependencies.PositionRequirement
    BreakPlan = dependencies.BreakPlan
    Unit = dependencies.Unit
    _current_unit_id = dependencies.current_unit_id
    utcnow = dependencies.now
    _month_add = dependencies.month_add
    month_range = dependencies.month_range
    is_admin_user = dependencies.is_admin_user
    normalise_assignment_snapshot = dependencies.normalise_snapshot
    get_shift = dependencies.get_shift
    _staff_has_shift_qualification = dependencies.staff_has_shift_qualification
    get_exclude_from_counters = dependencies.excluded_codes
    staff_is_countable_on = dependencies.staff_is_countable_on
    shift_counter_group_for_day = dependencies.shift_counter_group_for_day
    _night_active_on = dependencies.night_active_on
    _compliance_findings = dependencies.compliance_findings
    _position_assurance = dependencies.position_assurance
    _valid_email = dependencies.valid_email
    _send_account_email = dependencies.send_account_email

    def _roster_snapshot(year: int, month: int) -> dict:
        start = date(year, month, 1)
        ny, nm = _month_add(year, month, 1)
        end = date(ny, nm, 1)
        assignments = (
            Assignment.query.filter(
                Assignment.unit_id == _current_unit_id(),
                Assignment.day >= start,
                Assignment.day < end,
            )
            .order_by(Assignment.staff_id, Assignment.day)
            .all()
        )
        return {
            "generated_at": utcnow().isoformat(),
            "year": year,
            "month": month,
            "assignments": [
                {
                    "staff_id": row.staff_id,
                    "day": row.day.isoformat(),
                    "code": row.code,
                    "annotation": row.annotation or "",
                }
                for row in assignments
            ],
        }

    def _can_publish_roster(user) -> bool:
        """Month publication is available to accountable operational managers."""
        return bool(
            is_admin_user(user)
            or getattr(user, "is_wm", False)
            or getattr(user, "is_dwm", False)
        )

    def _active_roster_publication(year: int, month: int):
        return (
            RosterPublication.query.filter_by(
                unit_id=_current_unit_id(),
                year=year,
                month=month,
                state="published",
            )
            .order_by(RosterPublication.version.desc())
            .first()
        )

    def _publication_matches_live_roster(publication, year: int, month: int) -> bool:
        """A changed roster returns to Draft until the new state is published."""
        if not publication:
            return False
        try:
            published_rows = json.loads(publication.snapshot_json or "{}").get(
                "assignments", []
            )
        except (TypeError, json.JSONDecodeError):
            return False
        live_rows = _roster_snapshot(year, month)["assignments"]
        try:
            return normalise_assignment_snapshot(
                published_rows
            ) == normalise_assignment_snapshot(live_rows)
        except (KeyError, TypeError, ValueError):
            return False

    def _publication_preflight(year: int, month: int) -> dict:
        _, days = month_range(year, month)
        staff = Staff.query.filter_by(is_operational=True).order_by(Staff.name).all()
        assignments = Assignment.query.filter(
            Assignment.day >= days[0], Assignment.day <= days[-1]
        ).all()
        assignment_map = {(row.staff_id, row.day): row for row in assignments}
        requirement = Requirement.query.filter_by(year=year, month=month).first()
        counts = {day: Counter() for day in days}
        qualification_gaps = []
        unassigned = []

        for person in staff:
            for day in days:
                assignment = assignment_map.get((person.id, day))
                if not assignment:
                    unassigned.append({"staff": person, "day": day})
                    continue
                shift = get_shift(assignment.code)
                if (
                    shift
                    and shift.is_working
                    and shift.required_qualification
                    and not _staff_has_shift_qualification(person, shift, day)
                ):
                    qualification_gaps.append(
                        {
                            "staff": person,
                            "day": day,
                            "shift": shift,
                            "qualification": shift.required_qualification,
                        }
                    )
                if (
                    shift
                    and shift.is_working
                    and not shift.is_training
                    and assignment.code not in get_exclude_from_counters()
                    and staff_is_countable_on(person, day)
                ):
                    group = shift_counter_group_for_day(
                        assignment.code, day, _current_unit_id()
                    )
                    if group:
                        counts[day][group] += 1

        coverage_gaps = []
        for day in days:
            for group in ("M", "D", "A", "N"):
                needed = (
                    0
                    if group == "N" and not _night_active_on(_current_unit_id(), day)
                    else int(getattr(requirement, f"req_{group.lower()}", 0) or 0)
                )
                available = counts[day][group]
                if available < needed:
                    coverage_gaps.append(
                        {
                            "day": day,
                            "group": group,
                            "available": available,
                            "needed": needed,
                            "shortfall": needed - available,
                        }
                    )

        fatigue = _compliance_findings(year, month)
        position_rows = _position_assurance(year, month)
        position_shortfalls = [row for row in position_rows if row["shortfall"]]
        approved_rule = (
            RosterRuleVersion.query.filter(
                RosterRuleVersion.state == "approved",
                db.or_(
                    RosterRuleVersion.effective_from.is_(None),
                    RosterRuleVersion.effective_from <= days[0],
                ),
            )
            .order_by(RosterRuleVersion.version.desc())
            .first()
        )
        critical_reports = FatigueReport.query.filter(
            FatigueReport.duty_day >= days[0],
            FatigueReport.duty_day <= days[-1],
            FatigueReport.severity.in_(("high", "unfit")),
            FatigueReport.status != "closed",
        ).all()
        configuration_blocks = []
        if not OperationalPosition.query.filter_by(is_active=True).first():
            configuration_blocks.append("No active operational positions configured.")
        if not PositionRequirement.query.filter(
            PositionRequirement.day >= days[0],
            PositionRequirement.day <= days[-1],
        ).first():
            configuration_blocks.append(
                "No position requirements configured for the month."
            )
        if not approved_rule:
            configuration_blocks.append(
                "No approved rostering rule version governs the month."
            )
        if not BreakPlan.query.filter(
            BreakPlan.day >= days[0], BreakPlan.day <= days[-1]
        ).first():
            configuration_blocks.append(
                "No operational break plan is recorded for the month."
            )
        # Only incomplete roster cells and known competence failures prevent a
        # release. Other findings stay visible and require a manager rationale,
        # but do not trap a unit in optional setup workflows.
        hard_blocks = len(qualification_gaps) + len(unassigned)
        return {
            "fatigue_total": fatigue["total"],
            "fatigue_critical": fatigue["critical"],
            "coverage_gaps": coverage_gaps,
            "qualification_gaps": qualification_gaps,
            "unassigned": unassigned,
            "position_assurance": position_rows,
            "position_shortfalls": position_shortfalls,
            "critical_fatigue_reports": critical_reports,
            "configuration_blocks": configuration_blocks,
            "approved_rule": approved_rule,
            "hard_blocks": hard_blocks,
            "exceptions": (
                fatigue["total"]
                + len(coverage_gaps)
                + len(position_shortfalls)
                + len(critical_reports)
                + len(configuration_blocks)
            ),
            "ready": hard_blocks == 0,
        }

    def _send_roster_publication_emails(
        unit_id: int,
        year: int,
        month: int,
        published_at: datetime,
    ) -> tuple[int, int, int]:
        """Email each registered unit user without exposing recipient addresses."""
        recipients = (
            Staff.query.filter(
                Staff.unit_id == unit_id,
                Staff.membership_status == "active",
                Staff.email.isnot(None),
                Staff.email != "",
            )
            .order_by(Staff.name)
            .all()
        )
        unit = db.session.get(Unit, unit_id)
        unit_name = (unit.name or unit.code) if unit else "your airport"
        month_name = date(year, month, 1).strftime("%B %Y")
        roster_url = url_for(
            "roster_month", ym=f"{year:04d}-{month:02d}", _external=True
        )
        subject = f"{month_name} roster published — {unit_name}"
        sent = 0
        failed = 0
        unique_addresses: set[str] = set()

        for person in recipients:
            address = _valid_email(person.email)
            if not address or address in unique_addresses:
                continue
            unique_addresses.add(address)
            body = (
                f"Hello {person.name},\n\n"
                f"The {month_name} roster for {unit_name} was published on "
                f"{published_at.strftime('%d %B %Y')}.\n\n"
                f"View the roster: {roster_url}\n\n"
                "Please sign in to ATC Roster to review your duties. If you "
                "believe anything is incorrect, contact your unit management "
                "team.\n"
            )
            if _send_account_email(address, subject, body):
                sent += 1
            else:
                failed += 1

        return sent, failed, len(unique_addresses)

    return SimpleNamespace(
        snapshot=_roster_snapshot,
        can_publish=_can_publish_roster,
        active_publication=_active_roster_publication,
        matches_live=_publication_matches_live_roster,
        preflight=_publication_preflight,
        send_emails=_send_roster_publication_emails,
    )
