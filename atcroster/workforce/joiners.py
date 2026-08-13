"""Effective-dated workforce joiner workflow."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from flask import abort, flash, redirect, url_for


@dataclass(frozen=True)
class JoinerDependencies:
    db: Any
    Staff: Any
    WorkPattern: Any
    StaffWatchHistory: Any
    StaffPatternAssignment: Any
    QualificationType: Any
    PersonQualification: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    parse_date: Callable[[str | None], Any]
    work_pattern_service: Any
    record_qualification_history: Callable[[Any, str], None]
    sync_qualification: Callable[[Any, Any, Any], None]
    record_roster_impact: Callable[..., None]
    now: Callable[[], Any]


def create_joiner(form: Any, dependencies: JoinerDependencies):
    db = dependencies.db
    Staff = dependencies.Staff
    WorkPattern = dependencies.WorkPattern
    StaffWatchHistory = dependencies.StaffWatchHistory
    StaffPatternAssignment = dependencies.StaffPatternAssignment
    QualificationType = dependencies.QualificationType
    PersonQualification = dependencies.PersonQualification
    RosterImpactEventType = dependencies.RosterImpactEventType
    _current_unit_id = dependencies.current_unit_id
    _parse_date = dependencies.parse_date
    work_pattern_service = dependencies.work_pattern_service
    _record_qualification_history = dependencies.record_qualification_history
    _sync_qualification_to_roster_profile = dependencies.sync_qualification
    record_roster_impact = dependencies.record_roster_impact
    utcnow = dependencies.now
    name = form.get("name", "").strip()
    staff_no = form.get("staff_no", "").strip()
    username = form.get("username", "").strip()
    watch_id = form.get("watch_id")
    role = form.get("role", "user")
    try:
        employment_start = date.fromisoformat(
            form.get("employment_start_date") or date.today().isoformat()
        )
        unit_join = date.fromisoformat(
            form.get("unit_join_date") or employment_start.isoformat()
        )
        roster_start = date.fromisoformat(
            form.get("roster_start_date") or unit_join.isoformat()
        )
        pattern_anchor = date.fromisoformat(
            form.get("pattern_anchor") or roster_start.isoformat()
        )
        anchor_day_index = int(form.get("anchor_day_index") or 0)
        contracted_hours = float(form.get("contracted_hours_per_week") or 37)
    except (TypeError, ValueError):
        flash("Enter valid joining dates, pattern alignment and hours.", "error")
        return redirect(url_for("admin") + "#staff")
    employment_type = (form.get("employment_type") or "FULL_TIME").strip().upper()
    if employment_type not in {"FULL_TIME", "PART_TIME"}:
        abort(400, "Invalid employment type.")
    if not (
        employment_start <= unit_join <= roster_start
        and 0 <= contracted_hours <= 168
        and anchor_day_index >= 0
    ):
        flash(
            "Employment, joining and roster dates must be in order; "
            "hours and cycle day must be valid.",
            "error",
        )
        return redirect(url_for("admin") + "#staff")
    try:
        work_pattern_id = int(form.get("work_pattern_id") or 0)
    except (TypeError, ValueError):
        work_pattern_id = 0
    selected_pattern = WorkPattern.query.filter_by(
        id=work_pattern_id, unit_id=_current_unit_id(), is_active=True
    ).first()

    # NEW flags
    is_wm = bool(form.get("is_wm"))
    is_dwm = bool(form.get("is_dwm"))
    exclude_from_ot = bool(form.get("exclude_from_ot"))
    permissions = {
        "edit_roster": bool(form.get("permission_edit_roster")),
        "apply_annotations": bool(form.get("permission_apply_annotations")),
    }

    # Leave/TOIL config
    leave_year_start_month = int(form.get("leave_year_start_month", 4) or 4)
    leave_entitlement_days = int(form.get("leave_entitlement_days", 0) or 0)
    leave_public_holidays = int(form.get("leave_public_holidays", 0) or 0)
    leave_carryover_days = int(form.get("leave_carryover_days", 0) or 0)

    if not all([name, staff_no, watch_id, selected_pattern]):
        flash(
            "Name, staff number, watch and working pattern are required.",
            "error",
        )
    elif Staff.query.filter_by(
        unit_id=_current_unit_id(), staff_no=staff_no
    ).first() or (
        username
        and Staff.query.filter(
            Staff.unit_id == _current_unit_id(),
            db.func.lower(Staff.username) == username.lower(),
        ).first()
    ):
        flash("Username or Staff # already exists.", "error")
    else:
        username = username or (f"person-{_current_unit_id()}-{secrets.token_hex(8)}")
        s = Staff(
            name=name,
            staff_no=staff_no,
            username=username,
            watch_id=int(watch_id),
            role=role,
            is_wm=is_wm,
            is_dwm=is_dwm,
            permissions_json=json.dumps(permissions, sort_keys=True),
            exclude_from_ot=exclude_from_ot,
            leave_year_start_month=leave_year_start_month,
            leave_entitlement_days=leave_entitlement_days,
            leave_public_holidays=leave_public_holidays,
            leave_carryover_days=leave_carryover_days,
            employment_start_date=employment_start,
            unit_join_date=unit_join,
            roster_start_date=roster_start,
            employment_type=employment_type,
            contracted_minutes_per_week=int(round(contracted_hours * 60)),
            workforce_notes=(form.get("workforce_notes") or "").strip()[:2000],
            is_operational=bool(form.get("operational")),
            is_trainee=bool(form.get("trainee")),
            medical_expiry=_parse_date(form.get("medical_expiry")),
        )
        s.set_password("password")
        if not s.calendar_token:
            s.calendar_token = secrets.token_hex(16)
        db.session.add(s)
        db.session.flush()
        db.session.add(
            StaffWatchHistory(
                unit_id=s.unit_id,
                staff_id=s.id,
                watch_id=int(watch_id),
                effective_date=roster_start,
            )
        )
        pattern_assignment = StaffPatternAssignment(
            unit_id=s.unit_id,
            staff_id=s.id,
            work_pattern_id=selected_pattern.id,
            effective_from=roster_start,
            anchor_date=pattern_anchor,
            anchor_day_index=anchor_day_index,
            contracted_minutes_override=int(
                round(contracted_hours * 60 * selected_pattern.cycle_length_days / 7)
            ),
            notes="Initial assignment created by unit joiner workflow.",
        )
        work_pattern_service.validate_staff_pattern_assignment(pattern_assignment)
        db.session.add(pattern_assignment)
        selected_qualification_ids = {
            int(value)
            for value in form.getlist("qualification_ids")
            if str(value).isdigit()
        }
        medical_type = QualificationType.query.filter_by(
            unit_id=s.unit_id, code="MEDICAL", is_active=True
        ).first()
        if s.medical_expiry and medical_type:
            selected_qualification_ids.add(medical_type.id)
        qualification_rows = (
            QualificationType.query.filter(
                QualificationType.unit_id == s.unit_id,
                QualificationType.id.in_(selected_qualification_ids),
                QualificationType.is_active.is_(True),
            ).all()
            if selected_qualification_ids
            else []
        )
        for qtype in qualification_rows:
            raw_expiry = form.get(f"qualification_expiry_{qtype.id}")
            expiry = _parse_date(raw_expiry)
            if qtype.code == "MEDICAL" and s.medical_expiry:
                expiry = s.medical_expiry
            if qtype.expiry_required and not expiry:
                abort(400, f"{qtype.code} requires an expiry date.")
            qualification = PersonQualification(
                unit_id=s.unit_id,
                person_id=s.id,
                qualification_type_id=qtype.id,
                valid_from=roster_start,
                expires_on=expiry,
                status="valid",
                updated_at=utcnow(),
            )
            db.session.add(qualification)
            db.session.flush()
            _record_qualification_history(qualification, "joiner_created")
            _sync_qualification_to_roster_profile(s, qtype, expiry)
        record_roster_impact(
            RosterImpactEventType.UNIT_JOINER,
            roster_start,
            staff_ids=[s.id],
            rebuild_baseline=True,
            reason="New unit member and initial working arrangement created.",
        )
        db.session.commit()
        flash(
            "Roster profile created. Complete the profile, then "
            "issue account access when ready.",
            "ok",
        )
        return redirect(url_for("admin_staff_edit", sid=s.id))
