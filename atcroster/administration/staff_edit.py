"""Staff profile administration route."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required


@dataclass(frozen=True)
class StaffEditDependencies:
    db: Any
    Staff: Any
    Watch: Any
    QualificationType: Any
    PersonQualification: Any
    UnitMembership: Any
    PlatformIdentity: Any
    SecureInvitation: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    parse_date: Callable[[str | None], Any]
    valid_email: Callable[[str], str]
    normalise_phone: Callable[[str | None], str]
    validate_pattern: Callable[[str | None], list[str]]
    now: Callable[[], Any]
    record_qualification_history: Callable[[Any, str], None]
    record_roster_impact: Callable[..., None]
    user_permissions: Callable[[Any], dict[str, bool]]
    admin_required: Callable[[Callable[..., Any]], Callable[..., Any]]
    pattern_codes: Any


def create_staff_edit_blueprint(dependencies: StaffEditDependencies) -> Blueprint:
    blueprint = Blueprint("staff_edit_administration", __name__)
    db = dependencies.db
    Staff = dependencies.Staff
    Watch = dependencies.Watch
    QualificationType = dependencies.QualificationType
    PersonQualification = dependencies.PersonQualification
    UnitMembership = dependencies.UnitMembership
    PlatformIdentity = dependencies.PlatformIdentity
    SecureInvitation = dependencies.SecureInvitation
    RosterImpactEventType = dependencies.RosterImpactEventType
    _current_unit_id = dependencies.current_unit_id
    _parse_date = dependencies.parse_date
    _valid_email = dependencies.valid_email
    _normalise_phone_number = dependencies.normalise_phone
    _validated_pattern = dependencies.validate_pattern
    utcnow = dependencies.now
    _record_qualification_history = dependencies.record_qualification_history
    record_roster_impact = dependencies.record_roster_impact
    user_permissions = dependencies.user_permissions
    PATTERN_CODES = dependencies.pattern_codes

    @login_required
    @dependencies.admin_required
    def admin_staff_edit(sid):
        # remove: if not is_admin_user(current_user): ...
        ...

        s = (
            Staff.query.filter_by(id=sid, unit_id=_current_unit_id())
            .filter(Staff.role != "position_monitor")
            .first_or_404()
        )
        if request.method == "POST":
            old_profile = {
                "watch_id": s.watch_id,
                "is_operational": bool(s.is_operational),
                "pattern_override": bool(s.pattern_override),
                "pattern_csv": s.pattern_csv,
                "pattern_anchor": s.pattern_anchor,
                "medical_expiry": s.medical_expiry,
                "ue_expiries": (s.tower_ue_expiry, s.radar_ue_expiry, s.met_ue_expiry),
                "has_ojti": bool(s.has_ojti),
                "has_assessor": bool(s.has_assessor),
                "employment_type": s.employment_type,
                "contracted_minutes_per_week": s.contracted_minutes_per_week,
            }
            try:
                s.employment_start_date = _parse_date(
                    request.form.get("employment_start_date")
                )
                s.unit_join_date = _parse_date(request.form.get("unit_join_date"))
                s.roster_start_date = _parse_date(request.form.get("roster_start_date"))
                employment_type = (
                    (
                        request.form.get("employment_type")
                        or s.employment_type
                        or "FULL_TIME"
                    )
                    .strip()
                    .upper()
                )
                if employment_type not in {"FULL_TIME", "PART_TIME"}:
                    raise ValueError
                contracted_hours = float(
                    request.form.get("contracted_hours_per_week")
                    or ((s.contracted_minutes_per_week or 0) / 60)
                )
                if not 0 <= contracted_hours <= 168:
                    raise ValueError
                if (
                    s.employment_start_date
                    and s.unit_join_date
                    and s.employment_start_date > s.unit_join_date
                ) or (
                    s.unit_join_date
                    and s.roster_start_date
                    and s.unit_join_date > s.roster_start_date
                ):
                    raise ValueError
                s.employment_type = employment_type
                s.contracted_minutes_per_week = int(round(contracted_hours * 60))
                s.workforce_notes = (request.form.get("workforce_notes") or "").strip()[
                    :2000
                ]
            except (TypeError, ValueError):
                flash("Enter valid employment dates and contracted hours.", "error")
                return redirect(url_for("admin_staff_edit", sid=s.id))
            s.name = request.form.get("name", s.name).strip()
            s.staff_no = request.form.get("staff_no", s.staff_no).strip()
            s.caa_license_number = (
                request.form.get("caa_license_number") or s.caa_license_number or ""
            ).strip()[:40]
            s.username = request.form.get("username", s.username).strip()
            submitted_email = request.form.get("email", s.email)
            if submitted_email and not _valid_email(submitted_email):
                flash("Enter a valid email address.", "error")
                return redirect(url_for("admin_staff_edit", sid=s.id))
            s.email = _valid_email(submitted_email)
            s.phone_number = _normalise_phone_number(
                request.form.get("phone_number", s.phone_number)
            )
            s.watch_id = int(request.form.get("watch_id", s.watch_id or 0)) or None

            s.is_operational = bool(request.form.get("operational"))
            s.is_trainee = bool(request.form.get("trainee"))
            s.has_ojti = bool(request.form.get("ojti"))
            s.has_assessor = bool(request.form.get("has_assessor"))

            # NEW flags
            s.is_wm = bool(request.form.get("is_wm"))
            s.is_dwm = bool(request.form.get("is_dwm"))
            s.exclude_from_ot = bool(request.form.get("exclude_from_ot"))
            s.permissions_json = json.dumps(
                {
                    "edit_roster": bool(request.form.get("permission_edit_roster")),
                    "apply_annotations": bool(
                        request.form.get("permission_apply_annotations")
                    ),
                },
                sort_keys=True,
            )

            # update role
            s.role = request.form.get("role", s.role)

            s.pattern_override = bool(request.form.get("pattern_override"))
            requested_pattern = _validated_pattern(request.form.get("pattern_csv"))
            if s.pattern_override and not requested_pattern:
                flash(
                    "A personal pattern must contain M, A, D, N or OFF.",
                    "error",
                )
                return redirect(url_for("admin_staff_edit", sid=s.id))
            s.pattern_csv = ",".join(requested_pattern)
            s.pattern_anchor = _parse_date(request.form.get("pattern_anchor"))

            s.medical_expiry = _parse_date(request.form.get("medical_expiry"))
            s.tower_ue_expiry = _parse_date(request.form.get("tower_ue_expiry"))
            s.radar_ue_expiry = _parse_date(request.form.get("radar_ue_expiry"))
            s.met_ue_expiry = _parse_date(request.form.get("met_ue_expiry"))
            for code, expiry in {
                "MEDICAL": s.medical_expiry,
                "ADI": s.tower_ue_expiry,
                "APS": s.radar_ue_expiry,
                "MET": s.met_ue_expiry,
            }.items():
                qtype = QualificationType.query.filter_by(
                    unit_id=s.unit_id, code=code
                ).first()
                if not qtype:
                    continue
                qualification = PersonQualification.query.filter_by(
                    unit_id=s.unit_id,
                    person_id=s.id,
                    qualification_type_id=qtype.id,
                ).first()
                if not qualification and expiry:
                    qualification = PersonQualification(
                        unit_id=s.unit_id,
                        person_id=s.id,
                        qualification_type_id=qtype.id,
                        status="valid",
                    )
                    db.session.add(qualification)
                    db.session.flush()
                if qualification:
                    qualification.expires_on = expiry
                    qualification.updated_at = utcnow()
                    _record_qualification_history(
                        qualification, "roster_profile_updated"
                    )

            s.tower_ut = bool(request.form.get("tower_ut"))
            s.radar_ut = bool(request.form.get("radar_ut"))
            s.met_ut = bool(request.form.get("met_ut"))

            # Leave-year config
            s.leave_year_start_month = int(
                request.form.get(
                    "leave_year_start_month", s.leave_year_start_month or 4
                )
                or 4
            )
            s.leave_entitlement_days = int(
                request.form.get(
                    "leave_entitlement_days", s.leave_entitlement_days or 0
                )
                or 0
            )
            s.leave_public_holidays = int(
                request.form.get("leave_public_holidays", s.leave_public_holidays or 0)
                or 0
            )
            s.leave_carryover_days = int(
                request.form.get("leave_carryover_days", s.leave_carryover_days or 0)
                or 0
            )

            if request.form.get("reset_password"):
                s.set_password("password")

            if request.form.get("reset_calendar_token"):
                s.calendar_token = secrets.token_hex(16)

            try:
                membership = UnitMembership.query.filter_by(
                    unit_id=s.unit_id, person_id=s.id, status="active"
                ).first()
                if membership:
                    identity = db.session.get(PlatformIdentity, membership.identity_id)
                    if identity:
                        identity.email = s.email
                today = date.today()
                if old_profile["watch_id"] != s.watch_id:
                    record_roster_impact(
                        RosterImpactEventType.WATCH_TRANSFER,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=True,
                        reason="Staff watch changed in roster profile.",
                    )
                if old_profile["is_operational"] != bool(s.is_operational):
                    event_type = (
                        RosterImpactEventType.OPERATIONAL_ROSTER_ACTIVATION
                        if s.is_operational
                        else RosterImpactEventType.OPERATIONAL_ROSTER_DEACTIVATION
                    )
                    record_roster_impact(
                        event_type,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=True,
                        reason="Operational roster status changed.",
                    )
                if (old_profile["pattern_override"], old_profile["pattern_csv"]) != (
                    bool(s.pattern_override),
                    s.pattern_csv,
                ):
                    record_roster_impact(
                        RosterImpactEventType.WORK_PATTERN_CHANGE,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=True,
                        reason="Personal roster pattern changed.",
                    )
                if old_profile["pattern_anchor"] != s.pattern_anchor:
                    record_roster_impact(
                        RosterImpactEventType.PATTERN_ANCHOR_CHANGE,
                        s.pattern_anchor or today,
                        staff_ids=[s.id],
                        rebuild_baseline=True,
                        reason="Personal roster pattern anchor changed.",
                    )
                old_medical = old_profile["medical_expiry"]
                if old_medical != s.medical_expiry:
                    is_valid = bool(s.medical_expiry and s.medical_expiry >= today)
                    record_roster_impact(
                        RosterImpactEventType.MEDICAL_RESTORED
                        if is_valid
                        else RosterImpactEventType.MEDICAL_EXPIRED,
                        today
                        if is_valid
                        else (s.medical_expiry or old_medical or today),
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="Medical validity changed in roster profile.",
                    )
                old_ue = old_profile["ue_expiries"]
                new_ue = (s.tower_ue_expiry, s.radar_ue_expiry, s.met_ue_expiry)
                if old_ue != new_ue:
                    old_count = sum(bool(value and value >= today) for value in old_ue)
                    new_count = sum(bool(value and value >= today) for value in new_ue)
                    event_type = (
                        RosterImpactEventType.FIRST_UE_ACHIEVED
                        if old_count == 0 and new_count > 0
                        else RosterImpactEventType.UE_EXPIRED
                        if old_count > 0 and new_count == 0
                        else RosterImpactEventType.ADDITIONAL_UE_ACHIEVED
                    )
                    record_roster_impact(
                        event_type,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="Unit endorsement validity changed in roster profile.",
                    )
                if not old_profile["has_ojti"] and s.has_ojti:
                    record_roster_impact(
                        RosterImpactEventType.OJTI_ACHIEVED,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="OJTI qualification recorded.",
                    )
                if not old_profile["has_assessor"] and s.has_assessor:
                    record_roster_impact(
                        RosterImpactEventType.ASSESSOR_ACHIEVED,
                        today,
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="Assessor qualification recorded.",
                    )
                if old_profile["employment_type"] != s.employment_type:
                    record_roster_impact(
                        RosterImpactEventType.PART_TIME_CHANGE
                        if s.employment_type == "PART_TIME"
                        else RosterImpactEventType.FULL_TIME_CHANGE,
                        date.today(),
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="Employment type changed in roster profile.",
                    )
                elif (
                    old_profile["contracted_minutes_per_week"]
                    != s.contracted_minutes_per_week
                ):
                    record_roster_impact(
                        RosterImpactEventType.WORK_PATTERN_CHANGE,
                        date.today(),
                        staff_ids=[s.id],
                        rebuild_baseline=False,
                        reason="Contracted weekly hours changed in roster profile.",
                    )
                db.session.commit()
                flash("Staff updated.", "ok")
            except Exception as e:
                db.session.rollback()
                flash(f"Update failed: {e}", "error")

            return redirect(url_for("admin"))

        watches = Watch.query.order_by(Watch.order_index).all()
        account_membership = (
            UnitMembership.query.filter_by(unit_id=_current_unit_id(), person_id=s.id)
            .order_by(UnitMembership.id.desc())
            .first()
        )
        pending_access_invitation = (
            SecureInvitation.query.filter_by(
                unit_id=_current_unit_id(),
                target_person_id=s.id,
                accepted_at=None,
                disabled_at=None,
            )
            .order_by(SecureInvitation.id.desc())
            .first()
        )
        return render_template(
            "staff_edit.html",
            s=s,
            watches=watches,
            permissions=user_permissions(s),
            pattern_codes=PATTERN_CODES,
            account_membership=account_membership,
            pending_access_invitation=pending_access_invitation,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin/staff/<int:sid>",
            "admin_staff_edit",
            admin_staff_edit,
            methods=("GET", "POST"),
        )

    return blueprint
