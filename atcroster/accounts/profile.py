"""Self-service staff profile route."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

import pyotp
from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy.exc import ProgrammingError


@dataclass(frozen=True)
class StaffProfileDependencies:
    db: Any
    Staff: Any
    UnitMembership: Any
    PlatformIdentity: Any
    SmsSenderRegistration: Any
    MfaCredential: Any
    Assignment: Any
    Notification: Any
    current_unit_id: Callable[[], int]
    is_editor_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]
    normalise_uk_mobile: Callable[[str | None], str]
    valid_email: Callable[[str], str]
    normalise_phone: Callable[[str | None], str]
    now: Callable[[], Any]
    qr_data_uri: Callable[[str], str]
    absence_types: Callable[..., list[dict[str, Any]]]
    month_range: Callable[[int, int], tuple[date, list[date]]]
    get_shift: Callable[[str], Any]
    shift_duration_minutes: Callable[[Any], int]


def create_staff_profile_blueprint(dependencies: StaffProfileDependencies) -> Blueprint:
    blueprint = Blueprint("staff_profile", __name__)
    db = dependencies.db
    Staff = dependencies.Staff
    UnitMembership = dependencies.UnitMembership
    PlatformIdentity = dependencies.PlatformIdentity
    SmsSenderRegistration = dependencies.SmsSenderRegistration
    MfaCredential = dependencies.MfaCredential
    Assignment = dependencies.Assignment
    Notification = dependencies.Notification
    _current_unit_id = dependencies.current_unit_id
    is_editor_user = dependencies.is_editor_user
    _validate_csrf = dependencies.validate_csrf
    _normalise_uk_mobile = dependencies.normalise_uk_mobile
    _valid_email = dependencies.valid_email
    _normalise_phone_number = dependencies.normalise_phone
    utcnow = dependencies.now
    _totp_qr_data_uri = dependencies.qr_data_uri
    get_absence_types = dependencies.absence_types
    month_range = dependencies.month_range
    get_shift = dependencies.get_shift
    shift_duration_minutes = dependencies.shift_duration_minutes

    @login_required
    def staff_profile(sid):
        s = (
            Staff.query.filter_by(id=sid, unit_id=_current_unit_id())
            .filter(Staff.role != "position_monitor")
            .first_or_404()
        )
        if s.id != current_user.id and not is_editor_user(current_user):
            abort(403)
        if request.method == "POST":
            _validate_csrf()
            if s.id != current_user.id:
                abort(403)
            if request.form.get("form") == "sms_sender_registration":
                if not s.is_wm:
                    abort(403)
                number = _normalise_uk_mobile(request.form.get("sender_number"))
                if not number:
                    flash(
                        "Enter a UK mobile number, for example +447700900123.", "error"
                    )
                else:
                    try:
                        row = SmsSenderRegistration.query.filter_by(
                            unit_id=s.unit_id,
                            staff_id=s.id,
                            number=number,
                            provider="messagemedia",
                        ).first()
                    except ProgrammingError:
                        db.session.rollback()
                        flash(
                            "Personal sender registration is being enabled for this airport. Please try again shortly.",
                            "error",
                        )
                        return redirect(url_for("staff_profile", sid=s.id) + "#contact")
                    if not row:
                        row = SmsSenderRegistration(
                            unit_id=s.unit_id,
                            staff_id=s.id,
                            number=number,
                        )
                        db.session.add(row)
                    else:
                        row.status = "pending_dashboard_verification"
                        row.verification_requested_at = utcnow()
                        row.verified_at = None
                        row.expires_at = None
                    db.session.commit()
                    flash(
                        "Number recorded. Verify it in Sinch MessageMedia, then ask a Unit Administrator to confirm it here.",
                        "ok",
                    )
                return redirect(url_for("staff_profile", sid=s.id) + "#contact")
            email = _valid_email(request.form.get("email") or "")
            phone = _normalise_phone_number(request.form.get("phone_number"))
            if not email:
                flash("Enter a valid email address.", "error")
                return redirect(url_for("staff_profile", sid=s.id) + "#contact")
            if phone and not re.fullmatch(r"\+?[0-9]{7,15}", phone):
                flash(
                    "Enter a valid phone number with 7–15 digits. Include the "
                    "international country code for SMS messages.",
                    "error",
                )
                return redirect(url_for("staff_profile", sid=s.id) + "#contact")
            s.email = email
            s.phone_number = phone
            membership = UnitMembership.query.filter_by(
                unit_id=s.unit_id, person_id=s.id, status="active"
            ).first()
            if membership:
                identity = db.session.get(PlatformIdentity, membership.identity_id)
                if identity:
                    identity.email = email
            db.session.commit()
            flash("Contact details updated.", "ok")
            return redirect(url_for("staff_profile", sid=s.id) + "#contact")
        today = date.today()
        mfa_enabled = False
        mfa_secret = ""
        mfa_provisioning_uri = ""
        mfa_qr_data_uri = ""
        mfa_recovery_codes = []
        if s.id == current_user.id:
            credential = MfaCredential.query.filter_by(
                person_id=current_user.id
            ).first()
            mfa_enabled = bool(credential and credential.enabled)
            mfa_recovery_codes = session.pop("_new_mfa_recovery_codes", [])
            if not mfa_enabled:
                mfa_secret = session.get("_pending_mfa_secret")
                if not mfa_secret:
                    mfa_secret = pyotp.random_base32()
                    session["_pending_mfa_secret"] = mfa_secret
                mfa_provisioning_uri = pyotp.TOTP(mfa_secret).provisioning_uri(
                    name=current_user.username, issuer_name="ATCRoster"
                )
                mfa_qr_data_uri = _totp_qr_data_uri(mfa_provisioning_uri)

        # ensure_month_requirement(today.year, today.month)
        # generate_month(today.year, today.month)

        yr_ago = today - timedelta(days=365)

        al_days = sum(
            (lv.end - lv.start).days + 1
            for lv in s.leaves
            if lv.leave_type == "AL" and lv.end >= yr_ago and lv.start <= today
        )

        # Sickness categories configured for this airport, counted via assignments.
        sickness_codes = {
            item["code"] for item in get_absence_types("sickness", active_only=False)
        }
        q = Assignment.query.filter(
            Assignment.staff_id == s.id,
            Assignment.day >= yr_ago,
            Assignment.day <= today,
        )
        sick_days = sum(1 for a in q.all() if a.code in sickness_codes)

        month_start, days = month_range(today.year, today.month)
        month_end = days[-1]
        assigns = Assignment.query.filter(
            Assignment.staff_id == s.id,
            Assignment.day >= month_start,
            Assignment.day <= month_end,
        ).all()
        minutes = 0
        for a in assigns:
            sh = get_shift(a.code) if a and a.code else None
            if sh and sh.is_working:
                minutes += shift_duration_minutes(sh)
        hours_this_month = round(minutes / 60, 1)

        cal_link = None
        google_link = None
        apple_link = None
        if s.calendar_token:
            cal_link = url_for(
                "calendar_feed", sid=s.id, token=s.calendar_token, _external=True
            )
            # Apple uses webcal:// for subscription
            apple_link = cal_link.replace("http://", "webcal://").replace(
                "https://", "webcal://"
            )
            # Google "Add by URL" link
            from urllib.parse import quote

            google_link = (
                f"https://calendar.google.com/calendar/r?cid={quote(cal_link)}"
            )

        upcoming = (
            Assignment.query.filter(
                Assignment.staff_id == s.id,
                Assignment.day >= today,
                Assignment.day <= today + timedelta(days=45),
            )
            .order_by(Assignment.day.asc())
            .all()
        )
        next_duty = next(
            (a for a in upcoming if getattr(get_shift(a.code), "is_working", False)),
            None,
        )
        notifications = (
            Notification.query.filter_by(recipient_id=s.id)
            .order_by(Notification.created_at.desc())
            .limit(8)
            .all()
        )
        sms_sender_registrations = []
        if s.id == current_user.id and s.is_wm:
            try:
                sms_sender_registrations = (
                    SmsSenderRegistration.query.filter_by(
                        unit_id=s.unit_id, staff_id=s.id, provider="messagemedia"
                    )
                    .order_by(SmsSenderRegistration.id.desc())
                    .all()
                )
            except ProgrammingError:
                db.session.rollback()
        return render_template(
            "staff_profile.html",
            staff=s,
            al_days=al_days,
            sick_days=sick_days,
            hours_this_month=hours_this_month,
            cal_link=cal_link,
            apple_link=apple_link,
            google_link=google_link,
            upcoming=upcoming[:10],
            next_duty=next_duty,
            notifications=notifications,
            unread_notifications=sum(1 for item in notifications if not item.read_at),
            mfa_enabled=mfa_enabled,
            mfa_secret=mfa_secret,
            mfa_provisioning_uri=mfa_provisioning_uri,
            mfa_qr_data_uri=mfa_qr_data_uri,
            mfa_recovery_codes=mfa_recovery_codes,
            sms_sender_registrations=sms_sender_registrations,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/staff/<int:sid>", "staff_profile", staff_profile, methods=("GET", "POST")
        )

    return blueprint
