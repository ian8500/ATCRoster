"""Guided unit onboarding route."""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class OnboardingDependencies:
    db: Any
    Unit: Any
    QualificationType: Any
    Watch: Any
    Staff: Any
    ShiftType: Any
    UnitMembership: Any
    SecureInvitation: Any
    Requirement: Any
    current_unit_id: Callable[[], int]
    is_admin_user: Callable[[Any], bool]
    validate_csrf: Callable[[], None]


def create_onboarding_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> OnboardingDependencies:
    """Bind onboarding's persisted models at its composition boundary."""
    return OnboardingDependencies(
        db=db,
        Unit=operational_models.Unit,
        QualificationType=saas_models.QualificationType,
        Watch=operational_models.Watch,
        Staff=operational_models.Staff,
        ShiftType=operational_models.ShiftType,
        UnitMembership=saas_models.UnitMembership,
        SecureInvitation=saas_models.SecureInvitation,
        Requirement=operational_models.Requirement,
        **services,
    )


def create_onboarding_blueprint(dependencies: OnboardingDependencies) -> Blueprint:
    blueprint = Blueprint("unit_onboarding", __name__)
    db = dependencies.db
    Unit = dependencies.Unit
    QualificationType = dependencies.QualificationType
    Watch = dependencies.Watch
    Staff = dependencies.Staff
    ShiftType = dependencies.ShiftType
    UnitMembership = dependencies.UnitMembership
    SecureInvitation = dependencies.SecureInvitation
    Requirement = dependencies.Requirement
    _current_unit_id = dependencies.current_unit_id
    is_admin_user = dependencies.is_admin_user
    _validate_csrf = dependencies.validate_csrf

    @login_required
    def unit_onboarding():
        if not is_admin_user(current_user):
            abort(403)
        unit = db.session.get(Unit, _current_unit_id())
        if not unit:
            abort(404)
        csv_preview = None
        if request.method == "POST":
            _validate_csrf()
            action = (request.form.get("action") or "identity").strip()
            if action == "complete_setup":
                if request.form.get("confirm_complete") != "yes":
                    flash(
                        "Confirm that you are ready to leave guided setup.",
                        "error",
                    )
                    return redirect(url_for("unit_onboarding"))
                unit.onboarding_step = 100
                db.session.commit()
                flash(
                    "Airport setup marked complete. Welcome to your operational dashboard.",
                    "ok",
                )
                return redirect(url_for("index"))
            if action == "identity":
                unit.name = (request.form.get("name") or unit.name).strip()[:120]
                code = (request.form.get("code") or unit.code).strip().upper()
                if not re.fullmatch(r"[A-Z0-9]{2,12}", code):
                    abort(400, "Invalid airport code.")
                duplicate = Unit.query.filter(
                    Unit.code == code, Unit.id != unit.id
                ).first()
                if duplicate:
                    abort(409, "That airport code is already used.")
                unit.code = code
                unit.timezone = (request.form.get("timezone") or unit.timezone).strip()[
                    :64
                ]
                unit.locale = (request.form.get("locale") or unit.locale).strip()[:20]
                unit.date_format = (
                    request.form.get("date_format") or unit.date_format
                ).strip()[:30]
                primary = request.form.get("primary_colour") or "#16283a"
                accent = request.form.get("accent_colour") or "#2c7be5"
                if not re.fullmatch(r"#[0-9A-Fa-f]{6}", primary) or not re.fullmatch(
                    r"#[0-9A-Fa-f]{6}", accent
                ):
                    abort(400, "Brand colours must be six-digit hex values.")
                unit.branding_json = json.dumps(
                    {
                        "primary_colour": primary,
                        "accent_colour": accent,
                        "display_name": (
                            request.form.get("display_name") or unit.name
                        ).strip()[:120],
                    },
                    sort_keys=True,
                )
                unit.onboarding_step = max(unit.onboarding_step, 2)
                db.session.commit()
                flash("Airport identity and branding saved.", "ok")
                return redirect(url_for("unit_onboarding"))
            if action == "request_rules":
                try:
                    months = int(request.form.get("request_months_ahead") or 3)
                    lock_day = int(request.form.get("request_lock_day") or 20)
                    protected_months = int(
                        request.form.get("protected_roster_months_ahead")
                        or unit.protected_roster_months_ahead
                    )
                except ValueError:
                    abort(400, "Request rules must be whole numbers.")
                if (
                    not 1 <= months <= 24
                    or not 1 <= lock_day <= 28
                    or not 0 <= protected_months <= 24
                ):
                    abort(
                        400,
                        "Request window must be 1–24 months, lock day 1–28, "
                        "and protected horizon 0–24 months.",
                    )
                unit.request_months_ahead = months
                unit.request_lock_day = lock_day
                unit.protected_roster_months_ahead = protected_months
                unit.onboarding_step = max(unit.onboarding_step, 8)
                db.session.commit()
                flash("Fatigue and request rules saved.", "ok")
                return redirect(url_for("unit_onboarding"))
            if action == "seed_qualifications":
                defaults = (
                    "MEDICAL",
                    "ADI",
                    "APP",
                    "APS",
                    "OJTI",
                    "UCA",
                    "ENGLISH_LANGUAGE",
                )
                for code in defaults:
                    if not QualificationType.query.filter_by(
                        unit_id=unit.id, code=code
                    ).first():
                        db.session.add(
                            QualificationType(
                                unit_id=unit.id,
                                code=code,
                                label=code.replace("_", " ").title(),
                                warning_days_csv="180,90,60,30",
                            )
                        )
                unit.onboarding_step = max(unit.onboarding_step, 5)
                db.session.commit()
                flash("Default qualification types added.", "ok")
                return redirect(url_for("unit_onboarding"))
            if action == "csv_preview":
                upload = request.files.get("csv_file")
                if not upload or not upload.filename:
                    abort(400, "Choose a CSV file.")
                try:
                    content = upload.read().decode("utf-8-sig")
                except UnicodeDecodeError:
                    abort(400, "CSV must use UTF-8 encoding.")
                reader = csv.DictReader(io.StringIO(content))
                required = {"name", "staff_no", "watch"}
                if not reader.fieldnames or not required.issubset(
                    {field.strip() for field in reader.fieldnames}
                ):
                    abort(400, "CSV requires name, staff_no and watch columns.")
                watches = {
                    row.name.strip().lower(): row
                    for row in Watch.query.filter_by(unit_id=unit.id).all()
                }
                seen_numbers = set()
                rows, errors = [], []
                for line_number, raw in enumerate(reader, start=2):
                    if len(rows) + len(errors) >= 500:
                        errors.append("CSV is limited to 500 records.")
                        break
                    name = (raw.get("name") or "").strip()
                    staff_no = (raw.get("staff_no") or "").strip()
                    watch_name = (raw.get("watch") or "").strip()
                    watch = watches.get(watch_name.lower())
                    line_errors = []
                    if not name:
                        line_errors.append("name is required")
                    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,20}", staff_no):
                        line_errors.append("invalid staff_no")
                    if (
                        staff_no.lower() in seen_numbers
                        or Staff.query.filter_by(
                            unit_id=unit.id, staff_no=staff_no
                        ).first()
                    ):
                        line_errors.append("duplicate staff_no")
                    if not watch:
                        line_errors.append("unknown watch")
                    if line_errors:
                        errors.append(f"Line {line_number}: {', '.join(line_errors)}")
                        continue
                    seen_numbers.add(staff_no.lower())
                    rows.append(
                        {
                            "name": name[:80],
                            "staff_no": staff_no,
                            "watch_id": watch.id,
                            "watch": watch.name,
                        }
                    )
                nonce = secrets.token_urlsafe(18)
                if not errors:
                    session["_onboarding_csv_preview"] = {
                        "unit_id": unit.id,
                        "nonce": nonce,
                        "rows": rows,
                    }
                else:
                    session.pop("_onboarding_csv_preview", None)
                csv_preview = {
                    "rows": rows,
                    "errors": errors,
                    "nonce": nonce,
                }
            elif action == "csv_apply":
                saved = session.get("_onboarding_csv_preview") or {}
                nonce = request.form.get("nonce") or ""
                if saved.get("unit_id") != unit.id or not secrets.compare_digest(
                    nonce, str(saved.get("nonce") or "")
                ):
                    abort(409, "The import preview has expired.")
                for row in saved.get("rows") or []:
                    person = Staff(
                        unit_id=unit.id,
                        username=(f"person-{unit.code.lower()}-{secrets.token_hex(8)}"),
                        name=row["name"],
                        staff_no=row["staff_no"],
                        watch_id=row["watch_id"],
                        role="user",
                        membership_status="no_login",
                        is_operational=True,
                    )
                    person.set_password(secrets.token_urlsafe(32))
                    db.session.add(person)
                unit.onboarding_step = max(unit.onboarding_step, 9)
                db.session.commit()
                session.pop("_onboarding_csv_preview", None)
                flash("Validated staff records imported.", "ok")
                return redirect(url_for("unit_onboarding"))
            else:
                abort(400, "Unknown onboarding action.")
        active = UnitMembership.query.filter_by(
            unit_id=unit.id, status="active"
        ).count()
        pending = SecureInvitation.query.filter_by(
            unit_id=unit.id, accepted_at=None, disabled_at=None
        ).count()
        readiness = [
            (
                "Airport identity",
                bool(unit.name and unit.code and unit.timezone),
                "unit_onboarding",
            ),
            ("Watches configured", Watch.query.count() > 0, "admin"),
            (
                "Active shifts configured",
                ShiftType.query.filter_by(is_active=True).count() > 0,
                "admin",
            ),
            (
                "Operational staff added",
                Staff.query.filter_by(is_operational=True).count() > 0,
                "admin",
            ),
            ("Staffing requirements set", Requirement.query.count() > 0, "admin"),
            (
                "Qualification types set",
                QualificationType.query.count() > 0,
                "qualification_compliance",
            ),
            ("Roster warning rules available", True, "admin_fatigue_rules"),
            ("Unit Admin access active", active > 0, "unit_accounts"),
        ]
        readiness_complete = sum(1 for _, complete, _ in readiness if complete)
        return render_template(
            "unit_onboarding.html",
            unit=unit,
            active_accounts=active,
            pending_invitations=pending,
            readiness=readiness,
            readiness_complete=readiness_complete,
            readiness_percent=round(readiness_complete / len(readiness) * 100),
            csv_preview=csv_preview,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/unit/onboarding",
            "unit_onboarding",
            unit_onboarding,
            methods=("GET", "POST"),
        )

    return blueprint
