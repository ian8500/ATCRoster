"""Qualification administration routes and transactional workflow."""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
from dataclasses import dataclass
from datetime import date
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
class QualificationDependencies:
    db: Any
    Staff: Any
    QualificationType: Any
    PersonQualification: Any
    PersonQualificationHistory: Any
    RosterImpactEventType: Any
    current_unit_id: Callable[[], int]
    is_editor_user: Callable[[Any], bool]
    is_admin_user: Callable[[Any], bool]
    now: Callable[[], Any]
    qualification_impact_type: Callable[..., tuple[Any, date]]
    person_has_other_valid_ue: Callable[..., bool]
    record_roster_impact: Callable[..., None]


def create_qualification_blueprint(
    dependencies: QualificationDependencies,
) -> Blueprint:
    """Create the /compliance route with its qualification-owned workflow."""
    blueprint = Blueprint("qualifications", __name__)
    db = dependencies.db
    Staff = dependencies.Staff
    QualificationType = dependencies.QualificationType
    PersonQualification = dependencies.PersonQualification
    PersonQualificationHistory = dependencies.PersonQualificationHistory
    RosterImpactEventType = dependencies.RosterImpactEventType
    _current_unit_id = dependencies.current_unit_id
    is_editor_user = dependencies.is_editor_user
    is_admin_user = dependencies.is_admin_user
    utcnow = dependencies.now
    _qualification_impact_type = dependencies.qualification_impact_type
    _person_has_other_valid_ue = dependencies.person_has_other_valid_ue
    record_roster_impact = dependencies.record_roster_impact

    def _qualification_snapshot(record: PersonQualification) -> dict:
        return {
            "person_id": record.person_id,
            "qualification_type_id": record.qualification_type_id,
            "issued_on": record.issued_on,
            "valid_from": record.valid_from,
            "expires_on": record.expires_on,
            "status": record.status,
        }

    def _record_qualification_history(record: PersonQualification, action: str) -> None:
        db.session.add(
            PersonQualificationHistory(
                unit_id=record.unit_id,
                person_qualification_id=record.id,
                actor_id=current_user.id,
                action=action,
                snapshot_json=json.dumps(
                    _qualification_snapshot(record), default=str, sort_keys=True
                ),
            )
        )

    def _sync_qualification_to_roster_profile(
        person: Staff, qtype: QualificationType, expires_on: date | None
    ) -> None:
        legacy_field = {
            "MEDICAL": "medical_expiry",
            "ADI": "tower_ue_expiry",
            "APS": "radar_ue_expiry",
            "MET": "met_ue_expiry",
        }.get(qtype.code)
        if legacy_field:
            setattr(person, legacy_field, expires_on)

    @login_required
    def qualification_compliance():
        if not is_editor_user(current_user):
            abort(403)
        unit_id = _current_unit_id()
        import_preview = None
        if request.method == "POST":
            if not is_admin_user(current_user):
                abort(403)
            action = (request.form.get("action") or "").strip()
            if action in {"create_type", "edit_type"}:
                code = (request.form.get("code") or "").strip().upper()
                label = (request.form.get("label") or "").strip()
                warning_csv = (request.form.get("warning_days_csv") or "").strip()
                if not re.fullmatch(r"[A-Z0-9_ -]{2,30}", code) or not label:
                    abort(400, "Enter a valid qualification code and label.")
                try:
                    warnings = sorted(
                        {
                            int(value.strip())
                            for value in warning_csv.split(",")
                            if value.strip()
                        },
                        reverse=True,
                    )
                except ValueError:
                    abort(400, "Warning periods must be comma-separated days.")
                if not warnings or any(value < 0 or value > 3650 for value in warnings):
                    abort(
                        400, "Configure at least one warning period from 0–3650 days."
                    )
                if action == "create_type":
                    if QualificationType.query.filter_by(
                        unit_id=unit_id, code=code
                    ).first():
                        abort(409, "That qualification code already exists.")
                    qtype = QualificationType(unit_id=unit_id, code=code)
                    db.session.add(qtype)
                else:
                    qtype = QualificationType.query.filter_by(
                        id=int(request.form.get("type_id") or 0),
                        unit_id=unit_id,
                    ).first_or_404()
                    if (
                        qtype.code != code
                        and PersonQualification.query.filter_by(
                            unit_id=unit_id, qualification_type_id=qtype.id
                        ).first()
                    ):
                        abort(409, "A used qualification code cannot be changed.")
                    qtype.code = code
                qtype.label = label[:100]
                qtype.warning_days_csv = ",".join(str(value) for value in warnings)
                qtype.expiry_required = request.form.get("expiry_required") == "yes"
                qtype.is_active = request.form.get("is_active") == "yes"
                db.session.commit()
                flash("Qualification type saved.", "ok")
                return redirect(url_for("qualification_compliance"))
            if action == "save_person":
                person = Staff.query.filter_by(
                    id=int(request.form.get("person_id") or 0),
                    unit_id=unit_id,
                    is_operational=True,
                ).first_or_404()
                qtype = QualificationType.query.filter_by(
                    id=int(request.form.get("type_id") or 0),
                    unit_id=unit_id,
                    is_active=True,
                ).first_or_404()
                status = (request.form.get("status") or "valid").strip()
                if status not in {"valid", "suspended", "revoked", "inactive"}:
                    abort(400, "Invalid qualification status.")

                def optional_date(name):
                    raw = (request.form.get(name) or "").strip()
                    return date.fromisoformat(raw) if raw else None

                try:
                    issued_on = optional_date("issued_on")
                    valid_from = optional_date("valid_from")
                    expires_on = optional_date("expires_on")
                    valid_to = optional_date("valid_to")
                    suspended_from = optional_date("suspended_from")
                    suspended_to = optional_date("suspended_to")
                except ValueError:
                    abort(400, "Qualification dates must be valid ISO dates.")
                if qtype.expiry_required and status == "valid" and not expires_on:
                    abort(400, "This qualification requires an expiry date.")
                record = PersonQualification.query.filter_by(
                    unit_id=unit_id,
                    person_id=person.id,
                    qualification_type_id=qtype.id,
                ).first()
                old_state = (
                    record.status if record else None,
                    record.valid_from if record else None,
                    record.expires_on if record else None,
                )
                action_name = "renewed" if record else "assigned"
                if not record:
                    record = PersonQualification(
                        unit_id=unit_id,
                        person_id=person.id,
                        qualification_type_id=qtype.id,
                    )
                    db.session.add(record)
                    db.session.flush()
                record.issued_on = issued_on
                record.valid_from = valid_from
                record.expires_on = expires_on
                record.valid_to = valid_to
                record.suspended_from = suspended_from
                record.suspended_to = suspended_to
                record.evidence_reference = (
                    request.form.get("evidence_reference") or ""
                ).strip()[:500]
                record.created_by_user_id = (
                    record.created_by_user_id
                    or getattr(current_user, "person_id", None)
                    or getattr(current_user, "id", None)
                )
                record.status = status
                record.updated_at = utcnow()
                _sync_qualification_to_roster_profile(person, qtype, expires_on)
                _record_qualification_history(record, action_name)
                impact_type, impact_date = _qualification_impact_type(
                    qtype.code,
                    *old_state,
                    record.status,
                    record.valid_from,
                    record.expires_on,
                )
                if (
                    impact_type == RosterImpactEventType.FIRST_UE_ACHIEVED
                    and _person_has_other_valid_ue(
                        unit_id, person.id, qtype.id, impact_date
                    )
                ):
                    impact_type = RosterImpactEventType.ADDITIONAL_UE_ACHIEVED
                if impact_type:
                    record_roster_impact(
                        impact_type,
                        impact_date,
                        staff_ids=[person.id],
                        rebuild_baseline=False,
                        reason=f"{qtype.code} qualification {action_name}.",
                    )
                db.session.commit()
                flash("Person qualification saved.", "ok")
                return redirect(url_for("qualification_compliance"))
            if action == "import_preview":
                upload = request.files.get("csv_file")
                if not upload:
                    abort(400, "Choose a qualification CSV file.")
                try:
                    reader = csv.DictReader(
                        io.StringIO(upload.read().decode("utf-8-sig"))
                    )
                except UnicodeDecodeError:
                    abort(400, "CSV must use UTF-8 encoding.")
                required = {"staff_no", "type_code", "status"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    abort(400, "CSV requires staff_no,type_code,status.")
                rows, errors = [], []
                for line, raw in enumerate(reader, start=2):
                    person = Staff.query.filter_by(
                        unit_id=unit_id,
                        staff_no=(raw.get("staff_no") or "").strip(),
                        is_operational=True,
                    ).first()
                    qtype = QualificationType.query.filter_by(
                        unit_id=unit_id,
                        code=(raw.get("type_code") or "").strip().upper(),
                        is_active=True,
                    ).first()
                    status = (raw.get("status") or "").strip()
                    try:
                        parsed = {
                            key: (
                                date.fromisoformat((raw.get(key) or "").strip())
                                if (raw.get(key) or "").strip()
                                else None
                            )
                            for key in ("issued_on", "valid_from", "expires_on")
                        }
                    except ValueError:
                        errors.append(f"Line {line}: invalid date.")
                        continue
                    if (
                        not person
                        or not qtype
                        or status not in {"valid", "suspended", "revoked", "inactive"}
                    ):
                        errors.append(f"Line {line}: unknown person/type/status.")
                        continue
                    if (
                        qtype.expiry_required
                        and status == "valid"
                        and not parsed["expires_on"]
                    ):
                        errors.append(f"Line {line}: expiry is required.")
                        continue
                    rows.append(
                        {
                            "person_id": person.id,
                            "person": person.name,
                            "type_id": qtype.id,
                            "type": qtype.code,
                            "status": status,
                            **{
                                key: value.isoformat() if value else ""
                                for key, value in parsed.items()
                            },
                        }
                    )
                nonce = secrets.token_urlsafe(18)
                if not errors:
                    session["_qualification_import_preview"] = {
                        "unit_id": unit_id,
                        "nonce": nonce,
                        "rows": rows,
                    }
                import_preview = {"rows": rows, "errors": errors, "nonce": nonce}
            elif action == "import_apply":
                saved = session.get("_qualification_import_preview") or {}
                if saved.get("unit_id") != unit_id or not secrets.compare_digest(
                    request.form.get("nonce") or "",
                    saved.get("nonce") or "",
                ):
                    abort(409, "The qualification preview has expired.")
                for row in saved.get("rows") or []:
                    record = PersonQualification.query.filter_by(
                        unit_id=unit_id,
                        person_id=row["person_id"],
                        qualification_type_id=row["type_id"],
                    ).first()
                    old_state = (
                        record.status if record else None,
                        record.valid_from if record else None,
                        record.expires_on if record else None,
                    )
                    if not record:
                        record = PersonQualification(
                            unit_id=unit_id,
                            person_id=row["person_id"],
                            qualification_type_id=row["type_id"],
                        )
                        db.session.add(record)
                        db.session.flush()
                    for key in ("issued_on", "valid_from", "expires_on"):
                        setattr(
                            record,
                            key,
                            date.fromisoformat(row[key]) if row[key] else None,
                        )
                    record.status = row["status"]
                    record.updated_at = utcnow()
                    person = Staff.query.filter_by(
                        unit_id=unit_id, id=row["person_id"]
                    ).first_or_404()
                    qtype = QualificationType.query.filter_by(
                        unit_id=unit_id, id=row["type_id"]
                    ).first_or_404()
                    _sync_qualification_to_roster_profile(
                        person, qtype, record.expires_on
                    )
                    _record_qualification_history(record, "imported")
                    impact_type, impact_date = _qualification_impact_type(
                        qtype.code,
                        *old_state,
                        record.status,
                        record.valid_from,
                        record.expires_on,
                    )
                    if (
                        impact_type == RosterImpactEventType.FIRST_UE_ACHIEVED
                        and _person_has_other_valid_ue(
                            unit_id, person.id, qtype.id, impact_date
                        )
                    ):
                        impact_type = RosterImpactEventType.ADDITIONAL_UE_ACHIEVED
                    if impact_type:
                        record_roster_impact(
                            impact_type,
                            impact_date,
                            staff_ids=[person.id],
                            rebuild_baseline=False,
                            reason=f"{qtype.code} qualification imported.",
                        )
                db.session.commit()
                session.pop("_qualification_import_preview", None)
                flash("Qualification import applied.", "ok")
                return redirect(url_for("qualification_compliance"))
            else:
                abort(400, "Unknown qualification action.")
        today = date.today()
        qualification_types = (
            QualificationType.query.filter_by(unit_id=unit_id)
            .order_by(QualificationType.code)
            .all()
        )
        people = (
            Staff.query.filter_by(unit_id=unit_id, is_operational=True)
            .filter(Staff.role != "position_monitor")
            .order_by(Staff.name)
            .all()
        )
        qualifications = PersonQualification.query.filter_by(unit_id=unit_id).all()
        by_person_type = {
            (row.person_id, row.qualification_type_id): row for row in qualifications
        }
        rows = []
        for person in people:
            for qtype in qualification_types:
                qual = by_person_type.get((person.id, qtype.id))
                expires_on = qual.expires_on if qual else None
                days = None if not expires_on else (expires_on - today).days
                try:
                    warning_days = max(
                        int(value.strip())
                        for value in (qtype.warning_days_csv or "180").split(",")
                        if value.strip()
                    )
                except (TypeError, ValueError):
                    warning_days = 180
                if not qual:
                    state = "missing"
                elif qual.status != "valid":
                    state = qual.status
                elif qual.valid_from and qual.valid_from > today:
                    state = "not-yet-valid"
                elif qtype.expiry_required and not expires_on:
                    state = "missing"
                elif expires_on and days < 0:
                    state = "expired"
                elif expires_on and days <= warning_days:
                    state = "expiring"
                else:
                    state = "valid"
                rows.append(
                    {
                        "person": person,
                        "type": qtype,
                        "qualification": qual,
                        "expires_on": expires_on,
                        "days": days,
                        "state": state,
                    }
                )
        history = (
            PersonQualificationHistory.query.filter_by(unit_id=unit_id)
            .order_by(PersonQualificationHistory.occurred_at.desc())
            .limit(100)
            .all()
        )
        return render_template(
            "qualification_compliance.html",
            rows=rows,
            qualification_types=qualification_types,
            people=people,
            history=history,
            import_preview=import_preview,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/compliance",
            "qualification_compliance",
            qualification_compliance,
            methods=("GET", "POST"),
        )

    return blueprint
