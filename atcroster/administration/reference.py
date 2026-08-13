"""Reference-data administration for roster settings and annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from werkzeug.exceptions import HTTPException


@dataclass(frozen=True)
class ReferenceDataDependencies:
    db: Any
    AnnotationType: Any
    AnnotationAudit: Any
    Assignment: Any
    ShiftType: Any
    current_unit_id: Callable[[], int]
    validate_csrf: Callable[[], None]
    refresh_annotation_cache: Callable[[], None]
    normalise_codes: Callable[[list[str]], list[str]]
    save_codes_setting: Callable[[str, list[str]], None]
    prune_roster_code_settings: Callable[[int], int]
    working_codes: Callable[[], set[str]]
    banned_codes: Callable[[], set[str]]
    excluded_codes: Callable[[], set[str]]
    non_working_codes: Callable[[], set[str]]
    admin_required: Callable[[Callable[..., Any]], Callable[..., Any]]


def create_reference_data_blueprint(
    dependencies: ReferenceDataDependencies,
) -> Blueprint:
    blueprint = Blueprint("reference_data", __name__)
    db = dependencies.db
    AnnotationType = dependencies.AnnotationType
    AnnotationAudit = dependencies.AnnotationAudit
    Assignment = dependencies.Assignment
    ShiftType = dependencies.ShiftType
    _current_unit_id = dependencies.current_unit_id
    _validate_csrf = dependencies.validate_csrf
    refresh_annotation_cache = dependencies.refresh_annotation_cache
    _normalise_codes = dependencies.normalise_codes
    _save_codes_setting = dependencies.save_codes_setting
    _prune_roster_code_settings = dependencies.prune_roster_code_settings
    get_working_codes = dependencies.working_codes
    get_banned_roster_codes = dependencies.banned_codes
    get_exclude_from_counters = dependencies.excluded_codes
    get_non_working_codes = dependencies.non_working_codes

    @login_required
    @dependencies.admin_required
    def admin_reference():
        unit_id = _current_unit_id()
        if not unit_id:
            abort(403)
        settings_meta = {
            "working_codes": {
                "label": "Working shift codes",
                "help": "Codes treated as working when checking fatigue and consecutive days.",
            },
            "banned_codes": {
                "label": "Roster grid exclusions",
                "help": "Codes that cannot be set directly from the roster grid (must use dedicated forms).",
            },
            "exclude_from_counters": {
                "label": "Daily counter exclusions",
                "help": "Codes ignored when calculating the M/D/A/N requirement counters.",
            },
            "non_working_codes": {
                "label": "Non-working codes",
                "help": "Codes that always count as non-working when evaluating fatigue rules.",
            },
        }

        if request.method == "POST":
            _validate_csrf()
            form = request.form.get("form", "")
            try:
                if form == "annotation_new":
                    code = (request.form.get("code") or "").strip().upper()
                    if not re.fullmatch(r"[A-Z0-9]{1,10}", code):
                        flash(
                            "Annotation code must be 1–10 letters or numbers.",
                            "error",
                        )
                        return redirect(url_for("admin_reference"))
                    if AnnotationType.query.filter_by(
                        unit_id=unit_id, code=code
                    ).first():
                        flash("That annotation code already exists.", "error")
                        return redirect(url_for("admin_reference"))
                    label = (request.form.get("label") or code).strip()
                    category = (request.form.get("category") or "Other").strip()
                    try:
                        toil_half_days = int(request.form.get("toil_half_days") or 0)
                    except ValueError:
                        toil_half_days = 0
                    toil_half_days = max(-200, min(toil_half_days, 200))
                    is_active = bool(request.form.get("is_active", True))

                    ann = AnnotationType(
                        unit_id=unit_id,
                        code=code,
                        label=label or code,
                        category=category or "Other",
                        colour=(
                            request.form.get("colour")
                            if re.fullmatch(
                                r"#[0-9A-Fa-f]{6}",
                                request.form.get("colour") or "",
                            )
                            else "#6c757d"
                        ),
                        description=(request.form.get("description") or "")[:1000],
                        allow_suffix=False,
                        suffixes="",
                        toil_half_days=toil_half_days,
                        tags="",
                        note_required=bool(request.form.get("note_required")),
                        admin_only=bool(request.form.get("admin_only")),
                        is_active=is_active,
                        sort_order=100,
                    )
                    db.session.add(ann)
                    db.session.flush()
                    db.session.add(
                        AnnotationAudit(
                            unit_id=unit_id,
                            annotation_type_id=ann.id,
                            actor_id=current_user.id,
                            action="definition_created",
                            new_value=json.dumps({"code": code, "label": label}),
                        )
                    )
                    db.session.commit()
                    refresh_annotation_cache()
                    flash("Annotation added.", "ok")
                    return redirect(url_for("admin_reference"))

                if form == "annotation_edit":
                    try:
                        aid = int(request.form.get("annotation_id") or "")
                    except ValueError:
                        abort(400, "Invalid annotation ID.")
                    ann = AnnotationType.query.filter_by(
                        id=aid, unit_id=unit_id
                    ).first_or_404()
                    new_code = (request.form.get("code") or ann.code).strip().upper()
                    if not re.fullmatch(r"[A-Z0-9]{1,10}", new_code):
                        abort(400, "Invalid annotation code.")
                    if ann.has_been_used and new_code != ann.code:
                        abort(
                            409,
                            "A used annotation code is immutable; deactivate it and create a new definition.",
                        )
                    duplicate = AnnotationType.query.filter(
                        AnnotationType.unit_id == unit_id,
                        AnnotationType.code == new_code,
                        AnnotationType.id != ann.id,
                    ).first()
                    if duplicate:
                        abort(409, "That annotation code already exists.")
                    old_value = {
                        "code": ann.code,
                        "label": ann.label,
                        "category": ann.category,
                        "active": ann.is_active,
                    }
                    ann.code = new_code or ann.code
                    ann.label = (
                        request.form.get("label") or ann.label or new_code
                    ).strip() or new_code
                    ann.category = (
                        request.form.get("category") or ann.category or "Other"
                    ).strip() or "Other"
                    requested_colour = request.form.get("colour") or ann.colour
                    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", requested_colour or ""):
                        abort(400, "Invalid annotation colour.")
                    ann.colour = requested_colour
                    ann.description = (
                        request.form.get("description") or ann.description or ""
                    )[:1000]
                    try:
                        ann.toil_half_days = int(
                            request.form.get("toil_half_days") or 0
                        )
                    except ValueError:
                        ann.toil_half_days = 0
                    ann.toil_half_days = max(-200, min(ann.toil_half_days, 200))
                    ann.note_required = bool(request.form.get("note_required"))
                    ann.admin_only = bool(request.form.get("admin_only"))
                    ann.is_active = bool(request.form.get("is_active"))
                    db.session.add(
                        AnnotationAudit(
                            unit_id=unit_id,
                            annotation_type_id=ann.id,
                            actor_id=current_user.id,
                            action="definition_updated",
                            old_value=json.dumps(old_value, sort_keys=True),
                            new_value=json.dumps(
                                {
                                    "code": ann.code,
                                    "label": ann.label,
                                    "category": ann.category,
                                    "active": ann.is_active,
                                },
                                sort_keys=True,
                            ),
                        )
                    )
                    db.session.commit()
                    refresh_annotation_cache()
                    flash("Annotation updated.", "ok")
                    return redirect(url_for("admin_reference"))

                if form == "annotation_delete":
                    try:
                        aid = int(request.form.get("annotation_id") or "")
                    except ValueError:
                        abort(400, "Invalid annotation ID.")
                    ann = AnnotationType.query.filter_by(
                        id=aid, unit_id=unit_id
                    ).first_or_404()
                    used = (
                        Assignment.query.filter(
                            Assignment.unit_id == unit_id,
                            Assignment.annotation.like(f"{ann.code}%"),
                        ).first()
                        is not None
                    )
                    ann.has_been_used = ann.has_been_used or used
                    ann.is_active = False
                    db.session.add(
                        AnnotationAudit(
                            unit_id=unit_id,
                            annotation_type_id=ann.id,
                            actor_id=current_user.id,
                            action="definition_deactivated",
                            old_value="active",
                            new_value="inactive",
                        )
                    )
                    db.session.commit()
                    refresh_annotation_cache()
                    flash(
                        "Annotation deactivated; historical use remains readable.", "ok"
                    )
                    return redirect(url_for("admin_reference"))

                if form == "settings_codes":
                    key = request.form.get("key", "")
                    if key not in settings_meta:
                        flash("Unknown setting.", "error")
                        return redirect(url_for("admin_reference"))
                    values = _normalise_codes(request.form.getlist("values"))
                    valid_codes = {
                        str(code or "").strip().upper()
                        for (code,) in db.session.query(ShiftType.code)
                        .filter_by(unit_id=unit_id)
                        .all()
                    }
                    unknown = sorted(set(values) - valid_codes)
                    if unknown:
                        flash(
                            "The list was not saved because these roster codes "
                            f"do not exist: {', '.join(unknown)}.",
                            "error",
                        )
                        return redirect(url_for("admin_reference"))
                    _save_codes_setting(key, values)
                    flash("Reference list updated.", "ok")
                    return redirect(url_for("admin_reference"))

                flash("Unknown action.", "error")
                return redirect(url_for("admin_reference"))
            except HTTPException:
                db.session.rollback()
                raise
            except Exception as exc:
                db.session.rollback()
                flash(f"Update failed: {exc}", "error")
                return redirect(url_for("admin_reference"))

        if _prune_roster_code_settings(unit_id):
            db.session.commit()

        annotations = (
            AnnotationType.query.filter_by(unit_id=unit_id)
            .order_by(AnnotationType.code)
            .all()
        )

        settings_view = []
        roster_codes = (
            ShiftType.query.filter_by(unit_id=unit_id).order_by(ShiftType.code).all()
        )
        current_values = {
            "working_codes": sorted(get_working_codes()),
            "banned_codes": sorted(get_banned_roster_codes()),
            "exclude_from_counters": sorted(get_exclude_from_counters()),
            "non_working_codes": sorted(get_non_working_codes()),
        }
        for key, meta in settings_meta.items():
            settings_view.append(
                {
                    "key": key,
                    "label": meta["label"],
                    "help": meta["help"],
                    "selected_codes": set(current_values.get(key, [])),
                }
            )

        return render_template(
            "admin_reference.html",
            annotations=annotations,
            settings=settings_view,
            roster_codes=roster_codes,
        )

    @blueprint.record_once
    def register_legacy_endpoint(state) -> None:
        state.app.add_url_rule(
            "/admin/reference",
            "admin_reference",
            admin_reference,
            methods=("GET", "POST"),
        )

    return blueprint
