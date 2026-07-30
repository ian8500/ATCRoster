"""Optional airport briefing module.

The module is deliberately isolated behind the ``briefing_module`` feature
flag.  Its tables can be downgraded without changing roster records.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import io
import json
import re
import secrets
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import (
    Blueprint, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app import db, utcnow
from briefing_storage import (
    BriefingStorageError, configured_briefing_storage,
)


briefing_blueprint = Blueprint("briefing", __name__, url_prefix="/briefing")

ALLOWED_DOCUMENTS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_ACTIVE_VIEW_SECONDS_PER_HEARTBEAT = 30


class BriefingMessageType(db.Model):
    __tablename__ = "briefing_message_type"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, nullable=False, index=True)
    name = db.Column(db.String(80), nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    display_order = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    __table_args__ = (
        db.UniqueConstraint(
            "unit_id", "name", name="uq_briefing_message_type_name",
        ),
    )


class BriefingItem(db.Model):
    __tablename__ = "briefing_item"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, nullable=False, index=True)
    kind = db.Column(db.String(20), nullable=False)  # instruction | daily | notam
    title = db.Column(db.String(160), nullable=False)
    message_type_id = db.Column(db.Integer, index=True)
    message_type_name = db.Column(db.String(80), nullable=False, default="")
    body = db.Column(db.Text, nullable=False, default="")
    effective_at = db.Column(db.DateTime, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    mandatory = db.Column(db.Boolean, nullable=False, default=False)
    priority = db.Column(db.String(20), nullable=False, default="routine")
    status = db.Column(db.String(20), nullable=False, default="draft", index=True)
    target_json = db.Column(db.Text, nullable=False, default='{"scope":"all"}')
    version = db.Column(db.Integer, nullable=False, default=1)
    original_filename = db.Column(db.String(255), nullable=False, default="")
    stored_filename = db.Column(db.String(255), nullable=False, default="")
    content_type = db.Column(db.String(120), nullable=False, default="")
    content_sha256 = db.Column(db.String(64), nullable=False, default="")
    created_by_id = db.Column(db.Integer, nullable=False)
    created_by_name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    published_at = db.Column(db.DateTime)
    withdrawn_at = db.Column(db.DateTime)


class BriefingDelivery(db.Model):
    __tablename__ = "briefing_delivery"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, nullable=False, index=True)
    briefing_id = db.Column(
        db.Integer, db.ForeignKey("briefing_item.id"),
        nullable=False, index=True,
    )
    recipient_id = db.Column(db.Integer, nullable=False, index=True)
    recipient_name = db.Column(db.String(80), nullable=False)
    delivered_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    first_opened_at = db.Column(db.DateTime)
    last_opened_at = db.Column(db.DateTime)
    active_view_seconds = db.Column(db.Integer, nullable=False, default=0)
    acknowledged_at = db.Column(db.DateTime)
    acknowledged_version = db.Column(db.Integer)
    archived_at = db.Column(db.DateTime)
    deleted_at = db.Column(db.DateTime)
    __table_args__ = (
        db.UniqueConstraint(
            "unit_id", "briefing_id", "recipient_id",
            name="uq_briefing_delivery_recipient",
        ),
    )


class BriefingAudit(db.Model):
    __tablename__ = "briefing_audit"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, nullable=False, index=True)
    briefing_id = db.Column(db.Integer, index=True)
    delivery_id = db.Column(db.Integer, index=True)
    actor_id = db.Column(db.Integer, nullable=False)
    actor_name = db.Column(db.String(80), nullable=False)
    event_type = db.Column(db.String(40), nullable=False, index=True)
    occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    detail_json = db.Column(db.Text, nullable=False, default="{}")


class BriefingAssuranceRun(db.Model):
    __tablename__ = "briefing_assurance_run"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, nullable=False, index=True)
    operational_date = db.Column(db.Date, nullable=False, index=True)
    run_by_id = db.Column(db.Integer, nullable=False)
    run_by_name = db.Column(db.String(80), nullable=False)
    run_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    result_json = db.Column(db.Text, nullable=False)


def _app_models():
    # Imported lazily to avoid a circular import while app.py registers models.
    import app as roster_app
    return roster_app


def briefing_enabled(unit_id: int | None = None) -> bool:
    roster_app = _app_models()
    resolved = int(unit_id or getattr(current_user, "unit_id", 0) or 0)
    if not resolved:
        return False
    row = roster_app.FeatureFlag.query.filter_by(
        unit_id=resolved, key="briefing_module", enabled=True
    ).first()
    return bool(row)


def _require_module() -> None:
    if not briefing_enabled():
        abort(404)


def _require_admin() -> None:
    _require_module()
    if not _app_models().is_admin_user(current_user):
        abort(403)


def _target(item: BriefingItem) -> dict:
    try:
        value = json.loads(item.target_json or "{}")
    except (TypeError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


def _matches_target(person, target: dict) -> bool:
    scope = target.get("scope", "all")
    if scope == "all":
        return True
    if scope == "operational":
        return bool(person.is_operational)
    if scope == "watch":
        return str(person.watch_id or "") in {
            str(value) for value in target.get("watch_ids", [])
        }
    if scope == "role":
        return person.role in target.get("roles", [])
    if scope == "individual":
        return str(person.id) in {
            str(value) for value in target.get("staff_ids", [])
        }
    return False


def _target_from_form() -> dict:
    scope = (request.form.get("target_scope") or "all").strip()
    if scope not in {"all", "operational", "watch", "role", "individual"}:
        abort(400, "Unknown briefing audience.")
    target = {"scope": scope}
    if scope == "watch":
        target["watch_ids"] = [
            int(value) for value in request.form.getlist("watch_ids")
            if value.isdigit()
        ]
    elif scope == "role":
        allowed = {"admin", "editor", "user"}
        target["roles"] = [
            value for value in request.form.getlist("roles")
            if value in allowed
        ]
    elif scope == "individual":
        target["staff_ids"] = [
            int(value) for value in request.form.getlist("staff_ids")
            if value.isdigit()
        ]
    if scope not in {"all", "operational"} and len(target) == 1:
        abort(400, "Choose at least one recipient group.")
    return target


def _parse_local_datetime(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        abort(400, f"Enter a valid {label}.")
    # Existing operational timestamps are stored without timezone conversion.
    return parsed


def briefing_local_now(unit_id: int | None = None) -> datetime:
    """Return naive wall-clock time for the airport's configured timezone."""
    roster_app = _app_models()
    resolved = int(unit_id or getattr(current_user, "unit_id", 0) or 0)
    unit = roster_app.db.session.get(roster_app.Unit, resolved) if resolved else None
    timezone_name = (getattr(unit, "timezone", "") or "Europe/London").strip()
    try:
        local_zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        current_app.logger.error(
            "invalid_airport_timezone unit_id=%s timezone=%s",
            resolved,
            timezone_name,
        )
        local_zone = ZoneInfo("Europe/London")
    return datetime.now(local_zone).replace(tzinfo=None)


def _audit(
    event_type: str, item: BriefingItem | None = None,
    delivery: BriefingDelivery | None = None, **detail,
) -> None:
    db.session.add(BriefingAudit(
        unit_id=current_user.unit_id,
        briefing_id=item.id if item else None,
        delivery_id=delivery.id if delivery else None,
        actor_id=current_user.id,
        actor_name=current_user.name,
        event_type=event_type,
        detail_json=json.dumps(detail, default=str, sort_keys=True),
    ))


def _storage():
    return configured_briefing_storage(current_app.instance_path)


def _store_document(upload, unit_id: int) -> tuple[str, str, str, str]:
    original = secure_filename(upload.filename or "")
    extension = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if extension not in ALLOWED_DOCUMENTS:
        abort(400, "Upload a PDF or DOCX document.")
    content = upload.read()
    if not content:
        abort(400, "The uploaded document is empty.")
    if extension == "pdf" and not content.startswith(b"%PDF-"):
        abort(400, "The selected file is not a valid PDF.")
    if extension == "docx" and not content.startswith(b"PK"):
        abort(400, "The selected file is not a valid DOCX package.")
    digest = hashlib.sha256(content).hexdigest()
    stored = (
        f"airports/{unit_id}/controlled-documents/"
        f"{secrets.token_hex(24)}.{extension}"
    )
    try:
        _storage().put(stored, content, ALLOWED_DOCUMENTS[extension], digest)
    except BriefingStorageError as exc:
        current_app.logger.exception("briefing_document_storage_failed")
        abort(503, str(exc))
    return original[:255], stored, ALLOWED_DOCUMENTS[extension], digest


def _publish(item: BriefingItem) -> int:
    roster_app = _app_models()
    people = roster_app.Staff.query.filter_by(
        unit_id=current_user.unit_id, membership_status="active"
    ).order_by(roster_app.Staff.name).all()
    target = _target(item)
    recipients = [person for person in people if _matches_target(person, target)]
    if not recipients:
        abort(400, "The selected audience contains no active users.")
    now = utcnow()
    for person in recipients:
        db.session.add(BriefingDelivery(
            unit_id=current_user.unit_id,
            briefing_id=item.id,
            recipient_id=person.id,
            recipient_name=person.name,
            delivered_at=now,
        ))
    item.status = "published"
    item.published_at = now
    _audit("published", item, recipient_count=len(recipients))
    return len(recipients)


@briefing_blueprint.before_request
def _module_boundary():
    if current_user.is_authenticated:
        _require_module()


@briefing_blueprint.get("/")
@login_required
def home():
    current_time = briefing_local_now()
    deliveries = (
        db.session.query(BriefingDelivery, BriefingItem)
        .join(BriefingItem, BriefingItem.id == BriefingDelivery.briefing_id)
        .filter(
            BriefingDelivery.recipient_id == current_user.id,
            BriefingDelivery.archived_at.is_(None),
            BriefingDelivery.deleted_at.is_(None),
            BriefingItem.status == "published",
            BriefingItem.effective_at <= current_time,
            BriefingItem.expires_at >= current_time,
        )
        .order_by(BriefingItem.effective_at.desc())
        .all()
    )
    daily_deliveries = [
        row for row in deliveries if row[1].kind == "daily"
    ]
    mandatory_deliveries = [
        row for row in deliveries
        if row[1].kind != "daily" and row[1].mandatory
    ]
    other_deliveries = [
        row for row in deliveries
        if row[1].kind != "daily" and not row[1].mandatory
    ]
    return render_template(
        "briefing/home.html", deliveries=deliveries,
        daily_deliveries=daily_deliveries,
        mandatory_deliveries=mandatory_deliveries,
        other_deliveries=other_deliveries,
        briefing_current_time=current_time,
    )


@briefing_blueprint.get("/archive")
@login_required
def archive():
    rows = (
        db.session.query(BriefingDelivery, BriefingItem)
        .join(BriefingItem, BriefingItem.id == BriefingDelivery.briefing_id)
        .filter(
            BriefingDelivery.recipient_id == current_user.id,
            BriefingDelivery.archived_at.is_not(None),
            BriefingDelivery.deleted_at.is_(None),
        )
        .order_by(BriefingItem.kind, BriefingDelivery.archived_at.desc())
        .all()
    )
    grouped = {}
    for delivery, item in rows:
        if item.kind == "instruction":
            label = item.message_type_name or "Uncategorised instructions"
        elif item.kind == "daily":
            label = "Briefs of the day"
        else:
            label = "NOTAMs"
        grouped.setdefault(label, []).append((delivery, item))
    groups = [
        {"label": label, "rows": grouped[label]}
        for label in sorted(grouped, key=str.casefold)
    ]
    return render_template("briefing/archive.html", archive_groups=groups)


def _personal_delivery(item_id: int) -> tuple[BriefingDelivery, BriefingItem]:
    row = (
        db.session.query(BriefingDelivery, BriefingItem)
        .join(BriefingItem, BriefingItem.id == BriefingDelivery.briefing_id)
        .filter(
            BriefingDelivery.unit_id == current_user.unit_id,
            BriefingDelivery.recipient_id == current_user.id,
            BriefingDelivery.briefing_id == item_id,
            BriefingDelivery.deleted_at.is_(None),
        )
        .first()
    )
    if not row:
        abort(404)
    return row


@briefing_blueprint.post("/item/<int:item_id>/archive")
@login_required
def archive_item(item_id: int):
    delivery, item = _personal_delivery(item_id)
    if delivery.acknowledged_at is None:
        abort(409, "A briefing must be acknowledged before it can be archived.")
    delivery.archived_at = utcnow()
    _audit("recipient_archived", item, delivery)
    db.session.commit()
    flash("Briefing moved to your archive.", "ok")
    return redirect(url_for("briefing.home"))


@briefing_blueprint.post("/item/<int:item_id>/delete")
@login_required
def delete_item(item_id: int):
    delivery, item = _personal_delivery(item_id)
    if delivery.acknowledged_at is None:
        abort(409, "A briefing must be acknowledged before it can be removed.")
    delivery.deleted_at = utcnow()
    _audit(
        "recipient_deleted", item, delivery,
        previously_archived=bool(delivery.archived_at),
    )
    db.session.commit()
    flash("Briefing removed from your personal view.", "ok")
    destination = (
        "briefing.archive" if delivery.archived_at else "briefing.home"
    )
    return redirect(url_for(destination))


@briefing_blueprint.get("/item/<int:item_id>")
@login_required
def view_item(item_id: int):
    item = BriefingItem.query.filter_by(
        id=item_id, unit_id=current_user.unit_id, status="published"
    ).first_or_404()
    delivery = BriefingDelivery.query.filter_by(
        unit_id=current_user.unit_id,
        briefing_id=item.id,
        recipient_id=current_user.id,
        deleted_at=None,
    ).first_or_404()
    now = utcnow()
    if delivery.first_opened_at is None:
        delivery.first_opened_at = now
        _audit("first_opened", item, delivery, version=item.version)
    else:
        _audit("opened", item, delivery, version=item.version)
    delivery.last_opened_at = now
    db.session.commit()
    return render_template("briefing/item.html", item=item, delivery=delivery)


@briefing_blueprint.post("/item/<int:item_id>/heartbeat")
@login_required
def heartbeat(item_id: int):
    delivery = BriefingDelivery.query.filter_by(
        unit_id=current_user.unit_id,
        briefing_id=item_id,
        recipient_id=current_user.id,
        deleted_at=None,
    ).first_or_404()
    try:
        seconds = int(request.form.get("seconds") or 0)
    except ValueError:
        abort(400)
    seconds = max(0, min(seconds, MAX_ACTIVE_VIEW_SECONDS_PER_HEARTBEAT))
    delivery.active_view_seconds += seconds
    delivery.last_opened_at = utcnow()
    db.session.commit()
    return jsonify({"active_view_seconds": delivery.active_view_seconds})


@briefing_blueprint.post("/item/<int:item_id>/acknowledge")
@login_required
def acknowledge(item_id: int):
    item = BriefingItem.query.filter_by(
        id=item_id, unit_id=current_user.unit_id, status="published"
    ).first_or_404()
    delivery = BriefingDelivery.query.filter_by(
        unit_id=current_user.unit_id,
        briefing_id=item.id,
        recipient_id=current_user.id,
        deleted_at=None,
    ).first_or_404()
    if request.form.get("confirmation") != "yes":
        abort(400, "Confirm that you have read and understood the briefing.")
    delivery.acknowledged_at = utcnow()
    delivery.acknowledged_version = item.version
    _audit("acknowledged", item, delivery, version=item.version)
    db.session.commit()
    flash("Briefing acknowledged.", "ok")
    return redirect(url_for("briefing.home"))


@briefing_blueprint.get("/item/<int:item_id>/document")
@login_required
def document(item_id: int):
    item = BriefingItem.query.filter_by(
        id=item_id, unit_id=current_user.unit_id
    ).first_or_404()
    if not _app_models().is_admin_user(current_user):
        BriefingDelivery.query.filter_by(
            unit_id=current_user.unit_id,
            briefing_id=item.id,
            recipient_id=current_user.id,
            deleted_at=None,
        ).first_or_404()
    if not item.stored_filename:
        abort(404)
    try:
        content = _storage().get(item.stored_filename)
    except BriefingStorageError:
        current_app.logger.exception("briefing_document_read_failed")
        abort(404)
    download = request.args.get("download") == "1"
    response = send_file(
        io.BytesIO(content),
        mimetype=item.content_type, download_name=item.original_filename,
        as_attachment=download or item.content_type != "application/pdf",
        conditional=True,
    )
    if item.content_type == "application/pdf" and not download:
        # The application defaults to denying all framing. This private endpoint
        # is the single exception so its PDF can be shown by our own reader.
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        # Do not sandbox the response: Safari's native PDF renderer will open a
        # sandboxed PDF in a tab but may leave the embedded frame blank.
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    return response


@briefing_blueprint.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    _require_admin()
    roster_app = _app_models()
    if request.method == "POST":
        kind = (request.form.get("kind") or "instruction").strip()
        if kind not in {"instruction", "daily"}:
            abort(400)
        message_type = None
        if kind == "instruction":
            raw_message_type_id = request.form.get("message_type_id") or ""
            if not raw_message_type_id.isdigit():
                abort(400, "Choose an instruction message type.")
            message_type = BriefingMessageType.query.filter_by(
                id=int(raw_message_type_id),
                unit_id=current_user.unit_id,
                active=True,
            ).first()
            if not message_type:
                abort(400, "Choose an active instruction message type.")
        title = (request.form.get("title") or "").strip()[:160]
        if not title:
            abort(400, "Instruction title is required.")
        effective_at = _parse_local_datetime(
            request.form.get("effective_at") or "", "effective date"
        )
        expires_at = _parse_local_datetime(
            request.form.get("expires_at") or "", "expiry date"
        )
        if expires_at <= effective_at:
            abort(400, "Expiry must be after the effective date.")
        item = BriefingItem(
            unit_id=current_user.unit_id,
            kind=kind,
            title=title,
            message_type_id=message_type.id if message_type else None,
            message_type_name=message_type.name if message_type else "",
            body=(request.form.get("body") or "").strip(),
            effective_at=effective_at,
            expires_at=expires_at,
            mandatory=(
                kind == "instruction"
                and request.form.get("mandatory") == "yes"
            ),
            priority="routine",
            target_json=json.dumps(_target_from_form(), sort_keys=True),
            created_by_id=current_user.id,
            created_by_name=current_user.name,
        )
        upload = request.files.get("document")
        if kind == "instruction":
            if not upload or not upload.filename:
                abort(400, "Choose a PDF or DOCX instruction.")
            (
                item.original_filename, item.stored_filename,
                item.content_type, item.content_sha256,
            ) = _store_document(upload, current_user.unit_id)
        elif not item.body:
            abort(400, "Enter the briefing text.")
        db.session.add(item)
        db.session.flush()
        _audit("created", item, version=item.version, kind=item.kind)
        recipient_count = 0
        if request.form.get("action") == "publish":
            recipient_count = _publish(item)
        db.session.commit()
        flash(
            f"Briefing published to {recipient_count} users."
            if recipient_count else "Briefing saved as a draft.",
            "ok",
        )
        return redirect(url_for("briefing.admin"))
    items = BriefingItem.query.filter_by(
        unit_id=current_user.unit_id
    ).order_by(BriefingItem.created_at.desc()).all()
    current_time = briefing_local_now()
    current_items = [
        item for item in items if item.expires_at >= current_time
    ]
    historic_groups = {}
    for item in sorted(
        (item for item in items if item.expires_at < current_time),
        key=lambda row: row.expires_at,
        reverse=True,
    ):
        year = item.expires_at.year
        month = item.expires_at.month
        historic_groups.setdefault(year, {}).setdefault(month, []).append(item)
    historic_years = [
        {
            "year": year,
            "count": sum(len(rows) for rows in months.values()),
            "months": [
                {
                    "number": month,
                    "label": datetime(year, month, 1).strftime("%B"),
                    "rows": months[month],
                }
                for month in sorted(months, reverse=True)
            ],
        }
        for year, months in sorted(
            historic_groups.items(), reverse=True
        )
    ]
    watches = roster_app.Watch.query.order_by(
        roster_app.Watch.order_index, roster_app.Watch.name
    ).all()
    people = roster_app.Staff.query.filter_by(
        membership_status="active"
    ).order_by(roster_app.Staff.name).all()
    message_types = BriefingMessageType.query.order_by(
        BriefingMessageType.active.desc(),
        BriefingMessageType.display_order,
        BriefingMessageType.name,
    ).all()
    try:
        storage_ok, storage_message = _storage().health()
    except BriefingStorageError as exc:
        storage_ok, storage_message = False, str(exc)
    return render_template(
        "briefing/admin.html", current_items=current_items,
        historic_years=historic_years, watches=watches, people=people,
        message_types=message_types,
        storage_ok=storage_ok, storage_message=storage_message,
    )


@briefing_blueprint.post("/admin/message-types/configure")
@login_required
def configure_message_types():
    _require_admin()
    names = []
    seen = set()
    for raw_name in (request.form.get("message_types") or "").splitlines():
        name = raw_name.strip()[:80]
        key = name.casefold()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    if not names:
        abort(400, "Enter at least one instruction message type.")

    existing = BriefingMessageType.query.order_by(
        BriefingMessageType.display_order,
        BriefingMessageType.name,
    ).all()
    existing_by_name = {row.name.casefold(): row for row in existing}
    now = utcnow()
    for order, name in enumerate(names, start=1):
        row = existing_by_name.get(name.casefold())
        if row is None:
            row = BriefingMessageType(
                unit_id=current_user.unit_id,
                name=name,
                created_at=now,
            )
            db.session.add(row)
        row.active = True
        row.display_order = order * 10
        row.updated_at = now
    for row in existing:
        if row.name.casefold() not in seen:
            row.active = False
            row.updated_at = now
    _audit("message_types_configured", active_names=names)
    db.session.commit()
    flash("Instruction message types updated.", "ok")
    return redirect(url_for("briefing.settings"))


@briefing_blueprint.get("/admin/settings")
@login_required
def settings():
    _require_admin()
    message_types = BriefingMessageType.query.order_by(
        BriefingMessageType.active.desc(),
        BriefingMessageType.display_order,
        BriefingMessageType.name,
    ).all()
    return render_template(
        "briefing/settings.html",
        message_types=message_types,
    )


@briefing_blueprint.post("/admin/<int:item_id>/publish")
@login_required
def publish(item_id: int):
    _require_admin()
    item = BriefingItem.query.filter_by(
        id=item_id, unit_id=current_user.unit_id, status="draft"
    ).first_or_404()
    count = _publish(item)
    db.session.commit()
    flash(f"Briefing published to {count} users.", "ok")
    return redirect(url_for("briefing.admin"))


@briefing_blueprint.post("/admin/<int:item_id>/withdraw")
@login_required
def withdraw(item_id: int):
    _require_admin()
    item = BriefingItem.query.filter_by(
        id=item_id, unit_id=current_user.unit_id, status="published"
    ).first_or_404()
    item.status = "withdrawn"
    item.withdrawn_at = utcnow()
    _audit("withdrawn", item)
    db.session.commit()
    flash("Briefing withdrawn. Its audit history has been retained.", "ok")
    return redirect(url_for("briefing.admin"))


def _duty_start(assignment, shift) -> datetime:
    return datetime.combine(assignment.day, shift.start_time or time.min)


@briefing_blueprint.route("/admin/reports", methods=["GET", "POST"])
@login_required
def assurance():
    _require_admin()
    roster_app = _app_models()
    selected_date = briefing_local_now().date()
    results = None
    run = None
    if request.method == "POST":
        try:
            selected_date = date.fromisoformat(request.form.get("date") or "")
        except ValueError:
            abort(400, "Choose a valid operational date.")
        publication = roster_app._active_roster_publication(
            selected_date.year, selected_date.month
        )
        if not publication:
            abort(
                409,
                "Briefing reports require an active published roster.",
            )
        shift_map = {
            row.code: row for row in roster_app.ShiftType.query.all()
        }
        working_codes = {
            code for code, shift in shift_map.items() if shift.is_working
        }
        people = roster_app.Staff.query.filter_by(
            membership_status="active"
        ).order_by(roster_app.Staff.name).all()
        people_by_id = {person.id: person for person in people}

        identities = roster_app.PlatformIdentity.query.filter(
            roster_app.db.func.lower(
                roster_app.PlatformIdentity.username
            ).in_([person.username.lower() for person in people])
        ).all()
        identities_by_username = {
            identity.username.lower(): identity for identity in identities
        }
        unit = roster_app.db.session.get(
            roster_app.Unit, current_user.unit_id
        )
        try:
            airport_zone = ZoneInfo(
                (getattr(unit, "timezone", "") or "Europe/London").strip()
            )
        except ZoneInfoNotFoundError:
            airport_zone = ZoneInfo("Europe/London")

        last_rostered = {}
        working_assignments = (
            roster_app.Assignment.query
            .filter(
                roster_app.Assignment.day <= selected_date,
                roster_app.Assignment.code.in_(working_codes),
            )
            .order_by(
                roster_app.Assignment.staff_id,
                roster_app.Assignment.day.desc(),
            )
            .all()
        )
        for assignment in working_assignments:
            last_rostered.setdefault(assignment.staff_id, assignment.day)

        login_roster = []
        for person in people:
            identity = identities_by_username.get(person.username.lower())
            last_login_at = getattr(identity, "last_active_at", None)
            if last_login_at:
                if last_login_at.tzinfo is None:
                    last_login_at = last_login_at.replace(tzinfo=timezone.utc)
                last_login_date = last_login_at.astimezone(
                    airport_zone
                ).date()
            else:
                last_login_date = None
            rostered_date = last_rostered.get(person.id)
            login_roster.append({
                "staff_id": person.id,
                "name": person.name,
                "last_login_date": (
                    last_login_date.isoformat() if last_login_date else None
                ),
                "last_rostered_date": (
                    rostered_date.isoformat() if rostered_date else None
                ),
                "different": last_login_date != rostered_date,
            })

        assignments = roster_app.Assignment.query.filter_by(
            day=selected_date
        ).order_by(roster_app.Assignment.staff_id).all()
        on_duty_mandatory = []
        for assignment in assignments:
            shift = shift_map.get(assignment.code)
            if not shift or not shift.is_working:
                continue
            person = people_by_id.get(assignment.staff_id)
            if not person:
                continue
            duty_start = _duty_start(assignment, shift)
            rows = (
                db.session.query(BriefingDelivery, BriefingItem)
                .join(
                    BriefingItem,
                    BriefingItem.id == BriefingDelivery.briefing_id,
                )
                .filter(
                    BriefingDelivery.recipient_id == person.id,
                    BriefingItem.status == "published",
                    BriefingItem.mandatory.is_(True),
                    BriefingItem.effective_at <= duty_start,
                    BriefingItem.expires_at >= duty_start,
                )
                .all()
            )
            outstanding = []
            for delivery, item in rows:
                if (
                    not delivery.acknowledged_at
                    or delivery.acknowledged_at.replace(tzinfo=None) > duty_start
                    or delivery.acknowledged_version != item.version
                ):
                    outstanding.append({
                        "title": item.title,
                        "opened": bool(delivery.first_opened_at),
                    })
            if outstanding:
                on_duty_mandatory.append({
                    "staff_id": person.id,
                    "name": person.name,
                    "shift": assignment.code,
                    "duty_start": duty_start.isoformat(),
                    "outstanding": outstanding,
                })

        briefing_now = briefing_local_now()
        unread_rows = (
            db.session.query(BriefingDelivery, BriefingItem)
            .join(
                BriefingItem,
                BriefingItem.id == BriefingDelivery.briefing_id,
            )
            .filter(
                BriefingDelivery.recipient_id.in_(list(people_by_id)),
                BriefingDelivery.deleted_at.is_(None),
                BriefingItem.kind == "instruction",
                BriefingItem.status == "published",
                BriefingItem.effective_at <= briefing_now,
                BriefingItem.expires_at >= briefing_now,
                db.or_(
                    BriefingDelivery.acknowledged_at.is_(None),
                    BriefingDelivery.acknowledged_version
                    != BriefingItem.version,
                ),
            )
            .order_by(
                BriefingDelivery.recipient_id,
                BriefingItem.effective_at,
            )
            .all()
        )
        unread_by_staff = {person.id: [] for person in people}
        for delivery, item in unread_rows:
            unread_by_staff.setdefault(delivery.recipient_id, []).append({
                "title": item.title,
                "message_type": (
                    item.message_type_name or "Uncategorised instruction"
                ),
                "mandatory": item.mandatory,
                "opened": bool(delivery.first_opened_at),
                "effective_at": item.effective_at.isoformat(),
                "expires_at": item.expires_at.isoformat(),
            })
        unread_profiles = [{
            "staff_id": person.id,
            "name": person.name,
            "count": len(unread_by_staff.get(person.id, [])),
            "mandatory_count": sum(
                1 for item in unread_by_staff.get(person.id, [])
                if item["mandatory"]
            ),
            "instructions": unread_by_staff.get(person.id, []),
        } for person in people]

        read_rows = (
            db.session.query(BriefingDelivery, BriefingItem)
            .join(
                BriefingItem,
                BriefingItem.id == BriefingDelivery.briefing_id,
            )
            .filter(
                BriefingDelivery.unit_id == current_user.unit_id,
                BriefingDelivery.recipient_id.in_(list(people_by_id)),
                BriefingDelivery.acknowledged_at.is_not(None),
                BriefingItem.kind == "instruction",
            )
            .order_by(
                BriefingDelivery.recipient_id,
                BriefingDelivery.acknowledged_at.desc(),
            )
            .all()
        )
        read_by_staff = {person.id: [] for person in people}
        for delivery, item in read_rows:
            read_by_staff.setdefault(delivery.recipient_id, []).append({
                "title": item.title,
                "message_type": (
                    item.message_type_name or "Uncategorised instruction"
                ),
                "acknowledged_at": delivery.acknowledged_at.isoformat(),
                "active_view_seconds": delivery.active_view_seconds,
                "version": delivery.acknowledged_version or item.version,
            })
        read_profiles = [{
            "staff_id": person.id,
            "name": person.name,
            "count": len(read_by_staff.get(person.id, [])),
            "total_active_view_seconds": sum(
                item["active_view_seconds"]
                for item in read_by_staff.get(person.id, [])
            ),
            "instructions": read_by_staff.get(person.id, []),
        } for person in people]

        results = {
            "login_roster": login_roster,
            "on_duty_mandatory": on_duty_mandatory,
            "unread_profiles": unread_profiles,
            "read_profiles": read_profiles,
        }
        run = BriefingAssuranceRun(
            unit_id=current_user.unit_id,
            operational_date=selected_date,
            run_by_id=current_user.id,
            run_by_name=current_user.name,
            result_json=json.dumps(results, sort_keys=True),
        )
        db.session.add(run)
        db.session.flush()
        _audit(
            "report_run", None, operational_date=selected_date,
            result_count=len(people), run_id=run.id,
            on_duty_exception_count=len(on_duty_mandatory),
            unread_instruction_count=len(unread_rows),
            read_instruction_count=len(read_rows),
            roster_publication_id=publication.id,
            roster_version=publication.version,
        )
        db.session.commit()
    previous_runs = BriefingAssuranceRun.query.filter_by(
        unit_id=current_user.unit_id
    ).order_by(BriefingAssuranceRun.run_at.desc()).limit(20).all()
    previous_reports = []
    for previous in previous_runs:
        try:
            previous_result = json.loads(previous.result_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            previous_result = {}
        if not isinstance(previous_result, dict):
            previous_result = {}

        def report_rows(key):
            rows = previous_result.get(key)
            return (
                [row for row in rows if isinstance(row, dict)]
                if isinstance(rows, list) else []
            )

        def count_value(row, key):
            value = row.get(key, 0)
            return value if isinstance(value, int) and value >= 0 else 0

        login_rows = report_rows("login_roster")
        unread_profiles = report_rows("unread_profiles")
        read_profiles = report_rows("read_profiles")
        previous_reports.append({
            "run": previous,
            "users_checked": len(login_rows),
            "login_roster_differences": sum(
                1 for row in login_rows if row.get("different")
            ),
            "on_duty_exceptions": len(
                report_rows("on_duty_mandatory")
            ),
            "unread_instructions": sum(
                count_value(profile, "count") for profile in unread_profiles
            ),
            "read_instructions": sum(
                count_value(profile, "count") for profile in read_profiles
            ),
            "reading_seconds": sum(
                count_value(profile, "total_active_view_seconds")
                for profile in read_profiles
            ),
        })
    return render_template(
        "briefing/assurance.html", selected_date=selected_date,
        results=results, run=run, previous_reports=previous_reports,
    )


@briefing_blueprint.post("/admin/reports/<int:run_id>/delete")
@login_required
def delete_assurance_report(run_id: int):
    _require_admin()
    report = BriefingAssuranceRun.query.filter_by(
        id=run_id, unit_id=current_user.unit_id
    ).first_or_404()
    _audit(
        "report_deleted", None, run_id=report.id,
        operational_date=report.operational_date,
        original_run_at=report.run_at,
        original_run_by=report.run_by_name,
    )
    db.session.delete(report)
    db.session.commit()
    flash("Previous briefing report deleted.", "ok")
    return redirect(url_for("briefing.assurance"))


@briefing_blueprint.get("/admin/assurance")
@login_required
def legacy_assurance():
    _require_admin()
    return redirect(url_for("briefing.assurance"), code=308)


@briefing_blueprint.get("/admin/audit")
@login_required
def audit():
    _require_admin()
    events = BriefingAudit.query.filter_by(
        unit_id=current_user.unit_id
    ).order_by(BriefingAudit.occurred_at.desc()).limit(500).all()
    return render_template("briefing/audit.html", events=events)
