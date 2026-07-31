import flask as _flask
from functools import wraps
from calendar import monthrange
from collections import defaultdict, Counter, OrderedDict, deque
from typing import Optional, Tuple
import base64
from urllib import parse as urllib_parse, request as urllib_request, error as urllib_error
from flask import Flask, render_template, request, redirect, url_for, flash, Response, abort, session, g, send_from_directory, jsonify
from flask import render_template as flask_render_template
import os
import re
import io
import csv
import secrets
from functools import lru_cache
from datetime import date, datetime, time, timedelta, timezone
import json
import json as _json
import logging
import click
import hashlib
import pyotp
import qrcode
import qrcode.image.svg
from cryptography.fernet import Fernet, InvalidToken

from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy.session import Session as FlaskSqlAlchemySession
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, SecurityError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from rate_limiting import (
    LimiterUnavailable, MemoryRateLimiter, RedisRateLimiter, privacy_key,
)
from reporting import (
    compute_annotation_metrics,
    current_leave_year_window,
    financial_year_start,
    group_consecutive_days,
    leave_summary_for_month,
)
from roster_logic import (
    add_months,
    daily_requirements,
    expand_pattern,
    iter_year_months,
    month_days,
    normalise_assignment_snapshot,
    parse_year_month,
    roster_lock_date,
    roster_month_is_locked,
    shift_minutes,
    validated_pattern,
)
from absence_requests import (
    add_months as add_request_months,
    group_sickness_instances,
    normalise_request_rules,
    request_date_bounds,
    request_lock_date,
    request_month_is_locked,
    safe_admin_month,
)
from access_policy import (
    has_permission,
    is_admin,
    is_editor,
    is_trainee,
    may_apply_annotations,
    may_edit_roster,
    may_manage_training,
    may_override_roster_conflicts,
    may_record_training,
    may_send_unit_messages,
    permissions_for,
)
from atcroster import create_app, get_runtime_settings
from tenancy import (
    authenticated_unit_id,
    bind_authenticated_unit,
    bind_platform_control,
    clear_request_context,
    operational_engine_for_authenticated_unit,
    operational_unit_context,
    reset_authenticated_unit,
    reset_platform_control,
)

try:
    from flask_caching import Cache
except Exception:
    Cache = None

# -------------------- App setup --------------------
app = create_app()
_runtime_settings = get_runtime_settings(app)

# Writable local instance folder for development and tests.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = app.instance_path
os.makedirs(INSTANCE_DIR, exist_ok=True)

DEPLOYMENT_ENV = _runtime_settings.deployment_environment
FIELD_ENCRYPTION_KEY = _runtime_settings.field_encryption_key
FIELD_ENCRYPTION_KEYS = _runtime_settings.field_encryption_keys

if DEPLOYMENT_ENV == "production":
    import redis
    _rate_limiter = RedisRateLimiter(redis.from_url(
        os.environ["REDIS_URL"], socket_connect_timeout=2,
        socket_timeout=2, decode_responses=True,
    ))
else:
    _rate_limiter = MemoryRateLimiter()


def _field_ciphers() -> list[tuple[str, Fernet]]:
    result = []
    for item in FIELD_ENCRYPTION_KEYS.split(","):
        version, separator, key = item.strip().partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", version):
            raise RuntimeError("Invalid field-encryption key version.")
        try:
            result.append((version, Fernet(key.encode())))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Invalid field-encryption key material.") from exc
    if not result:
        raise RuntimeError("At least one field-encryption key is required.")
    return result


def _encrypt_field(value: str) -> str:
    version, cipher = _field_ciphers()[0]
    return f"{version}.{cipher.encrypt(value.encode()).decode()}"


def _decrypt_field(value: str) -> str:
    version, separator, ciphertext = value.partition(".")
    if separator:
        candidates = [
            cipher for candidate, cipher in _field_ciphers()
            if candidate == version
        ]
    else:
        ciphertext = value
        candidates = [cipher for _version, cipher in _field_ciphers()]
    for cipher in candidates:
        try:
            return cipher.decrypt(ciphertext.encode()).decode()
        except InvalidToken:
            continue
    raise ValueError("Encrypted field cannot be decrypted with configured keys.")


# Validate configured material during startup rather than at first MFA use.
_field_ciphers()

# Jinja helper
app.jinja_env.globals['now'] = lambda: datetime.now()


def _asset_url(filename: str, **extra: object) -> str:
    """Return a cache-busting static asset URL using the file mtime."""

    static_folder = app.static_folder
    version: Optional[int] = None

    if static_folder:
        try:
            path = os.path.join(static_folder, filename)
            version = int(os.path.getmtime(path))
        except (OSError, TypeError, ValueError):
            version = None

    if version is not None:
        return url_for("static", filename=filename, v=version, **extra)

    return url_for("static", filename=filename, **extra)


app.jinja_env.globals["asset_url"] = _asset_url


def utcnow():
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


REQUEST_STATUSES = frozenset({"pending", "approved", "rejected", "fulfilled", "cancelled"})
REQUEST_TRANSITIONS = {
    "pending": frozenset({"approved", "rejected", "cancelled"}),
    "approved": frozenset({"rejected", "cancelled"}),
    "rejected": frozenset(),
    "cancelled": frozenset(),
    "fulfilled": frozenset(),
}
PLATFORM_FEATURE_FLAGS = frozenset({
    "advanced_coverage", "scenario_planning", "calendar_exports",
    "fatigue_reporting", "custom_branding", "briefing_module",
    "training_module", "competency_module",
})


def _current_unit_id() -> int:
    """Derive tenancy from the authenticated membership, never request data."""
    return int(getattr(current_user, "unit_id", 0) or 0)


def csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _validate_csrf() -> None:
    supplied = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
    expected = session.get("_csrf_token")
    if not expected or not supplied or not secrets.compare_digest(str(expected), str(supplied)):
        abort(400, "Invalid or missing CSRF token.")


app.jinja_env.globals["csrf_token"] = csrf_token


def csp_nonce() -> str:
    return getattr(g, "csp_nonce", "")


app.jinja_env.globals["csp_nonce"] = csp_nonce


@app.before_request
def _start_request_tenant_boundary():
    clear_request_context()


@app.before_request
def _enforce_csrf():
    """Apply a default-deny CSRF boundary to every browser mutation.

    There are currently no exempt unsafe routes. New machine-to-machine
    endpoints must use a separate authenticated blueprint and document any
    exemption explicitly rather than weakening this browser boundary.
    """
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        _validate_csrf()


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder, "favicon.svg", mimetype="image/svg+xml"
    )


def _legal_context() -> dict[str, str]:
    """Public contact details used by the legal and privacy notices."""
    return {
        "legal_entity": os.environ.get(
            "ATCROSTER_LEGAL_ENTITY",
            "Ian John Dickson trading as IDAviation",
        ).strip(),
        "privacy_email": os.environ.get(
            "ATCROSTER_PRIVACY_EMAIL",
            os.environ.get(
                "ATCROSTER_SUPPORT_EMAIL", "privacy@atcroster.com"
            ),
        ).strip(),
        "legal_address": os.environ.get(
            "ATCROSTER_LEGAL_ADDRESS",
            "Flat 0/2, 24 Caird Drive, Glasgow, Scotland, G11 5DT",
        ).strip(),
        "company_number": os.environ.get(
            "ATCROSTER_COMPANY_NUMBER", ""
        ).strip(),
        "policy_date": "27 July 2026",
    }


@app.get("/privacy")
def privacy_notice():
    return render_template("privacy.html", **_legal_context())


@app.get("/cookies")
def cookie_notice():
    return render_template("cookies.html", **_legal_context())


@app.get("/terms")
def terms_of_service():
    return render_template("terms.html", **_legal_context())


@app.get("/subprocessors")
def subprocessor_notice():
    return render_template("subprocessors.html", **_legal_context())


@app.get("/health/live")
def health_live():
    return jsonify({
        "status": "ok",
        "service": "atcroster",
        "environment": DEPLOYMENT_ENV,
    })


@app.get("/health/ready")
def health_ready():
    try:
        connection = db.session.connection()
        connection.execute(text("SELECT 1"))
        from sqlalchemy import inspect
        from alembic.config import Config as AlembicConfig
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from migrations.fresh_schema import CONTROL_TABLES

        present = set(inspect(connection).get_table_names())
        revision = MigrationContext.configure(connection).get_current_revision()
        expected_revision = ScriptDirectory.from_config(
            AlembicConfig("alembic.ini")
        ).get_current_head()
        if (
            not CONTROL_TABLES.issubset(present)
            or (
                DEPLOYMENT_ENV == "production"
                and revision != expected_revision
            )
        ):
            return jsonify({"status": "not_ready"}), 503
        return jsonify({"status": "ready"})
    except Exception:
        app.logger.error(
            "readiness_check_failed request_id=%s",
            getattr(g, "request_id", ""),
        )
        return jsonify({"status": "not_ready"}), 503


@app.errorhandler(500)
def _internal_error(error):
    app.logger.error(
        "unhandled_request_error request_id=%s path=%s",
        getattr(g, "request_id", ""), request.path, exc_info=error,
    )
    module_context = _module_error_navigation()
    return render_template(
        "error.html", request_id=getattr(g, "request_id", ""),
        **module_context,
    ), 500


def _module_error_navigation() -> dict[str, str]:
    """Keep module errors inside the module instead of linking to Roster."""
    if request.path.startswith("/briefing"):
        return {
            "home_url": url_for("briefing.home"),
            "home_label": "Return to briefing",
        }
    if request.path.startswith("/training"):
        return {
            "home_url": url_for("training_home"),
            "home_label": "Return to training",
        }
    if request.path.startswith("/competency"):
        return {
            "home_url": url_for("competency_home"),
            "home_label": "Return to competency",
        }
    return {}


@app.errorhandler(400)
def _bad_request(error):
    if isinstance(error, SecurityError):
        return Response(
            "Bad Request: untrusted host.",
            status=400,
            content_type="text/plain; charset=utf-8",
        )
    description = getattr(error, "description", "") or ""
    if "CSRF" in description:
        message = (
            "This page or form has expired. Reload the page and try the action "
            "once more."
        )
    elif description and not description.startswith(
        "The browser (or proxy) sent a request"
    ):
        message = description
    else:
        message = (
            "The request was not valid. Check the entered values and try again."
        )
    return render_template(
        "error.html",
        status_code=400,
        error_title="We could not validate that request",
        error_message=message,
        request_id=getattr(g, "request_id", ""),
        **_module_error_navigation(),
    ), 400


@app.errorhandler(403)
def _forbidden(_error):
    is_platform_admin = (
        getattr(current_user, "is_authenticated", False)
        and getattr(current_user, "role", "") == "superadmin"
    )
    module_context = _module_error_navigation()
    return render_template(
        "error.html",
        status_code=403,
        error_title="You do not have access to this area",
        error_message=(
            "Platform administrators cannot access airport personnel or "
            "operational roster data. Return to Platform Administration."
            if is_platform_admin
            else (
                "Your account role does not permit this action. Ask your Unit "
                "Administrator for access."
            )
        ),
        home_url=(
            url_for("platform_admin") if is_platform_admin
            else module_context.get("home_url", url_for("index"))
        ),
        home_label=(
            "Return to Platform Administration"
            if is_platform_admin
            else module_context.get("home_label", "Return to roster")
        ),
        request_id=getattr(g, "request_id", ""),
    ), 403


@app.errorhandler(404)
def _not_found(_error):
    return render_template(
        "error.html",
        status_code=404,
        error_title="That page or record was not found",
        error_message=(
            "It may have moved, been removed, or belong to a different airport."
        ),
        request_id=getattr(g, "request_id", ""),
        **_module_error_navigation(),
    ), 404

OPERATIONAL_TABLE_NAMES = frozenset({
    "roster_setting", "annotation_type", "watch", "staff", "shift_type",
    "requirement", "special_requirement", "leave", "sickness", "assignment",
    "shift_request", "sms_audit",
    "request_audit", "notification", "annotation_audit", "ai_rule_set",
    "change_log", "staff_watch_history", "qualification_type",
    "person_qualification", "person_qualification_history",
    "roster_publication", "roster_acknowledgement", "scenario",
    "operational_position", "position_endorsement", "position_requirement",
    "break_plan", "achieved_duty", "fatigue_report",
    "roster_rule_version", "mfa_credential",
    "briefing_item", "briefing_delivery", "briefing_audit",
    "briefing_assurance_run", "briefing_message_type",
    "training_level", "training_objective", "training_session",
    "training_score",
})


class TenantRoutedSession(FlaskSqlAlchemySession):
    """Route operational mappers to the authenticated airport database."""

    def get_bind(self, mapper=None, clause=None, bind=None, **kwargs):
        if bind is not None:
            return bind
        table_name = None
        if mapper is not None:
            try:
                table_name = mapper.persist_selectable.name
            except AttributeError:
                try:
                    table_name = mapper.__table__.name
                except AttributeError:
                    table_name = None
        if table_name in OPERATIONAL_TABLE_NAMES:
            try:
                return operational_engine_for_authenticated_unit()
            except RuntimeError:
                # Local legacy databases remain available only outside
                # production while they are imported as the first unit.
                if DEPLOYMENT_ENV == "production":
                    raise RuntimeError(
                        "Operational database access requires an authenticated "
                        "airport route."
                    )
        return super().get_bind(
            mapper=mapper, clause=clause, bind=bind, **kwargs
        )


# Database & login
db = SQLAlchemy(app, session_options={
    "expire_on_commit": False,
    "class_": TenantRoutedSession,
})
login_manager = LoginManager(app)
login_manager.login_view = "login"


@app.before_request
def _bind_tenant_context():
    clear_request_context()
    g.request_id = request.headers.get("X-Request-ID") or secrets.token_hex(12)
    g.csp_nonce = secrets.token_urlsafe(18)
    g.tenant_context_token = None
    g.platform_control_token = None
    if current_user.is_authenticated:
        now_epoch = int(utcnow().timestamp())
        idle_limit = int(
            os.environ.get("ATCROSTER_SESSION_IDLE_MINUTES", "30")
        ) * 60
        last_seen = int(session.get("_last_seen_epoch") or now_epoch)
        absolute_limit = int(
            os.environ.get("ATCROSTER_SESSION_ABSOLUTE_MINUTES", "720")
        ) * 60
        started_raw = session.get("_session_started_at")
        try:
            started_epoch = int(
                datetime.fromisoformat(str(started_raw)).timestamp()
            )
        except (TypeError, ValueError):
            started_epoch = now_epoch
            session["_session_started_at"] = utcnow().isoformat()
        expiry_reason = (
            "absolute" if now_epoch - started_epoch > absolute_limit
            else "idle" if now_epoch - last_seen > idle_limit
            else ""
        )
        if expiry_reason:
            _security_event(
                "session_expired", reason=expiry_reason,
                principal=hashlib.sha256(
                    str(current_user.get_id()).encode()
                ).hexdigest()[:16],
            )
            logout_user()
            session.clear()
            flash("Your secure session has expired. Sign in again.", "error")
            return redirect(url_for("login"))
        expected_stamp = session.get("_auth_stamp")
        current_stamp = _current_auth_stamp(current_user)
        if expected_stamp and not secrets.compare_digest(
            str(expected_stamp), current_stamp
        ):
            _security_event(
                "session_forced_invalidation",
                principal=hashlib.sha256(
                    str(current_user.get_id()).encode()
                ).hexdigest()[:16],
            )
            logout_user()
            session.clear()
            flash(
                "Your account security or permissions changed. Sign in again.",
                "error",
            )
            return redirect(url_for("login"))
        session["_auth_stamp"] = current_stamp
        session["_last_seen_epoch"] = now_epoch
    if (
        current_user.is_authenticated
        and getattr(current_user, "role", "") == "superadmin"
    ):
        g.platform_control_token = bind_platform_control()
        if not session.get("_platform_mfa_verified"):
            logout_user()
            session.clear()
            return redirect(url_for("login"))
        allowed_platform_endpoints = {
            "platform_admin", "logout", "password_change",
            "platform_worker_health",
            "static", "favicon", "health_live", "health_ready",
        }
        if request.endpoint == "index":
            return redirect(url_for("platform_admin"))
        if request.endpoint not in allowed_platform_endpoints:
            abort(403)
    if current_user.is_authenticated and getattr(current_user, "role", "") != "superadmin":
        unit_id = int(getattr(current_user, "unit_id", 0) or 0)
        if unit_id and g.tenant_context_token is None:
            routing = db.session.get(DatabaseRoutingMetadata, unit_id)
            if DEPLOYMENT_ENV == "production" and not routing:
                abort(503, "Operational database routing is unavailable.")
            g.tenant_context_token = bind_authenticated_unit(
                unit_id, routing.secret_name if routing else None
            )
    if (
        current_user.is_authenticated
        and getattr(current_user, "role", "") != "superadmin"
        and (
            DEPLOYMENT_ENV == "production"
            or UnitMembership.query.filter_by(
                person_id=current_user.id,
                unit_id=getattr(current_user, "unit_id", 0),
                role="UnitAdmin",
                status="active",
            ).first() is not None
        )
        and request.endpoint not in {
            "mfa_setup", "logout", "static", "favicon",
            "health_live", "health_ready",
        }
    ):
        credential = MfaCredential.query.filter_by(
            person_id=current_user.id, enabled=True
        ).first()
        if not credential:
            return redirect(url_for("mfa_setup"))


@app.after_request
def _security_headers(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
    )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        f"script-src 'self' 'nonce-{getattr(g, 'csp_nonce', '')}' "
        "https://cdn.jsdelivr.net",
    )
    if request.is_secure or DEPLOYMENT_ENV == "production":
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    if current_user.is_authenticated:
        response.headers.setdefault("Cache-Control", "no-store, private")
    return response


@app.teardown_request
def _reset_tenant_context(_error=None):
    token = getattr(g, "tenant_context_token", None)
    g.tenant_context_token = None
    if token is not None:
        try:
            reset_authenticated_unit(token)
        except RuntimeError:
            # Flask test/request contexts may invoke teardown more than once.
            pass
    platform_token = getattr(g, "platform_control_token", None)
    g.platform_control_token = None
    if platform_token is not None:
        try:
            reset_platform_control(platform_token)
        except RuntimeError:
            pass
    clear_request_context()


_LOGIN_NEXT_ENDPOINTS = {
    "/": "index",
    "/modules": "module_home",
    "/briefing/": "briefing.home",
    "/training/": "training_home",
    "/competency/": "competency_home",
    "/admin": "admin",
    "/leave": "leave",
    "/messages": "unit_messages",
    "/overtime": "overtime",
    "/platform/admin": "platform_admin",
    "/reports": "reports_index",
    "/reports/leave-year": "report_leave_year",
    "/reports/sickness": "report_sickness",
    "/requests": "requests_page",
    "/unit/accounts": "unit_accounts",
}


def _canonical_login_redirect(
    target: str | None,
    *,
    default_endpoint: str = "index",
    user_id: int | None = None,
) -> str:
    """Return only a URL generated from an explicitly permitted app route."""
    if not target:
        return url_for(default_endpoint)
    parsed = urllib_parse.urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return url_for(default_endpoint)
    endpoint = _LOGIN_NEXT_ENDPOINTS.get(parsed.path)
    if endpoint:
        return url_for(endpoint)
    roster_match = re.fullmatch(r"/roster/(\d{4}-\d{2})", parsed.path)
    if roster_match:
        return url_for("roster_month", ym=roster_match.group(1))
    leave_report_match = re.fullmatch(
        r"/reports/leave/(\d{4}-\d{2})", parsed.path
    )
    if leave_report_match:
        return url_for("report_leave", ym=leave_report_match.group(1))
    profile_match = re.fullmatch(r"/staff/(\d+)", parsed.path)
    if profile_match and user_id and int(profile_match.group(1)) == user_id:
        return url_for("staff_profile", sid=user_id)
    return url_for(default_endpoint)


def _airport_login_endpoint(user) -> str:
    """Land multi-module airport users on the module launcher."""
    enabled = FeatureFlag.query.filter(
        FeatureFlag.unit_id == user.unit_id,
        FeatureFlag.key.in_((
            "briefing_module", "training_module", "competency_module"
        )),
        FeatureFlag.enabled.is_(True),
    ).first()
    return "module_home" if enabled else "index"


# ----- Lightweight caching -----
_cache = None
if Cache is not None:
    try:
        _cache = Cache(config={
            "CACHE_TYPE": "simple",            # in-memory
            "CACHE_DEFAULT_TIMEOUT": 120
        })
        _cache.init_app(app)
    except Exception:
        _cache = None


def _memoize(seconds=60):
    def wrap(fn):
        if _cache:
            return _cache.memoize(timeout=seconds)(fn)
        return fn
    return wrap


def _invalidate_month_cache_for_day(d: date):
    if _cache and d:
        try:
            _cache.delete_memoized(
                _load_month_roster_fast,
                int(_current_unit_id() or 1),
                d.year,
                d.month,
            )
        except Exception:
            pass


def _twilio_credentials() -> tuple[str, str, str]:
    return (
        os.getenv("TWILIO_ACCOUNT_SID", ""),
        os.getenv("TWILIO_AUTH_TOKEN", ""),
        os.getenv("TWILIO_FROM_NUMBER", ""),
    )


def _normalise_sms_number(value: str | None) -> str:
    """Return an E.164 number, accepting harmless display punctuation."""
    candidate = re.sub(r"[\s().-]+", "", value or "")
    return candidate if re.fullmatch(r"\+[1-9]\d{7,14}", candidate) else ""


def _sms_number_options(key: str, unit_id: int | None = None) -> list[dict[str, str]]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get(key, "[]")
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        values = []
    result = []
    seen = set()
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        number = _normalise_sms_number(item.get("number"))
        if not number or number in seen:
            continue
        seen.add(number)
        result.append({
            "number": number,
            "label": (str(item.get("label") or number).strip()[:80] or number),
        })
    return result


def _sms_sender_options(unit_id: int | None = None) -> list[dict[str, str]]:
    options = _sms_number_options("sms_sender_numbers", unit_id)
    fallback = _normalise_sms_number(_twilio_credentials()[2])
    if fallback and not options:
        options = [{"number": fallback, "label": "Twilio account default"}]
    return options


def _sms_operational_options(unit_id: int | None = None) -> list[dict[str, str]]:
    return _sms_number_options("sms_operational_numbers", unit_id)


def _sms_default_number(
    setting_key: str, options: list[dict[str, str]], unit_id: int | None = None
) -> str:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    configured = _normalise_sms_number(
        _roster_settings_snapshot(resolved_unit_id).get(setting_key)
    )
    allowed = {item["number"] for item in options}
    return configured if configured in allowed else (
        options[0]["number"] if options else ""
    )


def _sms_service_configured() -> bool:
    account_sid, auth_token, _from_number = _twilio_credentials()
    return bool(account_sid and auth_token and _sms_sender_options())


def _email_service_configured() -> bool:
    return bool(
        os.getenv("SMTP_HOST")
        and os.getenv("SMTP_FROM_ADDRESS")
    )


def _send_account_email(to_address: str, subject: str, body: str) -> bool:
    """Send a plain-text account email without logging its contents."""
    if not to_address or not _email_service_configured():
        return False
    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["To"] = to_address
    message["From"] = os.environ["SMTP_FROM_ADDRESS"]
    message["Subject"] = subject[:160]
    message.set_content(body)
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {
        "1", "true", "yes"
    }
    try:
        with smtplib.SMTP(host, port, timeout=10) as connection:
            if use_tls:
                connection.starttls()
            if username:
                connection.login(username, password)
            connection.send_message(message)
        return True
    except Exception:
        app.logger.exception("account_email_delivery_failed")
        return False


def _valid_email(value: str) -> str:
    candidate = (value or "").strip().casefold()
    if not re.fullmatch(
        r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
        r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+",
        candidate,
        re.IGNORECASE,
    ):
        return ""
    return candidate[:254]


def _platform_support_emails() -> list[str]:
    configured = _valid_email(os.getenv("ATCROSTER_SUPPORT_EMAIL", ""))
    rows = PlatformIdentity.query.filter(
        PlatformIdentity.public_id.like("platform-%"),
        PlatformIdentity.email != "",
    ).all()
    return list(dict.fromkeys(
        [address for address in [configured, *(row.email for row in rows)]
         if address]
    ))


def _unit_admin_emails(unit_id: int) -> list[str]:
    rows = (
        db.session.query(PlatformIdentity)
        .join(
            UnitMembership,
            UnitMembership.identity_id == PlatformIdentity.id,
        )
        .filter(
            UnitMembership.unit_id == unit_id,
            UnitMembership.status == "active",
            UnitMembership.role == "UnitAdmin",
            PlatformIdentity.email != "",
        )
        .all()
    )
    return list(dict.fromkeys(row.email for row in rows if row.email))


def _send_sms_via_twilio(
    to_number: str,
    body: str,
    creds: tuple[str, str, str] | None = None,
    from_number: str | None = None,
) -> tuple[bool, str]:
    account_sid, auth_token, configured_from = creds or _twilio_credentials()
    from_number = _normalise_sms_number(from_number or configured_from)
    if not (account_sid and auth_token and from_number):
        return False, "SMS credentials are not configured."

    to_number = _normalise_sms_number(to_number)
    if not to_number:
        return False, "Missing or invalid destination number."

    payload = urllib_parse.urlencode({
        "To": to_number,
        "From": from_number,
        "Body": body,
    }).encode("utf-8")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    req = urllib_request.Request(url, data=payload, method="POST")
    token = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
    req.add_header("Authorization", f"Basic {token}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        # The destination is the fixed HTTPS Twilio API origin above.
        with urllib_request.urlopen(req, timeout=10) as resp:  # nosec B310
            data = resp.read().decode("utf-8")
            if 200 <= resp.status < 300:
                try:
                    parsed = json.loads(data)
                except Exception:
                    parsed = {}
                return True, parsed.get("sid", "sent")
            return False, f"HTTP {resp.status}: {data[:200]}"
    except urllib_error.HTTPError as err:
        try:
            detail = err.read().decode("utf-8")
            parsed = json.loads(detail)
            message = parsed.get("message") or detail
        except Exception:
            message = getattr(err, "reason", None) or str(err)
        return False, f"{err.code}: {message}"
    except urllib_error.URLError as err:
        return False, getattr(err, "reason", None) or str(err)
    except Exception as exc:
        return False, str(exc)


def _record_sms_audit(
    *,
    sender_number: str,
    recipient_number: str,
    recipient_label: str,
    body: str,
    message_type: str,
    provider_message_id: str,
) -> None:
    """Persist one successful provider delivery in the airport database."""
    db.session.add(SmsAudit(
        unit_id=_current_unit_id(),
        sent_by_staff_id=current_user.id,
        sent_by_name=current_user.name,
        sender_number=_normalise_sms_number(sender_number),
        recipient_number=_normalise_sms_number(recipient_number),
        recipient_label=(recipient_label or recipient_number)[:120],
        message_type=(message_type or "unit")[:30],
        message_content=body,
        provider_message_id=(provider_message_id or "")[:64],
    ))
    db.session.commit()


def _send_overtime_sms_notifications(
    staff_list: list["Staff"], message: str
) -> tuple[int, list[tuple[Optional["Staff"], str]]]:
    creds = _twilio_credentials()
    sender_options = _sms_sender_options()
    from_number = _sms_default_number(
        "sms_default_sender", sender_options
    )
    if not (creds[0] and creds[1] and from_number):
        return 0, [(None, "SMS sending is not configured." )]

    sent = 0
    failures: list[tuple[Optional["Staff"], str]] = []
    for staff in staff_list:
        if not (staff and staff.phone_number):
            failures.append((staff, "No phone number on file."))
            continue
        ok, detail = _send_sms_via_twilio(
            staff.phone_number, message, creds, from_number
        )
        if ok:
            _record_sms_audit(
                sender_number=from_number,
                recipient_number=staff.phone_number,
                recipient_label=staff.name,
                body=message,
                message_type="overtime",
                provider_message_id=detail,
            )
            sent += 1
        else:
            failures.append((staff, detail))
    return sent, failures


def _default_overtime_sms_body(chosen_date: date | None, shift_code: str | None) -> str:
    if not (chosen_date and shift_code):
        return ""
    return (f"Overtime available on {chosen_date.isoformat()} for {shift_code} shift. "
            "Please reply if interested.")


def _flash_sms_result(
    sent: int, failures: list[tuple[Optional["Staff"], str]]
) -> None:
    if sent:
        flash(f"SMS sent to {sent} recipient{'s' if sent != 1 else ''}.", "ok")
    if failures:
        details = "; ".join(
            f"{staff.name if staff else 'System'}: {reason}"
            for staff, reason in failures[:8]
        )
        if len(failures) > 8:
            details += f"; and {len(failures) - 8} more"
        flash(f"Some messages were not sent. {details}", "error")


MIN_MONTH = date(2025, 4, 1)   # Start app from April 2025

# Reference defaults (used if DB rows missing)
DEFAULT_WORKING_CODES = ["M", "D", "A", "N", "SC", "SSC", "SBY"]
DEFAULT_BANNED_ROSTER_CODES = ["SIC", "SC", "SSC", "AL", "SP", "SPL", "PL", "TOU8", "TOUI"]
DEFAULT_EXCLUDE_FROM_COUNTERS = ["OSS"]
DEFAULT_NON_WORKING_CODES = [
    "OFF", "AL", "PL", "SPL", "TOU8", "TOUI", "OSS", "OFFICE", "WFH", "CTB", "MTG"
]

DEFAULT_ANNOTATION_TYPES = [
    {
        "code": "INFO",
        "label": "Information",
        "category": "Information",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "info,report_exclude",
        "is_active": True,
        "sort_order": 0,
    },
    {
        "code": "EXTS",
        "label": "Short EXT",
        "category": "Extensions",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ext,ext_short",
        "is_active": True,
    },
    {
        "code": "EXTL",
        "label": "Long EXT",
        "category": "Extensions",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ext,ext_long",
        "is_active": True,
    },
    {
        "code": "SWAP",
        "label": "Swap",
        "category": "Swaps",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "swap",
        "is_active": True,
    },
    {
        "code": "A2",
        "label": "A2",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A4",
        "label": "A4",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A6",
        "label": "A6",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A8",
        "label": "A8",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "SOAL",
        "label": "SOAL",
        "category": "Overtime",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ot,soal",
        "is_active": True,
    },
    {
        "code": "TOA8",
        "label": "TOA8 (TOIL +1.0)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 2,
        "tags": "toil",
        "is_active": True,
    },
    {
        "code": "TOAI",
        "label": "TOAI (TOIL +0.5)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 1,
        "tags": "toil",
        "is_active": True,
    },
    {
        "code": "TOAU",
        "label": "TOAU (legacy)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 2,
        "tags": "toil",
        "is_active": False,
    },
]

DEFAULT_ROSTER_SETTINGS = {
    "working_codes": DEFAULT_WORKING_CODES,
    "banned_codes": DEFAULT_BANNED_ROSTER_CODES,
    "exclude_from_counters": DEFAULT_EXCLUDE_FROM_COUNTERS,
    "non_working_codes": DEFAULT_NON_WORKING_CODES,
}

DEFAULT_ABSENCE_TYPES = [
    {"code": "AL", "label": "Annual leave", "category": "leave", "active": True},
    {"code": "PL", "label": "Parental leave", "category": "leave", "active": True},
    {"code": "SPL", "label": "Special leave", "category": "leave", "active": True},
    {"code": "SC", "label": "Sickness", "category": "sickness", "active": True},
    {"code": "SSC", "label": "Self-certified sickness", "category": "sickness", "active": True},
]

# -------------------- Models --------------------


class RosterSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(
        db.Integer, db.ForeignKey("unit.id"),
        nullable=False, default=1, index=True,
    )
    key = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Text, nullable=False, default="")
    __table_args__ = (
        db.UniqueConstraint(
            "unit_id", "key", name="uq_roster_setting_unit_key"
        ),
    )


class Unit(db.Model):
    """An airport tenant. Operational rows always belong to exactly one unit."""
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(12), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    timezone = db.Column(db.String(64), nullable=False, default="Europe/London")
    locale = db.Column(db.String(20), nullable=False, default="en-GB")
    date_format = db.Column(db.String(30), nullable=False, default="%d/%m/%Y")
    branding_json = db.Column(db.Text, nullable=False, default="{}")
    status = db.Column(db.String(20), nullable=False, default="active")
    plan = db.Column(db.String(40), nullable=False, default="starter")
    request_months_ahead = db.Column(db.Integer, nullable=False, default=3)
    request_lock_day = db.Column(db.Integer, nullable=False, default=20)
    active_user_limit = db.Column(db.Integer, nullable=False, default=10)
    onboarding_step = db.Column(db.Integer, nullable=False, default=1)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    trial_ends_at = db.Column(db.DateTime)
    renews_at = db.Column(db.DateTime)
    suspended_at = db.Column(db.DateTime)
    last_active_at = db.Column(db.DateTime)


class AnnotationType(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    code = db.Column(db.String(10), nullable=False)
    label = db.Column(db.String(80), nullable=False, default="")
    category = db.Column(db.String(40), nullable=False, default="General")
    colour = db.Column(db.String(20), nullable=False, default="#6c757d")
    description = db.Column(db.Text, nullable=False, default="")
    allow_suffix = db.Column(db.Boolean, default=False)
    suffixes = db.Column(db.String(20), default="")
    toil_half_days = db.Column(db.Integer, default=0)
    tags = db.Column(db.String(200), default="")
    note_required = db.Column(db.Boolean, default=False)
    admin_only = db.Column(db.Boolean, default=False)
    has_been_used = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    sort_order = db.Column(db.Integer, default=100)
    __table_args__ = (db.UniqueConstraint("unit_id", "code", name="uq_annotation_unit_code"),)


# -------------------- Models --------------------


class Watch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    name = db.Column(db.String(32), nullable=False)
    order_index = db.Column(db.Integer, nullable=False, default=0)
    pattern_csv = db.Column(db.String(500), nullable=False, default="")
    pattern_anchor = db.Column(db.Date)
    __table_args__ = (db.UniqueConstraint("unit_id", "name", name="uq_watch_unit_name"),)


class Staff(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    def get_id(self) -> str:
        membership = UnitMembership.query.filter_by(
            unit_id=self.unit_id,
            person_id=self.id,
            status="active",
        ).order_by(UnitMembership.id).first()
        if membership:
            return f"membership:{membership.id}"
        return f"legacy:{self.unit_id}:{self.id}"

    def set_password(self, password: str) -> None:
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        from werkzeug.security import check_password_hash
        # be robust if password_hash is None/empty
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)

    # Auth
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    # Legacy login remains globally unique until all deployments use
    # PlatformIdentity. This prevents ambiguous cross-unit authentication.
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(254), nullable=False, default="")

    # Roles: 'admin' | 'editor' | 'user'
    role = db.Column(db.String(10), nullable=False, default="user")
    membership_status = db.Column(db.String(20), nullable=False, default="active")
    permissions_json = db.Column(db.Text, nullable=False, default="{}")

    phone_number = db.Column(db.String(30), default="")

    @property
    def is_admin_role(self) -> bool:
        return (self.role or "user") == "admin"

    @property
    def is_editor_role(self) -> bool:
        return (self.role or "user") in ("editor", "admin")

    # Back-compat (kept but unused in logic)
    is_admin = db.Column(db.Boolean, default=False)

    # Public ICS token for calendar subscription
    calendar_token = db.Column(db.String(64), unique=True, nullable=True)

    # Identity / roster fields
    name = db.Column(db.String(80), nullable=False)
    staff_no = db.Column(db.String(20), nullable=False)
    caa_license_number = db.Column(db.String(40), nullable=False, default="")

    watch_id = db.Column(db.Integer, db.ForeignKey("watch.id"))
    watch = db.relationship("Watch", backref="members")

    medical_expiry = db.Column(db.Date, nullable=True)
    tower_ue_expiry = db.Column(db.Date, nullable=True)
    radar_ue_expiry = db.Column(db.Date, nullable=True)
    tower_ut = db.Column(db.Boolean, default=False)
    radar_ut = db.Column(db.Boolean, default=False)
    # --- MET qualification ---
    met_ue_expiry = db.Column(db.Date, nullable=True)
    met_ut = db.Column(db.Boolean, default=False)

    # Assessor flag
    has_assessor = db.Column(db.Boolean, default=False)

    is_operational = db.Column(db.Boolean, default=True)
    is_trainee = db.Column(db.Boolean, default=False)
    has_ojti = db.Column(db.Boolean, default=False)

    # NEW: watch manager flags + OT opt-out
    is_wm = db.Column(db.Boolean, default=False)
    is_dwm = db.Column(db.Boolean, default=False)
    exclude_from_ot = db.Column(db.Boolean, default=False)

    pattern_csv = db.Column(db.String, default="M,M,A,A,N,N,OFF,OFF,OFF,OFF")
    pattern_anchor = db.Column(db.Date, nullable=True)
    pattern_override = db.Column(db.Boolean, nullable=False, default=False)

    # TOIL: store in HALF-DAYS (1 day = 2 half-days)
    toil_half_days = db.Column(db.Integer, default=0)

    # Leave-year config per person
    leave_year_start_month = db.Column(db.Integer, default=4)  # 1..12
    leave_entitlement_days = db.Column(db.Integer, default=0)
    leave_public_holidays = db.Column(db.Integer, default=0)
    leave_carryover_days = db.Column(db.Integer, default=0)
    __table_args__ = (
        db.UniqueConstraint("unit_id", "staff_no", name="uq_staff_unit_number"),
    )


class TrainingLevel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    name = db.Column(db.String(80), nullable=False)
    sort_order = db.Column(db.Integer, nullable=False, default=100)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    __table_args__ = (
        db.UniqueConstraint("unit_id", "name", name="uq_training_level_unit_name"),
    )


class TrainingObjective(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    level_id = db.Column(db.Integer, db.ForeignKey("training_level.id"), nullable=False, index=True)
    position = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False, default="")
    level = db.relationship("TrainingLevel", backref="objectives")
    __table_args__ = (
        db.UniqueConstraint("level_id", "position", name="uq_training_objective_position"),
    )


class TrainingSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    trainee_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
    ojti_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
    level_id = db.Column(db.Integer, db.ForeignKey("training_level.id"), nullable=False, index=True)
    training_date = db.Column(db.Date, nullable=False, index=True)
    duration_minutes = db.Column(db.Integer, nullable=False)
    summary = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    trainee = db.relationship("Staff", foreign_keys=[trainee_id])
    ojti = db.relationship("Staff", foreign_keys=[ojti_id])
    level = db.relationship("TrainingLevel")


class TrainingScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("training_session.id"), nullable=False, index=True)
    objective_id = db.Column(db.Integer, db.ForeignKey("training_objective.id"), nullable=False, index=True)
    attainment = db.Column(db.Integer, nullable=False)
    assistance = db.Column(db.Integer, nullable=False)
    safety_critical = db.Column(db.Boolean, nullable=False, default=False)
    note = db.Column(db.Text, nullable=False, default="")
    session = db.relationship("TrainingSession", backref=db.backref("scores", cascade="all, delete-orphan"))
    objective = db.relationship("TrainingObjective")
    __table_args__ = (
        db.UniqueConstraint("session_id", "objective_id", name="uq_training_score_objective"),
    )


class ShiftType(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    code = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(40), nullable=False, default="")
    start_time = db.Column(db.Time, nullable=True)
    end_time = db.Column(db.Time, nullable=True)
    is_working = db.Column(db.Boolean, default=True)
    # training flag (counts to fatigue but excluded from daily M/D/A/N counters)
    is_training = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_requestable = db.Column(db.Boolean, nullable=False, default=False)
    required_qualification = db.Column(db.String(40), nullable=False, default="")
    __table_args__ = (db.UniqueConstraint("unit_id", "code", name="uq_shift_unit_code"),)


class Requirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    req_m = db.Column(db.Integer, default=0)
    req_d = db.Column(db.Integer, default=0)
    req_a = db.Column(db.Integer, default=0)
    req_n = db.Column(db.Integer, default=0)
    req_sat_m = db.Column(db.Integer, default=0)
    req_sat_d = db.Column(db.Integer, default=0)
    req_sat_a = db.Column(db.Integer, default=0)
    req_sat_n = db.Column(db.Integer, default=0)
    req_sun_m = db.Column(db.Integer, default=0)
    req_sun_d = db.Column(db.Integer, default=0)
    req_sun_a = db.Column(db.Integer, default=0)
    req_sun_n = db.Column(db.Integer, default=0)
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "year", "month", name="uniq_unit_year_month"),)


class SpecialRequirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(
        db.Integer, db.ForeignKey("unit.id"), nullable=False,
        default=1, index=True,
    )
    day = db.Column(db.Date, nullable=False, index=True)
    label = db.Column(db.String(80), nullable=False, default="")
    req_m = db.Column(db.Integer, nullable=False, default=0)
    req_d = db.Column(db.Integer, nullable=False, default=0)
    req_a = db.Column(db.Integer, nullable=False, default=0)
    req_n = db.Column(db.Integer, nullable=False, default=0)
    __table_args__ = (
        db.UniqueConstraint(
            "unit_id", "day", name="uniq_unit_special_requirement_day"
        ),
    )


class SmsAudit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(
        db.Integer, db.ForeignKey("unit.id"), nullable=False,
        default=1, index=True,
    )
    sent_at = db.Column(
        db.DateTime, nullable=False, default=utcnow, index=True
    )
    sent_by_staff_id = db.Column(db.Integer, nullable=False, index=True)
    sent_by_name = db.Column(db.String(80), nullable=False)
    sender_number = db.Column(db.String(20), nullable=False)
    recipient_number = db.Column(db.String(20), nullable=False)
    recipient_label = db.Column(db.String(120), nullable=False)
    message_type = db.Column(db.String(30), nullable=False, default="unit")
    message_content = db.Column(db.Text, nullable=False)
    provider_message_id = db.Column(db.String(64), nullable=False, default="")


class Leave(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
    staff = db.relationship("Staff", backref="leaves")
    leave_type = db.Column(db.String(10), nullable=False)  # AL/PL/SPL only
    start = db.Column(db.Date, nullable=False)
    end = db.Column(db.Date, nullable=False)


class Sickness(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
    start = db.Column(db.Date, nullable=False)
    end = db.Column(db.Date, nullable=False)
    code = db.Column(db.String(10), nullable=False, default="SC")
    staff = db.relationship("Staff", backref="sickness_periods")


class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), index=True)
    staff = db.relationship("Staff", backref="assignments")
    day = db.Column(db.Date, index=True)
    code = db.Column(db.String(10), nullable=False)
    source = db.Column(db.String(10), default="auto")
    note = db.Column(db.String(140), default="")
    # Annotation code (managed via AnnotationType, optional suffix like A6M)
    annotation = db.Column(db.String(20), default="")
    # User-facing detail for the annotation. Kept separate from system notes.
    annotation_note = db.Column(db.String(140), default="")
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "staff_id", "day", name="uniq_unit_staff_day"),)


class ShiftRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey(
        "staff.id"), index=True, nullable=False)
    staff = db.relationship("Staff", backref="shift_requests")
    day = db.Column(db.Date, index=True, nullable=False)
    code = db.Column(db.String(10), nullable=False)
    requester_comment = db.Column(db.String(500), nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    submitted_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    fulfilled_at = db.Column(db.DateTime)
    cancelled_at = db.Column(db.DateTime)
    resulting_assignment_id = db.Column(db.Integer, db.ForeignKey("assignment.id"))
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "staff_id", "day",
        name="uniq_shift_request_unit_staff_day",
    ),)
    # >>> NEW admin response fields
    admin_response = db.Column(db.Text, default="")
    responded_by_id = db.Column(db.Integer)  # FK optional (kept simple)
    responded_at = db.Column(db.DateTime)
    dismissed_by_requester_at = db.Column(db.DateTime)
    # pending/approved/rejected/fulfilled/cancelled
    status = db.Column(db.String(20), default="pending")


class RequestAudit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    request_id = db.Column(db.Integer, db.ForeignKey("shift_request.id"), nullable=False, index=True)
    actor_id = db.Column(db.Integer, nullable=False)
    occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    transition = db.Column(db.String(30), nullable=False)
    old_value = db.Column(db.Text, nullable=False, default="")
    new_value = db.Column(db.Text, nullable=False, default="")
    reason = db.Column(db.String(500), nullable=False, default="")


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    recipient_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
    kind = db.Column(db.String(40), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    read_at = db.Column(db.DateTime)


class AnnotationAudit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    annotation_type_id = db.Column(db.Integer, db.ForeignKey("annotation_type.id"), index=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey("assignment.id"), index=True)
    actor_id = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String(30), nullable=False)
    old_value = db.Column(db.Text, nullable=False, default="")
    new_value = db.Column(db.Text, nullable=False, default="")
    occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    transaction_key = db.Column(db.String(64), unique=True)


class ChangeLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    when = db.Column(db.DateTime, nullable=False,
                     default=utcnow, index=True)
    who_user_id = db.Column(db.Integer, index=True)
    entity_type = db.Column(db.String(40), index=True)
    entity_id = db.Column(db.Integer, index=True)
    field = db.Column(db.String(40))
    old_value = db.Column(db.Text)
    new_value = db.Column(db.Text)
    context_month = db.Column(db.String(7), index=True)  # 'YYYY-MM'
    note = db.Column(db.Text, default="")


class StaffWatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey(
        "staff.id"), nullable=False, index=True)
    watch_id = db.Column(db.Integer, db.ForeignKey("watch.id"), nullable=False)
    effective_date = db.Column(db.Date, nullable=False, index=True)
    staff = db.relationship("Staff", backref="watch_history")
    watch = db.relationship("Watch")


# Control-plane and advanced product entities live in a separate module so
# they can move to the central database without rewriting the legacy UI.
from saas_models import register_saas_models
SaaS = register_saas_models(db, utcnow)
PlatformIdentity = SaaS.PlatformIdentity
PlatformMfaCredential = SaaS.PlatformMfaCredential
UnitMembership = SaaS.UnitMembership
SecureInvitation = SaaS.SecureInvitation
SignupWorkflow = SaaS.SignupWorkflow
RecoveryRequest = SaaS.RecoveryRequest
DatabaseRoutingMetadata = SaaS.DatabaseRoutingMetadata
ProvisioningJob = SaaS.ProvisioningJob
WorkerHeartbeat = SaaS.WorkerHeartbeat
FeatureFlag = SaaS.FeatureFlag
PlanHistory = SaaS.PlanHistory
AggregateUsageEvent = SaaS.AggregateUsageEvent
SuperAdminAudit = SaaS.SuperAdminAudit
CentralSecurityAudit = SaaS.CentralSecurityAudit
QualificationType = SaaS.QualificationType
PersonQualification = SaaS.PersonQualification
PersonQualificationHistory = SaaS.PersonQualificationHistory
RosterPublication = SaaS.RosterPublication
RosterAcknowledgement = SaaS.RosterAcknowledgement
Scenario = SaaS.Scenario
OperationalPosition = SaaS.OperationalPosition
PositionEndorsement = SaaS.PositionEndorsement
PositionRequirement = SaaS.PositionRequirement
BreakPlan = SaaS.BreakPlan
AchievedDuty = SaaS.AchievedDuty
FatigueReport = SaaS.FatigueReport
RosterRuleVersion = SaaS.RosterRuleVersion
MfaCredential = SaaS.MfaCredential

from briefing_module import (
    BriefingAssuranceRun, BriefingAudit, BriefingDelivery, BriefingItem,
    BriefingMessageType, briefing_blueprint, briefing_enabled,
    briefing_local_now,
)

# Enforce the authenticated airport on all legacy operational SELECTs and
# stamp new rows. This protects older routes while they move to repositories.
from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession, with_loader_criteria
from tenancy import authenticated_unit_id

TENANT_OPERATIONAL_MODELS = (
    RosterSetting, AnnotationType, Watch, Staff, ShiftType, Requirement,
    SpecialRequirement, SmsAudit, Leave, Sickness,
    Assignment, ShiftRequest, RequestAudit, Notification, AnnotationAudit,
    ChangeLog, StaffWatchHistory, QualificationType,
    PersonQualification, PersonQualificationHistory,
    RosterPublication, RosterAcknowledgement, Scenario,
    OperationalPosition, PositionEndorsement, PositionRequirement, BreakPlan,
    AchievedDuty, FatigueReport, RosterRuleVersion,
    MfaCredential, BriefingMessageType, BriefingItem, BriefingDelivery,
    BriefingAudit,
    BriefingAssuranceRun,
    TrainingLevel, TrainingObjective, TrainingSession, TrainingScore,
)


@event.listens_for(OrmSession, "do_orm_execute")
def _scope_operational_selects(execute_state):
    if not execute_state.is_select or execute_state.execution_options.get("skip_tenant_scope"):
        return
    try:
        unit_id = authenticated_unit_id()
    except RuntimeError:
        return
    statement = execute_state.statement
    for model in TENANT_OPERATIONAL_MODELS:
        statement = statement.options(with_loader_criteria(
            model, lambda cls: cls.unit_id == unit_id,
            include_aliases=True,
            track_closure_variables=True,
        ))
    execute_state.statement = statement


@event.listens_for(OrmSession, "before_flush")
def _stamp_operational_writes(session_obj, _flush_context, _instances):
    try:
        unit_id = authenticated_unit_id()
    except RuntimeError:
        return
    for record in session_obj.new:
        if isinstance(record, TENANT_OPERATIONAL_MODELS):
            supplied = getattr(record, "unit_id", None)
            if supplied not in (None, unit_id):
                raise PermissionError("Cross-unit writes are forbidden")
            record.unit_id = unit_id

# -------------------- Reference data helpers --------------------


def _normalise_codes(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = []
    for val in values:
        code = (val or "").strip().upper()
        if code and code not in seen:
            seen.append(code)
    return seen


@lru_cache(maxsize=128)
def _roster_settings_snapshot(unit_id: int) -> dict[str, str]:
    rows = RosterSetting.query.filter_by(unit_id=unit_id).all()
    return {row.key: row.value for row in rows}


def refresh_roster_settings_cache() -> None:
    _roster_settings_snapshot.cache_clear()
    try:
        _shift_groups_snapshot.cache_clear()
    except NameError:
        pass


def _load_codes_setting(
    key: str, default: list[str], unit_id: int | None = None
) -> set[str]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get(key)
    if not raw:
        parsed = default
    else:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = default
    configured = set(_normalise_codes(parsed))
    existing = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(
            unit_id=resolved_unit_id
        ).all()
    }
    return configured & existing


def get_working_codes() -> set[str]:
    return _load_codes_setting("working_codes", DEFAULT_WORKING_CODES)


def get_absence_types(
    category: str | None = None,
    active_only: bool = True,
    unit_id: int | None = None,
) -> list[dict[str, object]]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get("absence_types")
    try:
        parsed = json.loads(raw) if raw else DEFAULT_ABSENCE_TYPES
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = DEFAULT_ABSENCE_TYPES
    if not isinstance(parsed, list):
        parsed = DEFAULT_ABSENCE_TYPES
    result = []
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().upper()
        item_category = str(item.get("category") or "").strip().lower()
        if (
            not re.fullmatch(r"[A-Z0-9]{1,10}", code)
            or item_category not in {"leave", "sickness"}
            or code in seen
        ):
            continue
        seen.add(code)
        normalised = {
            "code": code,
            "label": str(item.get("label") or code).strip()[:80] or code,
            "category": item_category,
            "active": bool(item.get("active", True)),
        }
        if category and item_category != category:
            continue
        if active_only and not normalised["active"]:
            continue
        result.append(normalised)
    return result


def _save_absence_types(items: list[dict[str, object]]) -> None:
    _save_roster_setting(
        "absence_types", json.dumps(items, separators=(",", ":"))
    )
    db.session.commit()
    refresh_roster_settings_cache()


def get_banned_roster_codes() -> set[str]:
    return _load_codes_setting("banned_codes", DEFAULT_BANNED_ROSTER_CODES)


def get_exclude_from_counters() -> set[str]:
    return _load_codes_setting("exclude_from_counters", DEFAULT_EXCLUDE_FROM_COUNTERS)


def get_non_working_codes() -> set[str]:
    return _load_codes_setting("non_working_codes", DEFAULT_NON_WORKING_CODES)


def staff_is_countable_on(person: Staff, on_date: date) -> bool:
    """Return whether a person holds the credentials needed for staffing counts."""
    if not person.medical_expiry or person.medical_expiry < on_date:
        return False
    return any(
        expiry is not None and expiry >= on_date
        for expiry in (
            person.tower_ue_expiry,
            person.radar_ue_expiry,
            person.met_ue_expiry,
        )
    )


def get_shift_counter_map(unit_id: int | None = None) -> dict[str, str]:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    raw = _roster_settings_snapshot(resolved_unit_id).get(
        "shift_counter_map", "{}"
    )
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        values = {}
    if not isinstance(values, dict):
        return {}
    return {
        str(code).upper(): str(group).upper()
        for code, group in values.items()
        if str(group).upper() in {"", "M", "D", "A", "N"}
    }


def shift_counter_group(
    code: str | None, unit_id: int | None = None
) -> str:
    value = (code or "").strip().upper()
    if not value:
        return ""
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    mapping = get_shift_counter_map(resolved_unit_id)
    if value in mapping:
        return mapping[value]
    # Legacy default grouping is only valid for a working shift that actually
    # exists for this airport. Pattern letters are not, by themselves, enough
    # to claim that somebody is covering that staffing group.
    shift = get_shift(value, resolved_unit_id)
    if (
        not shift
        or not shift.is_active
        or not shift.is_working
        or shift.is_training
    ):
        return ""
    if value == "EM":
        return "M"
    if value == "LA":
        return "A"
    return value if value in {"M", "D", "A", "N"} else ""


def shift_counter_group_for_day(
    code: str | None, on_date: date, unit_id: int | None = None
) -> str:
    """Return the staffing group, suppressing nights when the unit is closed."""
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    group = shift_counter_group(code, resolved_unit_id)
    if group == "N" and not _night_active_on(resolved_unit_id, on_date):
        return ""
    return group


@lru_cache(maxsize=128)
def _annotation_snapshot(unit_id: int) -> dict[str, object]:
    rows = (AnnotationType.query
            .filter(AnnotationType.unit_id == unit_id)
            .order_by(AnnotationType.code)
            .all())
    items = []
    for row in rows:
        tags = tuple(sorted({
            t.strip().lower() for t in (row.tags or "").split(",") if t.strip()
        }))
        suffixes = "".join(sorted({c for c in (row.suffixes or "").upper()}))
        items.append({
            "id": row.id,
            "code": (row.code or "").upper(),
            "label": row.label or row.code.upper(),
            "category": row.category or "Other",
            "colour": row.colour or "#6c757d",
            "description": row.description or "",
            "allow_suffix": bool(row.allow_suffix),
            "suffixes": suffixes,
            "toil_half_days": int(row.toil_half_days or 0),
            "tags": tags,
            "note_required": bool(row.note_required),
            "admin_only": bool(row.admin_only),
            "is_active": bool(row.is_active),
            "sort_order": row.sort_order if row.sort_order is not None else 0,
        })
    by_code = {item["code"]: item for item in items}
    return {"items": items, "by_code": by_code}


def refresh_annotation_cache() -> None:
    _annotation_snapshot.cache_clear()


def get_annotation_types(
    active_only: bool = True, unit_id: int | None = None
) -> list[dict[str, object]]:
    snap = _annotation_snapshot(int(unit_id or _current_unit_id() or 1))
    items = snap["items"]
    if active_only:
        items = [item for item in items if item["is_active"]]
    return items


def get_annotation_config(
    code: str | None, unit_id: int | None = None
) -> dict[str, object] | None:
    if not code:
        return None
    return _annotation_snapshot(
        int(unit_id or _current_unit_id() or 1)
    )["by_code"].get(code.strip().upper())


def get_annotation_groups() -> OrderedDict[str, list[dict[str, object]]]:
    groups: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for item in get_annotation_types(active_only=True):
        groups.setdefault(item["category"], []).append(item)
    return groups


def annotation_tags_for(code: str | None) -> set[str]:
    info = get_annotation_config(code)
    if not info:
        return set()
    tags = info.get("tags") or ()
    return {t for t in tags}


def annotation_codes_for_tag(tag: str, active_only: bool = True) -> list[str]:
    needle = (tag or "").lower().strip()
    if not needle:
        return []
    codes = []
    for item in get_annotation_types(active_only=active_only):
        tags = {t for t in (item.get("tags") or ())}
        if needle in tags:
            codes.append(item["code"])
    return codes


def _parse_codes_input(raw: str) -> list[str]:
    tokens = re.split(r"[\s,]+", raw or "")
    return _normalise_codes(tokens)


def _save_codes_setting(key: str, values: list[str]) -> None:
    payload = json.dumps(_normalise_codes(values))
    unit_id = int(_current_unit_id() or 1)
    row = RosterSetting.query.filter_by(unit_id=unit_id, key=key).first()
    if not row:
        row = RosterSetting(unit_id=unit_id, key=key, value=payload)
        db.session.add(row)
    else:
        row.value = payload
    db.session.commit()
    refresh_roster_settings_cache()


def _prune_roster_code_settings(unit_id: int) -> int:
    """Remove list entries that have no ShiftType in this airport."""
    valid_codes = {
        str(code or "").strip().upper()
        for (code,) in db.session.query(ShiftType.code).filter_by(
            unit_id=unit_id
        ).all()
    }
    changed = 0
    rows = RosterSetting.query.filter(
        RosterSetting.unit_id == unit_id,
        RosterSetting.key.in_(DEFAULT_ROSTER_SETTINGS),
    ).all()
    for row in rows:
        try:
            values = json.loads(row.value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            values = []
        if not isinstance(values, list):
            values = []
        normalised = _normalise_codes(values)
        cleaned = [code for code in normalised if code in valid_codes]
        if cleaned != normalised:
            row.value = json.dumps(cleaned)
            changed += 1
    if changed:
        refresh_roster_settings_cache()
    return changed


def _save_roster_setting(key: str, value: str) -> None:
    unit_id = int(_current_unit_id() or 1)
    row = RosterSetting.query.filter_by(unit_id=unit_id, key=key).first()
    if not row:
        row = RosterSetting(unit_id=unit_id, key=key, value=value)
        db.session.add(row)
    else:
        row.value = value
    refresh_roster_settings_cache()


def _parse_sms_number_lines(raw: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse one `label | +number` or plain `+number` entry per line."""
    result = []
    errors = []
    seen = set()
    for line_number, raw_line in enumerate((raw or "").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        label, separator, number_value = line.partition("|")
        if not separator:
            number_value, label = label, ""
        number = _normalise_sms_number(number_value)
        if not number:
            errors.append(f"line {line_number}")
            continue
        if number in seen:
            continue
        seen.add(number)
        result.append({
            "number": number,
            "label": (label.strip()[:80] or number),
        })
    return result, errors


def bootstrap_reference_data() -> None:
    Unit.__table__.create(bind=db.engine, checkfirst=True)
    AnnotationType.__table__.create(bind=db.engine, checkfirst=True)
    RosterSetting.__table__.create(bind=db.engine, checkfirst=True)
    if Unit.query.count() == 0:
        db.session.add(Unit(
            id=1, code="FIRST", name="First airport unit",
            status="active",
        ))
        db.session.flush()

    if AnnotationType.query.count() == 0:
        for idx, cfg in enumerate(DEFAULT_ANNOTATION_TYPES):
            ann = AnnotationType(
                code=cfg.get("code", "").upper(),
                label=cfg.get("label") or cfg.get("code", ""),
                category=cfg.get("category", "Other"),
                allow_suffix=bool(cfg.get("allow_suffix", False)),
                suffixes="".join(sorted({
                    c for c in (cfg.get("suffixes") or "").upper()
                })),
                toil_half_days=int(cfg.get("toil_half_days", 0) or 0),
                tags=cfg.get("tags", ""),
                is_active=bool(cfg.get("is_active", True)),
                sort_order=cfg.get("sort_order", idx * 10),
            )
            db.session.add(ann)
        db.session.commit()

    for unit in Unit.query.filter(Unit.status != "platform_control").all():
        if not AnnotationType.query.filter_by(
            unit_id=unit.id, code="INFO"
        ).first():
            db.session.add(AnnotationType(
                unit_id=unit.id,
                code="INFO",
                label="Information",
                category="Information",
                colour="#6c757d",
                description=(
                    "Additional roster information. Excluded from reports."
                ),
                allow_suffix=False,
                suffixes="",
                toil_half_days=0,
                tags="info,report_exclude",
                note_required=False,
                admin_only=False,
                is_active=True,
                sort_order=0,
            ))

    for key, values in DEFAULT_ROSTER_SETTINGS.items():
        if not RosterSetting.query.filter_by(unit_id=1, key=key).first():
            db.session.add(RosterSetting(
                unit_id=1, key=key,
                value=json.dumps(_normalise_codes(values)),
            ))
    db.session.commit()
    refresh_annotation_cache()
    refresh_roster_settings_cache()


if (
    DEPLOYMENT_ENV != "production"
    and os.environ.get("ATCROSTER_SKIP_BOOTSTRAP", "").lower()
    not in {"1", "true", "yes"}
):
    try:
        with app.app_context():
            bootstrap_reference_data()
    except Exception:
        with app.app_context():
            db.session.rollback()

# Cached shift lookup (define after models so ShiftType exists when called)


@lru_cache(maxsize=256)
def _shift_by_code(unit_id: int, code: str):
    return ShiftType.query.filter_by(unit_id=unit_id, code=code).first()


def refresh_shift_cache():
    _shift_by_code.cache_clear()

# -------------------- Login --------------------


@login_manager.user_loader
def load_user(user_id):
    value = str(user_id or "")
    if value.startswith("membership:"):
        try:
            membership_id = int(value.split(":", 1)[1])
        except ValueError:
            return None
        membership = db.session.get(UnitMembership, membership_id)
        if not membership or membership.status != "active":
            return None
        routing = db.session.get(
            DatabaseRoutingMetadata, membership.unit_id
        )
        if DEPLOYMENT_ENV == "production" and not routing:
            return None
        token = bind_authenticated_unit(
            membership.unit_id,
            routing.secret_name if routing else None,
        )
        g.tenant_context_token = token
        return db.session.get(Staff, membership.person_id)
    if value.startswith("platform-identity:"):
        try:
            return db.session.get(
                PlatformIdentity, int(value.split(":", 1)[1])
            )
        except ValueError:
            return None
    if value.startswith("legacy:") and DEPLOYMENT_ENV != "production":
        try:
            _, raw_unit_id, raw_person_id = value.split(":", 2)
            token = bind_authenticated_unit(int(raw_unit_id))
            g.tenant_context_token = token
            return db.session.get(Staff, int(raw_person_id))
        except ValueError:
            return None
    return None

# --------- Fast month loader & cache (uses functions defined later but safe) ----------


def _load_month_roster_core(unit_id: int, y: int, m: int):
    """
    Returns (days, staff_list, a_map, req) and NEVER returns None.
    On failure: returns ([], [], {}, ensure_month_requirement(y,m)).
    """
    try:
        start = date(y, m, 1)
        days_in_m = monthrange(y, m)[1]
        days = [start + timedelta(days=i) for i in range(days_in_m)]
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = date(ny, nm, 1)

        # Staff ordering
        try:
            staff = (Staff.query
                     .outerjoin(Watch, Staff.watch_id == Watch.id)
                     .order_by(Watch.order_index, Staff.name)
                     .all())
        except Exception:
            staff = Staff.query.order_by(Staff.id).all()

        # Assignments for the month (narrow columns)
        rows = (db.session.query(
            Assignment.staff_id,
            Assignment.day,
            Assignment.code,
            Assignment.source,
            Assignment.annotation,
            Assignment.annotation_note,
        )
            .filter(Assignment.day >= start, Assignment.day < end)
            .all())

        a_map = {}
        for sid, d, code, source, ann, ann_note in rows:
            a_map.setdefault(sid, {})[d] = (
                code, source, ann, ann_note or ""
            )

        req = Requirement.query.filter_by(year=y, month=m).first()
        if not req:
            req = ensure_month_requirement(y, m)

        return days, staff, a_map, req

    except Exception as e:
        try:
            app.logger.exception(
                "Failed _load_month_roster_core(%s,%s,%s): %s",
                unit_id, y, m, e,
            )
        except Exception:
            pass
        # Ensure we still return a valid 4-tuple
        return ([], [], {}, ensure_month_requirement(y, m))


# IMPORTANT: overwrite any previously memoized wrapper
_load_month_roster_fast = _memoize(seconds=300)(_load_month_roster_core)


# -------------------- Helpers --------------------
# === Unified permissions (admins, editors, WM, DWM) ===


def is_admin_user(u) -> bool:
    return is_admin(u)


def is_editor_user(u) -> bool:
    return is_editor(u)


def user_permissions(u) -> dict[str, bool]:
    return permissions_for(u)


def has_unit_permission(u, permission: str) -> bool:
    return has_permission(u, permission)


def is_under_training(person) -> bool:
    return is_trainee(person)


def can_record_training(u) -> bool:
    return may_record_training(u)


def can_manage_training(u) -> bool:
    return may_manage_training(u)


def training_enabled(unit_id: int) -> bool:
    return bool(FeatureFlag.query.filter_by(
        unit_id=unit_id, key="training_module", enabled=True
    ).first())


def competency_enabled(unit_id: int) -> bool:
    row = FeatureFlag.query.filter_by(
        unit_id=unit_id, key="competency_module"
    ).first()
    # Existing airports inherit their current combined-module entitlement
    # until Super Admin explicitly chooses a separate competency setting.
    return bool(row.enabled) if row else training_enabled(unit_id)


def can_edit_roster(u) -> bool:
    return may_edit_roster(u)


def can_apply_annotations(u) -> bool:
    return may_apply_annotations(u)


def can_send_unit_messages(u) -> bool:
    return may_send_unit_messages(u)


def can_override_roster_conflicts(u) -> bool:
    return may_override_roster_conflicts(u)


def tenant_get(model, record_id: int):
    """Fetch one operational record with an explicit mutation-safe boundary."""
    return model.query.filter_by(
        id=int(record_id), unit_id=_current_unit_id()
    ).first()


def roster_edit_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not can_edit_roster(current_user):
            return ("Forbidden", 403)
        return f(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_perms():
    au = current_user if getattr(
        current_user, "is_authenticated", False) else None
    return {"is_admin": (bool(au) and is_admin_user(au))}


def month_has_data(year: int, month: int) -> bool:
    """Fast check: do we already have any assignments for this month?"""
    start = date(year, month, 1)
    ny, nm = _month_add(year, month, 1)
    end = date(ny, nm, 1)  # exclusive
    return db.session.query(Assignment.id)\
        .filter(Assignment.day >= start, Assignment.day < end)\
        .limit(1).first() is not None


def month_range(year: int, month: int):
    return month_days(year, month)


def watch_id_for_staff_on(staff_id: int, on_date: date) -> int | None:
    return _watch_id_for_staff_on(
        authenticated_unit_id(), staff_id, on_date
    )


@lru_cache(maxsize=4096)
def _watch_id_for_staff_on(
    unit_id: int, staff_id: int, on_date: date
) -> int | None:
    """Return the watch_id that applies to this staff on a given date
    using StaffWatchHistory; fall back to Staff.watch_id if no history."""
    hist = (StaffWatchHistory.query
            .filter(StaffWatchHistory.unit_id == unit_id,
                    StaffWatchHistory.staff_id == staff_id,
                    StaffWatchHistory.effective_date <= on_date)
            .order_by(StaffWatchHistory.effective_date.desc())
            .first())
    if hist:
        return hist.watch_id
    s = Staff.query.filter_by(
        id=staff_id, unit_id=unit_id
    ).first()
    return s.watch_id if s else None


def parse_ym(ym: str):
    return parse_year_month(ym)


def get_shift(code: str, unit_id: int | None = None):
    # hot path → use cached lookup
    return _shift_by_code(
        int(unit_id or _current_unit_id() or 1), (code or "").upper()
    )


@lru_cache(maxsize=128)
def _shift_groups_snapshot(unit_id: int):
    all_shifts = ShiftType.query.filter_by(
        unit_id=unit_id
    ).order_by(ShiftType.code).all()
    banned = get_banned_roster_codes()
    allowed = [sh for sh in all_shifts if sh.code not in banned]
    working = sorted(
        [sh for sh in allowed if sh.is_working and not sh.is_training], key=lambda s: s.code)
    training = sorted(
        [sh for sh in allowed if sh.is_training], key=lambda s: s.code)
    nonwork = sorted(
        [sh for sh in allowed if not sh.is_working and not sh.is_training], key=lambda s: s.code)
    return working, training, nonwork


PATTERN_CODES = ("M", "A", "D", "N", "OFF")
DEFAULT_BASE_PATTERN = "M,M,A,A,N,N,OFF,OFF,OFF,OFF"


def _expand_pattern(raw_value: str | None) -> list[str]:
    return expand_pattern(raw_value)


def _validated_pattern(raw_value: str | None) -> list[str]:
    return validated_pattern(raw_value)


def _effective_watch(staff: Staff, on_date: date) -> Watch | None:
    move = (
        StaffWatchHistory.query.filter(
            StaffWatchHistory.unit_id == staff.unit_id,
            StaffWatchHistory.staff_id == staff.id,
            StaffWatchHistory.effective_date <= on_date,
        )
        .order_by(
            StaffWatchHistory.effective_date.desc(),
            StaffWatchHistory.id.desc(),
        )
        .first()
    )
    return move.watch if move else staff.watch


def _unit_pattern_context(unit_id: int) -> tuple[list[str], date]:
    settings = _roster_settings_snapshot(unit_id)
    pattern = _validated_pattern(
        settings.get("base_pattern_csv") or DEFAULT_BASE_PATTERN
    )
    try:
        anchor = date.fromisoformat(
            settings.get("base_pattern_anchor") or "2025-01-01"
        )
    except ValueError:
        anchor = date(2025, 1, 1)
    return pattern or _validated_pattern(DEFAULT_BASE_PATTERN), anchor


def _pattern_context(staff: Staff, on_date: date) -> tuple[list[str], date]:
    if staff.pattern_override:
        personal = _validated_pattern(staff.pattern_csv)
        if personal:
            return personal, staff.pattern_anchor or on_date
    unit_pattern, unit_anchor = _unit_pattern_context(staff.unit_id)
    watch = _effective_watch(staff, on_date)
    if watch:
        watch_pattern = _validated_pattern(watch.pattern_csv)
        # A watch anchor phases both a watch-specific pattern and the inherited
        # unit pattern. This is what makes two watches on the same base cycle
        # start on different cycle days.
        return (
            watch_pattern or unit_pattern,
            watch.pattern_anchor or unit_anchor,
        )
    return unit_pattern, unit_anchor


def pattern_for(staff: Staff, on_date: date | None = None):
    return _pattern_context(staff, on_date or date.today())[0]


def _night_active_on(unit_id: int, on_date: date) -> bool:
    raw = _roster_settings_snapshot(unit_id).get(
        "night_active_weekdays", "0,1,2,3,4,5,6"
    )
    try:
        active_days = {
            int(value) for value in raw.split(",")
            if value.strip() != ""
        }
    except ValueError:
        active_days = set(range(7))
    return on_date.weekday() in active_days


def day_leave_for(staff: Staff, d: date):
    for lv in staff.leaves:
        if lv.start <= d <= lv.end:
            return lv.leave_type
    return None


def code_from_pattern(staff: Staff, d: date):
    pat, anchor = _pattern_context(staff, d)
    if not pat:
        return "OFF"
    idx = (d - anchor).days % len(pat)
    code = pat[idx]
    return "OFF" if code == "N" and not _night_active_on(staff.unit_id, d) else code


def _cycle_day_for(staff: Staff, d: date) -> int | None:
    """Return the 1-indexed pattern cycle day for `staff` on date `d`."""
    pat, anchor = _pattern_context(staff, d)
    if not pat:
        return None
    return ((d - anchor).days % len(pat)) + 1


def set_assignment(staff: Staff, d: date, code: str, source="auto", note=""):
    a = Assignment.query.filter_by(staff_id=staff.id, day=d).first()
    if a and a.source == "manual":
        return a
    if not a:
        a = Assignment(staff=staff, day=d)
        db.session.add(a)
    a.code, a.source, a.note = code, source, note
    return a


def overwrite_assignment(staff: Staff, d: date, code: str, note: str = ""):
    """Set/replace assignment regardless of existing source (used when regenerating)."""
    a = Assignment.query.filter_by(staff_id=staff.id, day=d).first()
    if not a:
        a = Assignment(staff=staff, day=d)
        db.session.add(a)
    a.code = code
    a.source = "auto"
    a.note = note or a.note
    return a

# Respect manual edits & only clear annotations when auto changes the code


def refresh_day_from_pattern_and_leave(staff: Staff, d: date):
    """
    Recompute a single day based on pattern + leave overlay rules.
    - Preserve explicit sickness (SC/SSC) and TOIL use (TOU8/TOUI).
    - Do NOT clear annotations unless the auto logic changes the code.
    """
    existing = Assignment.query.filter_by(staff_id=staff.id, day=d).first()
    prev_code = existing.code if existing else None

    # Do not touch manual or AI-written cells (leave/sick handled earlier)
    if existing and (existing.code or "").strip() and existing.source in ("manual", "ai"):
        return

    # Keep explicit sickness & TOIL-use exactly as entered.
    sickness_codes = {
        item["code"] for item in get_absence_types(
            "sickness", active_only=False, unit_id=staff.unit_id
        )
    }
    if existing and existing.code in sickness_codes | {"TOU8", "TOUI"}:
        return existing

    pat_code = code_from_pattern(staff, d)
    lv = day_leave_for(staff, d)

    if lv == "AL":
        # AL overlays only on working pattern days
        pat_shift = get_shift(pat_code)
        if pat_shift and pat_shift.is_working:
            a = overwrite_assignment(staff, d, "AL", note="leave")
            a.annotation = ""  # leave days shouldn't carry OT/EXT flags
            return a
        # If pattern is non-working, just write pattern
        a = set_assignment(staff, d, pat_code, source="auto", note="pattern")
        if prev_code is None or (prev_code != a.code and a.source != "manual"):
            a.annotation = ""
        return a

    if lv:
        a = overwrite_assignment(staff, d, lv, note="leave")
        a.annotation = ""
        return a

    # No leave: (re)write pattern but preserve annotations unless code changes
    a = set_assignment(staff, d, pat_code, source="auto", note="pattern")
    if prev_code is None or (prev_code != a.code and a.source != "manual"):
        a.annotation = ""
    return a


def shift_duration_minutes(shift: ShiftType):
    return shift_minutes(shift)


def ensure_month_requirement(year, month, default=(4, 4, 4, 2)):
    r = Requirement.query.filter_by(year=year, month=month).first()
    if not r:
        if len(default) == 3:
            dm, da, dn = default[0], default[1], default[2]
            dd = 0
        else:
            dm, dd, da, dn = default
        r = Requirement(
            year=year, month=month,
            req_m=dm, req_d=dd, req_a=da, req_n=dn,
            req_sat_m=dm, req_sat_d=dd, req_sat_a=da, req_sat_n=dn,
            req_sun_m=dm, req_sun_d=dd, req_sun_a=da, req_sun_n=dn,
        )
        db.session.add(r)
        db.session.commit()
    return r


def requirements_for_day(
    requirement: Requirement | None,
    day: date,
    special: SpecialRequirement | None = None,
) -> dict[str, int]:
    return daily_requirements(requirement, day, special)

# Idempotent month generation that preserves manual entries


def generate_month(year: int, month: int, *args, **kwargs):
    """Ensure rows exist/are correct for the month without touching manual edits."""
    _, days = month_range(year, month)
    for s in Staff.query.order_by(Staff.id):
        for d in days:
            refresh_day_from_pattern_and_leave(s, d)
    db.session.commit()


def _is_working_day_code(code: str) -> bool:
    """
    True for working 'Day' shifts (codes that start with 'D'),
    excluding non-working types like OFF/leave/TOIL/etc.
    Uses ShiftType.is_working when known; otherwise falls back to prefix check.
    """
    c = (code or "").strip().upper()
    if not c:
        return False

    NON_WORKING = {"OFF", "AL", "PL", "SPL", "TOU8", "TOUI",
                   "OSS", "OFFICE", "WFH", "CTB", "MTG"}
    if c in NON_WORKING:
        return False

    try:
        sh = get_shift(c)
    except NameError:
        sh = None
    if sh is None:
        try:
            sh = ShiftType.query.filter_by(code=c).first()
        except Exception:
            sh = None

    if sh is not None:
        return bool(getattr(sh, "is_working", False)) and c.startswith("D")
    return c.startswith("D")


# -------------------- Fatigue helpers (SRATCOH D18–D43; On-Call ignored) --------------------

SYSTEM_FATIGUE_RULES = [
    {"code": "D21", "name": "Duty duration and rolling hours", "severity": "critical", "parameters": {
        "max_duty_hours": {"label": "Maximum single duty", "value": 10, "unit": "hours"},
        "max_rolling_hours": {"label": "Maximum rolling duty", "value": 200, "unit": "hours"},
        "rolling_days": {"label": "Rolling period", "value": 30, "unit": "days"},
    }},
    {"code": "D22", "name": "Minimum rest between duties", "severity": "critical", "parameters": {
        "normal_rest_hours": {"label": "Normal minimum rest", "value": 12, "unit": "hours"},
        "absolute_min_rest_hours": {"label": "Absolute minimum rest", "value": 11, "unit": "hours"},
        "reduced_rest_window_days": {"label": "Reduced-rest review period", "value": 30, "unit": "days"},
    }},
    {"code": "D23", "name": "Recovery after consecutive duties", "severity": "warning", "parameters": {
        "max_consecutive_duties": {"label": "Consecutive-duty trigger", "value": 6, "unit": "duties"},
        "max_consecutive_hours": {"label": "Consecutive-hours trigger", "value": 50, "unit": "hours"},
        "recovery_hours": {"label": "Required recovery", "value": 60, "unit": "hours"},
        "hard_recovery_hours": {"label": "Warning threshold", "value": 54, "unit": "hours"},
    }},
    {"code": "D24", "name": "Qualifying rest in rolling period", "severity": "warning", "parameters": {
        "qualifying_rest_hours": {"label": "Rest needed to qualify", "value": 54, "unit": "hours"},
        "required_rest_hours": {"label": "Total qualifying rest required", "value": 180, "unit": "hours"},
        "rest_window_days": {"label": "Review period", "value": 30, "unit": "days"},
    }},
    {"code": "D30", "name": "Night-duty limits", "severity": "critical", "parameters": {
        "max_night_hours": {"label": "Maximum night duty", "value": 9.5, "unit": "hours"},
        "max_consecutive_nights": {"label": "Maximum consecutive nights", "value": 2, "unit": "nights"},
    }},
    {"code": "D31", "name": "Recovery after night duties", "severity": "warning", "parameters": {
        "single_night_recovery_hours": {"label": "Recovery after one night", "value": 48, "unit": "hours"},
        "night_block_recovery_hours": {"label": "Recovery after two nights", "value": 54, "unit": "hours"},
    }},
    {"code": "D39", "name": "Early-start frequency", "severity": "warning", "parameters": {
        "max_early_starts": {"label": "Maximum early starts", "value": 2, "unit": "starts"},
        "early_window_hours": {"label": "Review period", "value": 144, "unit": "hours"},
    }},
    {"code": "D40", "name": "Early-start duty length", "severity": "warning", "parameters": {
        "max_early_duty_hours": {"label": "Maximum early-start duty", "value": 8, "unit": "hours"},
    }},
    {"code": "D43", "name": "Morning-duty limits", "severity": "warning", "parameters": {
        "max_morning_points": {"label": "Maximum consecutive morning points", "value": 5, "unit": "points"},
        "max_morning_duty_hours": {"label": "Maximum morning duty", "value": 8.5, "unit": "hours"},
    }},
]

CUSTOM_FATIGUE_RULE_TYPES = {
    "max_duty_hours": {
        "label": "Maximum single duty length",
        "unit": "hours", "default": 10, "uses_window": False,
    },
    "min_rest_hours": {
        "label": "Minimum rest between duties",
        "unit": "hours", "default": 11, "uses_window": False,
    },
    "max_consecutive_duties": {
        "label": "Maximum consecutive duties",
        "unit": "duties", "default": 6, "uses_window": False,
    },
    "max_consecutive_nights": {
        "label": "Maximum consecutive night duties",
        "unit": "nights", "default": 2, "uses_window": False,
    },
    "max_early_starts_in_window": {
        "label": "Maximum early starts in a period",
        "unit": "starts", "default": 2, "uses_window": True,
        "default_window": 6,
    },
    "max_hours_in_window": {
        "label": "Maximum duty hours in a period",
        "unit": "hours", "default": 200, "uses_window": True,
        "default_window": 30,
    },
}


def _fatigue_rule_config(unit_id: int | None = None) -> dict:
    resolved_unit_id = int(unit_id or _current_unit_id() or 1)
    system = {
        item["code"]: {
            **item,
            "parameters": {
                key: dict(parameter)
                for key, parameter in item["parameters"].items()
            },
            "enabled": True,
        }
        for item in SYSTEM_FATIGUE_RULES
    }
    custom = []
    definitions = {
        "early_start_before": "06:30",
        "night_period_start": "01:30",
        "night_period_end": "05:30",
    }
    row = RosterSetting.query.filter_by(
        unit_id=resolved_unit_id, key="fatigue_rule_config"
    ).first()
    if row and row.value:
        try:
            saved = json.loads(row.value)
            for key in definitions:
                candidate = str((saved.get("definitions") or {}).get(key) or "")
                try:
                    datetime.strptime(candidate, "%H:%M")
                    definitions[key] = candidate
                except ValueError:
                    pass
            for code, overrides in (saved.get("system") or {}).items():
                if code in system and isinstance(overrides, dict):
                    system[code].update({
                        "name": str(overrides.get("name") or system[code]["name"])[:120],
                        "severity": (
                            overrides.get("severity")
                            if overrides.get("severity") in {"warning", "critical"}
                            else system[code]["severity"]
                        ),
                        "enabled": bool(overrides.get("enabled", True)),
                    })
                    saved_parameters = overrides.get("parameters") or {}
                    for key, parameter in system[code]["parameters"].items():
                        try:
                            value = float(saved_parameters.get(
                                key, parameter["value"]
                            ))
                            if value > 0:
                                parameter["value"] = value
                        except (TypeError, ValueError):
                            pass
            for rule in saved.get("custom") or []:
                if (
                    isinstance(rule, dict)
                    and rule.get("rule_type") in CUSTOM_FATIGUE_RULE_TYPES
                ):
                    custom.append(rule)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return {"system": system, "custom": custom, "definitions": definitions}


def _save_fatigue_rule_config(config: dict) -> None:
    unit_id = _current_unit_id()
    row = RosterSetting.query.filter_by(
        unit_id=unit_id, key="fatigue_rule_config"
    ).first()
    if not row:
        row = RosterSetting(
            unit_id=unit_id, key="fatigue_rule_config"
        )
        db.session.add(row)
    row.value = json.dumps({
        "system": {
            code: {
                "name": item["name"],
                "severity": item["severity"],
                "enabled": item["enabled"],
                "parameters": {
                    key: parameter["value"]
                    for key, parameter in item["parameters"].items()
                },
            }
            for code, item in config["system"].items()
        },
        "custom": config["custom"],
        "definitions": config["definitions"],
    }, sort_keys=True)
    db.session.commit()


def _custom_fatigue_flags(segs: list, rules: list) -> dict:
    flags: dict[date, list[str]] = {}
    ordered = sorted(segs, key=lambda item: item["start"])
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rule_type = rule.get("rule_type")
        code = str(rule.get("code") or "CUSTOM")
        name = str(rule.get("name") or "Custom fatigue rule")
        try:
            threshold = float(rule.get("threshold"))
            window_days = max(1, int(rule.get("window_days") or 1))
        except (TypeError, ValueError):
            continue
        if rule_type == "max_duty_hours":
            for seg in ordered:
                hours = seg["mins"] / 60
                if hours > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {hours:g}h exceeds {threshold:g}h"
                    )
        elif rule_type == "min_rest_hours":
            for previous, current in zip(ordered, ordered[1:]):
                rest = (current["start"] - previous["end"]).total_seconds() / 3600
                if rest < threshold:
                    flags.setdefault(current["day"], []).append(
                        f"{code}: {name} — {rest:g}h is below {threshold:g}h"
                    )
        elif rule_type in {"max_consecutive_duties", "max_consecutive_nights"}:
            streak = 0
            previous_day = None
            for seg in ordered:
                qualifies = (
                    True if rule_type == "max_consecutive_duties"
                    else bool(seg["night"])
                )
                consecutive = (
                    previous_day is not None
                    and (seg["day"] - previous_day).days == 1
                )
                streak = streak + 1 if qualifies and consecutive else (1 if qualifies else 0)
                if streak > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {streak} exceeds {threshold:g}"
                    )
                previous_day = seg["day"] if qualifies else None
        elif rule_type in {"max_hours_in_window", "max_early_starts_in_window"}:
            window = deque()
            running = 0.0
            for seg in ordered:
                value = (
                    seg["mins"] / 60
                    if rule_type == "max_hours_in_window"
                    else (1.0 if seg["early"] else 0.0)
                )
                window.append((seg["start"], value))
                running += value
                while window and (
                    seg["start"] - window[0][0]
                ) > timedelta(days=window_days):
                    running -= window.popleft()[1]
                if running > threshold:
                    flags.setdefault(seg["day"], []).append(
                        f"{code}: {name} — {running:g} exceeds "
                        f"{threshold:g} in {window_days} days"
                    )
    return flags


def _span(d: date, sh: ShiftType):
    if not (sh and sh.start_time and sh.end_time):
        return None, None
    start_dt = datetime.combine(d, sh.start_time)
    end_dt = datetime.combine(d, sh.end_time)
    if sh.end_time <= sh.start_time:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _overlap_window(start_dt: datetime, end_dt: datetime, w_start_h: int, w_start_m: int, w_end_h: int, w_end_m: int) -> int:
    base = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    wnd_start = base.replace(hour=w_start_h, minute=w_start_m)
    wnd_end = base.replace(hour=w_end_h, minute=w_end_m)
    if wnd_end <= wnd_start:
        wnd_end += timedelta(days=1)
    total = 0
    for k in (-1, 0, 1):
        a = max(start_dt, wnd_start + timedelta(days=k))
        b = min(end_dt,  wnd_end + timedelta(days=k))
        if b > a:
            total += int((b - a).total_seconds() // 60)
    return total


def _is_working(sh: ShiftType) -> bool:
    return bool(sh and sh.is_working)


def _configured_time(value: str, fallback: time) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except (TypeError, ValueError):
        return fallback


def _is_night_duty(start_dt: datetime, end_dt: datetime, definitions: dict) -> bool:
    start = _configured_time(definitions.get("night_period_start"), time(1, 30))
    end = _configured_time(definitions.get("night_period_end"), time(5, 30))
    return _overlap_window(
        start_dt, end_dt, start.hour, start.minute, end.hour, end.minute
    ) > 0


def _is_early_start(start_dt: datetime, definitions: dict) -> Tuple[bool, bool]:
    hm = start_dt.time()
    threshold = _configured_time(
        definitions.get("early_start_before"), time(6, 30)
    )
    is_early = time(0, 0) <= hm < threshold
    is_pre0600 = is_early and (hm < time(6, 0))
    return is_early, is_pre0600


def _is_morning_duty(start_dt: datetime) -> bool:
    hm = start_dt.time()
    return time(6, 30) <= hm <= time(7, 59)


def _segments_for_staff(staff: Staff, start_day: date, end_day: date):
    segs = []
    definitions = _fatigue_rule_config(staff.unit_id)["definitions"]
    q = (Assignment.query
         .filter(Assignment.staff_id == staff.id,
                 Assignment.day >= start_day,
                 Assignment.day <= end_day)
         .order_by(Assignment.day.asc()))
    for a in q.all():
        code = (a.code or "").upper()

        # SC/SSC are sickness days – treat as REST for fatigue (do not create duty segments)
        if code in ("SC", "SSC"):
            continue

        sh = get_shift(code) if code else None
        if not _is_working(sh):
            continue

        sdt, edt = _span(a.day, sh)
        if not sdt:
            continue

        night = _is_night_duty(sdt, edt, definitions)
        is_early, is_pre0600 = _is_early_start(sdt, definitions)
        is_morning = _is_morning_duty(sdt)
        segs.append({
            "day": a.day,
            "start": sdt,
            "end": edt,
            "mins": int((edt - sdt).total_seconds() // 60),
            "night": night,
            "early": is_early,
            "early_pre0600": is_pre0600,
            "morning": is_morning,
        })
    return segs


def _analyze_segments(segs, rule_config=None, observation_start=None):
    segs = sorted(segs, key=lambda x: x["start"])
    flags = {}
    if not segs:
        return flags

    config = rule_config or _fatigue_rule_config()
    system = config["system"]

    def parameter(code, name):
        return float(system[code]["parameters"][name]["value"])

    d21_window = timedelta(days=parameter("D21", "rolling_days"))
    d22_window = timedelta(days=parameter("D22", "reduced_rest_window_days"))
    d24_window = timedelta(days=parameter("D24", "rest_window_days"))
    early_window_span = timedelta(hours=parameter("D39", "early_window_hours"))

    win30 = deque()
    duty_30 = 0
    reduced_intervals_30 = deque()
    night_block_count = 0
    last_night_end = None

    consec_queue = deque()

    morning_streak_points = 0
    early_window = deque()
    last_duty_day = None
    last_was_night = False
    last_was_early_pre0600 = False

    prev_end = None

    for seg in segs:
        start = seg["start"]
        end = seg["end"]
        mins = seg["mins"]
        night = seg["night"]
        early = seg["early"]
        early_pre0600 = seg["early_pre0600"]
        morning = seg["morning"]
        the_day = seg["day"]

        if night_block_count > 0 and not night and last_night_end is not None:
            gap = start - last_night_end
            req_hours = (
                parameter("D31", "single_night_recovery_hours")
                if night_block_count == 1
                else parameter("D31", "night_block_recovery_hours")
            )
            if gap < timedelta(hours=req_hours):
                flags.setdefault(the_day, []).append(
                    f"<{req_hours}h after {'single' if night_block_count == 1 else 'two consecutive'} night(s) (D31: {int(gap.total_seconds()//3600)}h)"
                )
            night_block_count = 0
            last_night_end = None

        if prev_end is not None:
            gap = start - prev_end
            while reduced_intervals_30 and (
                start - reduced_intervals_30[0]
            ) > d22_window:
                reduced_intervals_30.popleft()
            normal_rest = parameter("D22", "normal_rest_hours")
            absolute_rest = parameter("D22", "absolute_min_rest_hours")
            if gap < timedelta(hours=normal_rest):
                if gap >= timedelta(hours=absolute_rest):
                    if len(reduced_intervals_30) == 0:
                        reduced_intervals_30.append(start)
                    else:
                        flags.setdefault(the_day, []).append(
                            f"<{normal_rest:g}h between duties (D22) and "
                            f"{absolute_rest:g}–{normal_rest:g}h allowance "
                            f"already used within last "
                            f"{parameter('D22', 'reduced_rest_window_days'):g} days"
                        )
                else:
                    flags.setdefault(the_day, []).append(
                        f"<{absolute_rest:g}h between duties "
                        f"(D22: {int(gap.total_seconds()//3600)}h)"
                    )

        qualifying_rest = parameter("D24", "qualifying_rest_hours")
        required_rest = parameter("D24", "required_rest_hours")
        rest_window_start = start - d24_window
        has_complete_rest_window = (
            observation_start is not None
            and observation_start <= rest_window_start
        )
        qual_hours = 0.0
        if has_complete_rest_window:
            rest_start = rest_window_start
            for prior in segs:
                if prior["start"] >= start:
                    break
                if prior["end"] <= rest_window_start:
                    continue
                duty_start = max(prior["start"], rest_window_start)
                if duty_start > rest_start:
                    rest = duty_start - rest_start
                    if rest >= timedelta(hours=qualifying_rest):
                        qual_hours += rest.total_seconds() / 3600.0
                rest_start = max(rest_start, prior["end"])
            if start > rest_start:
                rest = start - rest_start
                if rest >= timedelta(hours=qualifying_rest):
                    qual_hours += rest.total_seconds() / 3600.0
        if has_complete_rest_window and qual_hours < required_rest:
            flags.setdefault(the_day, []).append(
                f"D24: qualifying rest {int(round(qual_hours))}h "
                f"(<{required_rest:g}h) in last "
                f"{parameter('D24', 'rest_window_days'):g}d"
            )

        prior_consec_count = len(consec_queue)
        prior_consec_minutes = sum(m for (_, _, m) in consec_queue)
        max_consecutive = parameter("D23", "max_consecutive_duties")
        max_consecutive_hours = parameter("D23", "max_consecutive_hours")
        if (
            prior_consec_count >= max_consecutive
            or prior_consec_minutes >= max_consecutive_hours * 60
        ):
            if prev_end is not None:
                gap = start - prev_end
                recovery = parameter("D23", "recovery_hours")
                hard_recovery = parameter("D23", "hard_recovery_hours")
                if gap < timedelta(hours=recovery):
                    if gap < timedelta(hours=hard_recovery):
                        flags.setdefault(the_day, []).append(
                            f"<{recovery:g}h after {max_consecutive:g} "
                            f"consecutive duties or ≥{max_consecutive_hours:g}h "
                            f"across consecutive duties "
                            f"(D23: {int(gap.total_seconds()//3600)}h)"
                        )

        max_duty_hours = parameter("D21", "max_duty_hours")
        if mins > max_duty_hours * 60:
            flags.setdefault(the_day, []).append(
                f"Duty > {max_duty_hours:g}h (D21)"
            )

        while win30 and (end - win30[0][1]) > d21_window:
            _, _, mo = win30.popleft()
            duty_30 -= mo
        win30.append((start, end, mins))
        duty_30 += mins
        max_rolling_hours = parameter("D21", "max_rolling_hours")
        if duty_30 > max_rolling_hours * 60:
            flags.setdefault(the_day, []).append(
                f">{max_rolling_hours:g}h duty in last "
                f"{parameter('D21', 'rolling_days'):g} days (D21)")

        if night:
            max_night_hours = parameter("D30", "max_night_hours")
            if mins > max_night_hours * 60:
                flags.setdefault(the_day, []).append(
                    f"Night duty > {max_night_hours:g}h (D30)"
                )
            if end.time() > time(7, 30):
                flags.setdefault(the_day, []).append(
                    "Night duty ends after 07:30 (D30)")

            if last_duty_day and (the_day - last_duty_day).days == 1 and last_was_night:
                night_block_count += 1
            else:
                night_block_count = 1

            max_nights = parameter("D30", "max_consecutive_nights")
            if night_block_count > max_nights:
                flags.setdefault(the_day, []).append(
                    f"More than {max_nights:g} consecutive night duties (D30)"
                )

            last_night_end = end

        if early:
            early_window.append(start)
            while early_window and (
                start - early_window[0]
            ) > early_window_span:
                early_window.popleft()
            max_early_starts = parameter("D39", "max_early_starts")
            if len(early_window) > max_early_starts:
                flags.setdefault(the_day, []).append(
                    f"More than {max_early_starts:g} early starts in "
                    f"{parameter('D39', 'early_window_hours'):g}h (D39)"
                )
            if early_pre0600 and last_was_early_pre0600 and last_duty_day and (the_day - last_duty_day).days == 1:
                flags.setdefault(the_day, []).append(
                    "Consecutive early starts both before 06:00 not permitted (D39)"
                )
            max_early_hours = parameter("D40", "max_early_duty_hours")
            if mins > max_early_hours * 60:
                flags.setdefault(the_day, []).append(
                    f"Early start duty > {max_early_hours:g}h (D40)"
                )

        if early or morning:
            points_today = 2 if early_pre0600 else 1
            if last_duty_day and (the_day - last_duty_day).days == 1 and (morning_streak_points > 0):
                morning_streak_points += points_today
            else:
                morning_streak_points = points_today
            max_morning_points = parameter("D43", "max_morning_points")
            if morning_streak_points > max_morning_points:
                flags.setdefault(the_day, []).append(
                    f"More than {max_morning_points:g} consecutive "
                    f"morning-duty points (D43)"
                )
        else:
            morning_streak_points = 0

        max_morning_hours = parameter("D43", "max_morning_duty_hours")
        if morning and mins > max_morning_hours * 60:
            flags.setdefault(the_day, []).append(
                f"Morning duty > {max_morning_hours:g}h (D43)"
            )

        if (last_duty_day is None) or ((the_day - last_duty_day).days >= 2):
            consec_queue.clear()
        consec_queue.append((start, end, mins))

        prev_end = end
        last_duty_day = the_day
        last_was_night = night
        last_was_early_pre0600 = early_pre0600

    return flags


def fatigue_flags_for_range(staff: Staff, day_list, lookback_days=30):
    if not day_list:
        return {}
    day_list = sorted(day_list)
    start_lb = day_list[0] - timedelta(days=lookback_days)
    end_day = day_list[-1]
    segs = _segments_for_staff(staff, start_lb, end_day)
    config = _fatigue_rule_config(staff.unit_id)
    all_flags = _analyze_segments(
        segs, config, observation_start=datetime.combine(start_lb, time.min)
    )
    enabled_system = {
        code for code, rule in config["system"].items()
        if rule["enabled"]
    }
    all_flags = {
        finding_day: [
            message for message in messages
            if not (
                match := re.search(r"\b(D\d{2})\b", message)
            ) or match.group(1) in enabled_system
        ]
        for finding_day, messages in all_flags.items()
    }
    for finding_day, messages in _custom_fatigue_flags(
        segs, config["custom"]
    ).items():
        all_flags.setdefault(finding_day, []).extend(messages)
    target_set = set(day_list)
    return {
        d: findings for d, findings in all_flags.items()
        if d in target_set and findings
    }


def roster_fatigue_flags_for_range(
    staff: Staff,
    day_list,
    code_by_day: dict[date, str],
    unit_id: int | None = None,
) -> dict[date, list[str]]:
    """Expose fatigue warnings only on active working-duty roster cells."""
    resolved_unit_id = int(unit_id or staff.unit_id)
    findings = fatigue_flags_for_range(staff, day_list)
    return {
        finding_day: messages
        for finding_day, messages in findings.items()
        if (
            (shift := get_shift(
                code_by_day.get(finding_day), resolved_unit_id
            ))
            and shift.is_active
            and shift.is_working
        )
    }


def would_trigger_fatigue(staff: Staff, day: date, code: str):
    sh = get_shift(code)
    if not _is_working(sh):
        return []
    start_lb = day - timedelta(days=30)
    end_day = day
    segs = _segments_for_staff(staff, start_lb, end_day)
    definitions = _fatigue_rule_config(staff.unit_id)["definitions"]
    sdt, edt = _span(day, sh)
    if sdt:
        is_early, is_pre0600 = _is_early_start(sdt, definitions)
        segs.append({
            "day": day, "start": sdt, "end": edt,
            "mins": int((edt - sdt).total_seconds() // 60),
            "night": _is_night_duty(sdt, edt, definitions),
            "early": is_early,
            "early_pre0600": is_pre0600,
            "morning": _is_morning_duty(sdt),
        })
    flags = _analyze_segments(
        segs, observation_start=datetime.combine(start_lb, time.min)
    )
    return flags.get(day, [])


def _year_month_iter(start_date: date, end_date: date):
    yield from iter_year_months(start_date, end_date)


def generate_range(start_day: date, end_day: date):
    """
    Ensure requirements and (re)build each month from start_day's month through
    end_day's month (inclusive). Safe to re-run; respects manual/protected codes.
    """
    for y, m in _year_month_iter(start_day, end_day):
        ensure_month_requirement(y, m)
        generate_month(y, m)


def ensure_assignments_for_range(start_day: date, end_day: date):
    for y, m in _year_month_iter(start_day, end_day):
        ensure_month_requirement(y, m)
        generate_month(y, m)


def would_create_new_fatigue_issues(
    staff: Staff,
    proposed_day: date,
    proposed_code: str,
    lookback_days: int = 30,
    lookahead_days: int = 14,
):
    sh = get_shift(proposed_code)
    if not _is_working(sh):
        return {}
    start = proposed_day - timedelta(days=lookback_days)
    end = proposed_day + timedelta(days=lookahead_days)
    segs_base = _segments_for_staff(staff, start, end)
    observation_start = datetime.combine(start, time.min)
    flags_base = _analyze_segments(
        segs_base, observation_start=observation_start
    )
    sdt, edt = _span(proposed_day, sh)
    if not sdt:
        return {}
    definitions = _fatigue_rule_config(staff.unit_id)["definitions"]
    is_early, is_pre0600 = _is_early_start(sdt, definitions)
    segs_prop = list(segs_base)
    segs_prop.append({
        "day": proposed_day,
        "start": sdt,
        "end": edt,
        "mins": int((edt - sdt).total_seconds() // 60),
        "night": _is_night_duty(sdt, edt, definitions),
        "early": is_early,
        "early_pre0600": is_pre0600,
        "morning": _is_morning_duty(sdt),
    })
    flags_prop = _analyze_segments(
        segs_prop, observation_start=observation_start
    )
    new_flags = {}
    for d, lst in flags_prop.items():
        if d < proposed_day:
            continue
        base_set = set(flags_base.get(d, []))
        diff = sorted(set(lst) - base_set)
        if diff:
            new_flags[d] = diff
    return new_flags


def _compliance_month(ym: str | None) -> tuple[int, int]:
    today = date.today()
    value = (ym or f"{today.year:04d}-{today.month:02d}").strip()
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        abort(400, "Month must use YYYY-MM.")
    year, month = map(int, value.split("-"))
    if month not in range(1, 13):
        abort(400, "Invalid month.")
    return year, month


def _compliance_findings(year: int, month: int) -> dict:
    _, days = month_range(year, month)
    people = (
        Staff.query.filter_by(is_operational=True)
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .order_by(Watch.order_index, Staff.name)
        .all()
    )
    rows = []
    rule_counts: Counter[str] = Counter()
    rule_config = _fatigue_rule_config()
    rule_metadata = {
        code: item for code, item in rule_config["system"].items()
    }
    rule_metadata.update({
        str(item.get("code")): item
        for item in rule_config["custom"]
    })
    for person in people:
        flags = fatigue_flags_for_range(person, days)
        issues = []
        for finding_day, messages in sorted(flags.items()):
            assignment = Assignment.query.filter_by(
                staff_id=person.id, day=finding_day
            ).first()
            for message in messages:
                code_match = re.search(r"\b(D\d{2}|USR-[A-F0-9]+)\b", message)
                code = code_match.group(1) if code_match else ""
                metadata = rule_metadata.get(code, {})
                rule = str(
                    metadata.get("name")
                    or message.split(":", 1)[0].split("(", 1)[0].strip()
                )
                severity = metadata.get("severity")
                if severity not in {"warning", "critical"}:
                    severity = "critical" if any(
                        token in message
                        for token in ("<11h", "3rd consecutive", ">200h", "> 10h")
                    ) else "warning"
                rule_counts[f"{code} · {rule}" if code else rule] += 1
                issues.append({
                    "day": finding_day,
                    "message": message,
                    "rule": rule,
                    "rule_code": code,
                    "severity": severity,
                    "assignment": assignment,
                })
        rows.append({"staff": person, "issues": issues, "total": len(issues)})
    total = sum(row["total"] for row in rows)
    return {
        "days": days,
        "rows": rows,
        "total": total,
        "affected": sum(1 for row in rows if row["total"]),
        "critical": sum(
            1 for row in rows for issue in row["issues"]
            if issue["severity"] == "critical"
        ),
        "rule_counts": rule_counts.most_common(),
    }


@app.route("/compliance-centre")
@login_required
def compliance_centre():
    """Retired standalone view; roster cells remain the monitoring surface."""
    if not is_admin_user(current_user):
        abort(403)
    year, month = _compliance_month(request.args.get("ym"))
    return redirect(
        url_for("roster_month", ym=f"{year:04d}-{month:02d}")
    )


@app.route("/admin/fatigue-rules", methods=["GET", "POST"])
@login_required
def admin_fatigue_rules():
    if not is_admin_user(current_user):
        abort(403)
    config = _fatigue_rule_config()
    if request.method == "POST":
        _validate_csrf()
        action = request.form.get("action") or ""
        try:
            if action == "update_definitions":
                definitions = {}
                for key in (
                    "early_start_before",
                    "night_period_start",
                    "night_period_end",
                ):
                    value = (request.form.get(key) or "").strip()
                    datetime.strptime(value, "%H:%M")
                    definitions[key] = value
                if definitions["night_period_start"] == definitions["night_period_end"]:
                    raise ValueError("The night-duty period must have a duration.")
                config["definitions"] = definitions
                _save_fatigue_rule_config(config)
                flash("Duty time definitions updated for this airport.", "ok")
            elif action == "update_system":
                code = (request.form.get("code") or "").upper()
                if code not in config["system"]:
                    abort(404)
                item = config["system"][code]
                item["name"] = (
                    request.form.get("name") or item["name"]
                ).strip()[:120]
                item["severity"] = (
                    request.form.get("severity")
                    if request.form.get("severity") in {"warning", "critical"}
                    else item["severity"]
                )
                item["enabled"] = request.form.get("enabled") == "on"
                for key, parameter_item in item["parameters"].items():
                    value = float(request.form.get(
                        f"parameter_{key}", parameter_item["value"]
                    ))
                    if not 0 < value <= 10000:
                        raise ValueError(
                            f"{parameter_item['label']} must be greater "
                            "than zero."
                        )
                    parameter_item["value"] = value
                _save_fatigue_rule_config(config)
                flash(f"{code} fatigue rule updated.", "ok")
            elif action in {"add_custom", "update_custom"}:
                rule_type = request.form.get("rule_type") or ""
                if rule_type not in CUSTOM_FATIGUE_RULE_TYPES:
                    raise ValueError("Choose a supported rule check.")
                name = (request.form.get("name") or "").strip()
                if len(name) < 3:
                    raise ValueError("Give the rule a clear name.")
                threshold = float(request.form.get("threshold") or 0)
                if threshold <= 0:
                    raise ValueError("The limit must be greater than zero.")
                type_meta = CUSTOM_FATIGUE_RULE_TYPES[rule_type]
                window_days = int(
                    request.form.get("window_days")
                    or type_meta.get("default_window", 1)
                )
                if not 1 <= window_days <= 365:
                    raise ValueError("The review period must be 1–365 days.")
                severity = request.form.get("severity")
                if severity not in {"warning", "critical"}:
                    severity = "warning"
                code = (request.form.get("code") or "").upper()
                existing = next((
                    item for item in config["custom"]
                    if item.get("code") == code
                ), None)
                if action == "update_custom" and not existing:
                    abort(404)
                if not existing:
                    existing = {
                        "code": f"USR-{secrets.token_hex(3).upper()}"
                    }
                    config["custom"].append(existing)
                existing.update({
                    "name": name[:120],
                    "rule_type": rule_type,
                    "threshold": threshold,
                    "window_days": window_days,
                    "severity": severity,
                    "enabled": request.form.get("enabled") == "on",
                })
                _save_fatigue_rule_config(config)
                flash(f"{existing['code']} fatigue rule saved.", "ok")
            elif action == "delete_custom":
                code = (request.form.get("code") or "").upper()
                before = len(config["custom"])
                config["custom"] = [
                    item for item in config["custom"]
                    if item.get("code") != code
                ]
                if len(config["custom"]) == before:
                    abort(404)
                _save_fatigue_rule_config(config)
                flash(f"{code} custom fatigue rule removed.", "ok")
            else:
                abort(400)
        except (TypeError, ValueError) as exc:
            flash(str(exc), "error")
        return redirect(url_for("admin_fatigue_rules"))
    return render_template(
        "admin_fatigue_rules.html",
        system_rules=list(config["system"].values()),
        custom_rules=config["custom"],
        definitions=config["definitions"],
        rule_types=CUSTOM_FATIGUE_RULE_TYPES,
        current_unit=db.session.get(Unit, _current_unit_id()),
    )


@app.route("/compliance-centre/export")
@login_required
def compliance_centre_export():
    """Retired with the standalone Compliance Centre."""
    if not is_admin_user(current_user):
        abort(403)
    year, month = _compliance_month(request.args.get("ym"))
    return redirect(
        url_for("roster_month", ym=f"{year:04d}-{month:02d}")
    )

# -------------------- Migrations / seeding --------------------


def migrate_tenant_foundation_compat():
    """Idempotently upgrade legacy SQLite desktops before normal startup."""
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if "unit" not in inspector.get_table_names():
        db.create_all()
        inspector = inspect(db.engine)
    if Unit.query.count() == 0:
        db.session.add(Unit(id=1, code="FIRST", name="First airport unit"))
        db.session.commit()
    additions = {
        "staff": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "membership_status": "VARCHAR(20) NOT NULL DEFAULT 'active'",
            "permissions_json": "TEXT NOT NULL DEFAULT '{}'",
        },
        "watch": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "requirement": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "leave": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "sickness": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "ai_rule_set": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "change_log": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "staff_watch_history": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "shift_type": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
            "is_requestable": "BOOLEAN NOT NULL DEFAULT 0",
            "required_qualification": "VARCHAR(40) NOT NULL DEFAULT ''",
        },
        "assignment": {"unit_id": "INTEGER NOT NULL DEFAULT 1"},
        "shift_request": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "requester_comment": "VARCHAR(500) NOT NULL DEFAULT ''",
            "created_at": "DATETIME",
            "updated_at": "DATETIME",
            "fulfilled_at": "DATETIME",
            "cancelled_at": "DATETIME",
            "resulting_assignment_id": "INTEGER",
        },
        "annotation_type": {
            "unit_id": "INTEGER NOT NULL DEFAULT 1",
            "colour": "VARCHAR(20) NOT NULL DEFAULT '#6c757d'",
            "description": "TEXT NOT NULL DEFAULT ''",
            "note_required": "BOOLEAN NOT NULL DEFAULT 0",
            "admin_only": "BOOLEAN NOT NULL DEFAULT 0",
            "has_been_used": "BOOLEAN NOT NULL DEFAULT 0",
        },
    }
    for table_name, columns in additions.items():
        if table_name not in inspector.get_table_names():
            continue
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        for name, ddl in columns.items():
            if name not in existing:
                db.session.execute(text(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{name}" {ddl}'
                ))
        db.session.execute(text(
            f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_unit_id" '
            f'ON "{table_name}" ("unit_id")'
        ))
    db.session.execute(text(
        "UPDATE shift_request SET created_at = COALESCE(created_at, submitted_at), "
        "updated_at = COALESCE(updated_at, submitted_at)"
    ))
    db.session.commit()


def migrate_add_perf_indexes():
    from sqlalchemy import text
    try:
        # Speeds leave/sickness/range scans
        db.session.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_day_code ON assignment(day, code)"))
        # Already have unique (staff_id, day); this helps pure day scans by staff
        db.session.execute(
            text("CREATE INDEX IF NOT EXISTS ix_assignment_day ON assignment(day)"))
        # Shift requests pages group by day a lot
        db.session.execute(
            text("CREATE INDEX IF NOT EXISTS ix_shift_request_day ON shift_request(day)"))
        db.session.commit()
    except Exception:
        db.session.rollback()


def migrate_add_role_and_calendar_token():
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(staff)"))]
        if "role" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN role VARCHAR(10) DEFAULT 'user'"))
            except Exception:
                pass
        if "calendar_token" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN calendar_token VARCHAR(64)"))
            except Exception:
                pass
        if "email" not in cols:
            try:
                conn.execute(text(
                    "ALTER TABLE staff ADD COLUMN email VARCHAR(254) "
                    "NOT NULL DEFAULT ''"
                ))
            except Exception:
                pass
        try:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_staff_calendar_token ON staff (calendar_token)"))
        except Exception:
            pass

    changed = False
    for u in Staff.query.all():
        if not u.role or u.role not in ("superadmin", "admin", "editor", "user"):
            u.role = "admin" if getattr(u, "is_admin", False) else "user"
            changed = True
        if not u.calendar_token:
            u.calendar_token = secrets.token_hex(16)
            changed = True
    if changed:
        db.session.commit()


def migrate_add_assignment_annotation():
    from sqlalchemy import text
    try:
        db.session.execute(
            text("ALTER TABLE assignment ADD COLUMN annotation VARCHAR(20)"))
        db.session.commit()
    except Exception:
        db.session.rollback()


def migrate_add_unique_assignment_key():
    from sqlalchemy import text
    try:
        db.session.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_assignment_staff_day ON assignment(staff_id, day)"))
        db.session.commit()
    except Exception:
        db.session.rollback()


def migrate_add_perf_indexes():
    """Create helpful indexes if missing (SQLite: IF NOT EXISTS is supported)."""
    from sqlalchemy import text
    with app.app_context():
        stmts = [
            "CREATE INDEX IF NOT EXISTS ix_assignment_day ON assignment(day)",
            "CREATE INDEX IF NOT EXISTS ix_assignment_staff_day ON assignment(staff_id, day)",
            "CREATE INDEX IF NOT EXISTS ix_requirement_ym ON requirement(year, month)"
        ]
        for s in stmts:
            db.session.execute(text(s))
        db.session.commit()


def migrate_add_requirement_req_d():
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            text("PRAGMA table_info(requirement)"))]
        if "req_d" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE requirement ADD COLUMN req_d INTEGER DEFAULT 0"))
            except Exception:
                pass
    db.session.commit()


def migrate_add_ut_flags():
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(staff)"))]
        if "tower_ut" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN tower_ut BOOLEAN DEFAULT 0"))
            except Exception:
                pass
        if "radar_ut" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN radar_ut BOOLEAN DEFAULT 0"))
            except Exception:
                pass
    db.session.commit()


def migrate_add_is_training():
    """Add is_training to shift_type if missing."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(shift_type)"))]
        if "is_training" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE shift_type ADD COLUMN is_training BOOLEAN DEFAULT 0"))
            except Exception:
                pass
    db.session.commit()


def migrate_add_wm_dwm_exclude():
    """Add is_wm, is_dwm, exclude_from_ot to staff if missing."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(staff)"))]
        if "is_wm" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN is_wm BOOLEAN DEFAULT 0"))
            except Exception:
                pass
        if "is_dwm" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN is_dwm BOOLEAN DEFAULT 0"))
            except Exception:
                pass
        if "exclude_from_ot" not in cols:
            try:
                conn.execute(
                    text("ALTER TABLE staff ADD COLUMN exclude_from_ot BOOLEAN DEFAULT 0"))
            except Exception:
                pass
    db.session.commit()


def migrate_add_phone_number():
    """Add phone_number column for SMS notifications if missing."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(staff)"))]
        if "phone_number" not in cols:
            try:
                conn.execute(text(
                    "ALTER TABLE staff ADD COLUMN phone_number VARCHAR(30) DEFAULT ''"))
            except Exception:
                pass
    db.session.commit()


def migrate_add_watch_pattern_configuration():
    """Add inherited roster-pattern fields to legacy SQLite databases."""
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    watch_columns = {
        column["name"] for column in inspector.get_columns("watch")
    }
    staff_columns = {
        column["name"] for column in inspector.get_columns("staff")
    }
    statements = []
    if "pattern_csv" not in watch_columns:
        statements.append(
            "ALTER TABLE watch ADD COLUMN pattern_csv VARCHAR(500) "
            "NOT NULL DEFAULT ''"
        )
    if "pattern_anchor" not in watch_columns:
        statements.append(
            "ALTER TABLE watch ADD COLUMN pattern_anchor DATE"
        )
    if "pattern_override" not in staff_columns:
        statements.append(
            "ALTER TABLE staff ADD COLUMN pattern_override BOOLEAN "
            "NOT NULL DEFAULT 0"
        )
    for statement in statements:
        db.session.execute(text(statement))
    if "pattern_override" not in staff_columns:
        # Preserve existing explicitly configured staff patterns.
        db.session.execute(text(
            "UPDATE staff SET pattern_override=1 "
            "WHERE COALESCE(pattern_csv, '') <> ''"
        ))
    db.session.commit()


def migrate_add_invitation_target():
    """Add targeted roster-person invitations to legacy local databases."""
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if "secure_invitation" not in inspector.get_table_names():
        return
    columns = {
        column["name"]
        for column in inspector.get_columns("secure_invitation")
    }
    if "target_person_id" not in columns:
        db.session.execute(text(
            "ALTER TABLE secure_invitation "
            "ADD COLUMN target_person_id INTEGER"
        ))
        db.session.execute(text(
            "CREATE INDEX IF NOT EXISTS "
            "ix_secure_invitation_target_person_id "
            "ON secure_invitation(target_person_id)"
        ))
        db.session.commit()


def migrate_add_toil_half_days_and_convert():
    """Add toil_half_days; add leave-year columns; convert legacy toil_minutes -> half-days if present."""
    from sqlalchemy import text
    with db.engine.connect() as conn:
        cols = [row[1]
                for row in conn.execute(text("PRAGMA table_info(staff)"))]

        def addcol(name, ddl):
            if name not in cols:
                try:
                    conn.execute(text(f"ALTER TABLE staff ADD COLUMN {ddl}"))
                except Exception:
                    pass

        addcol("toil_half_days",         "toil_half_days INTEGER DEFAULT 0")
        addcol("leave_year_start_month",
               "leave_year_start_month INTEGER DEFAULT 4")
        addcol("leave_entitlement_days",
               "leave_entitlement_days INTEGER DEFAULT 0")
        addcol("leave_public_holidays",
               "leave_public_holidays INTEGER DEFAULT 0")
        addcol("leave_carryover_days",
               "leave_carryover_days INTEGER DEFAULT 0")

        # Convert legacy toil_minutes → toil_half_days (240 min = 0.5 day)
        if "toil_minutes" in cols:
            try:
                res = conn.execute(
                    text("SELECT id, COALESCE(toil_minutes,0) FROM staff"))
                rows = res.fetchall()
                for sid, mins in rows:
                    half = int(round(mins / 240.0))
                    conn.execute(text("UPDATE staff SET toil_half_days=:half WHERE id=:sid"),
                                 {"half": half, "sid": sid})
            except Exception:
                pass
    db.session.commit()


def ensure_shift(code, name, start=None, end=None, is_working=False, is_training=False):
    sh = ShiftType.query.filter_by(code=code).first()
    if not sh:
        sh = ShiftType(code=code, name=name, start_time=start, end_time=end,
                       is_working=is_working, is_training=is_training)
        db.session.add(sh)
        db.session.commit()
    return sh


def ensure_watch(name: str, order_index: int):
    w = Watch.query.filter_by(name=name).first()
    if not w:
        w = Watch(name=name, order_index=order_index)
        db.session.add(w)
        db.session.commit()
    return w


def seed_once():
    # A deliberately bootstrapped platform starts without operational units.
    # Do not populate the platform-control tenant with legacy desktop demo data;
    # the Super Admin will create the first airport through the normal workflow.
    if Unit.query.filter_by(status="platform_control").first():
        return
    if Watch.query.count() > 0:
        # make sure TOU* & OSS exist if DB already seeded
        ensure_shift("TOUI", "TOIL (UI)", is_working=False)
        ensure_shift("TOU8", "TOIL (U8)", is_working=False)
        ensure_shift("OSS",  "Operational Support", is_working=False)
        return

    watches = []
    for idx, letter in enumerate(["A", "B", "C", "D", "E"], start=1):
        watches.append(Watch(name=f"Watch {letter}", order_index=idx))
    watches.append(Watch(name="Watch NOPS", order_index=6))
    db.session.add_all(watches)

    db.session.add_all([
        ShiftType(code="M",   name="Morning",     start_time=time(
            6, 0),  end_time=time(14, 0), is_working=True, is_requestable=True),
        ShiftType(code="D",   name="Day",         start_time=time(
            8, 0),  end_time=time(16, 0), is_working=True, is_requestable=True),
        ShiftType(code="A",   name="Afternoon",   start_time=time(
            14, 0), end_time=time(22, 0), is_working=True, is_requestable=True),
        ShiftType(code="N",   name="Night",       start_time=time(
            22, 0), end_time=time(6, 0),  is_working=True, is_requestable=True),
        ShiftType(code="OFF", name="Rest Day",    is_working=False),
        ShiftType(code="AL",  name="Annual Leave",    is_working=False),
        ShiftType(code="PL",  name="Parental Leave",  is_working=False),
        ShiftType(code="SPL", name="Special Leave",   is_working=False),
        # Sickness as training-type working (excluded from counters but shown)
        ShiftType(code="SC",  name="Sick Cert",       start_time=time(
            9, 0), end_time=time(17, 0), is_working=True, is_training=True),
        ShiftType(code="SSC", name="Sick Self Cert",  start_time=time(
            9, 0), end_time=time(17, 0), is_working=True, is_training=True),
        ShiftType(code="SBY", name="Standby",         start_time=time(
            8, 0), end_time=time(16, 0), is_working=True),
        ShiftType(code="TOUI", name="TOIL (UI)", is_working=False),
        ShiftType(code="TOU8", name="TOIL (U8)", is_working=False),
        ShiftType(code="OSS",  name="Operational Support", is_working=False),
        ShiftType(code="OFFICE", name="Office", is_working=False),
        ShiftType(code="WFH",    name="Work from home", is_working=False),
        ShiftType(code="MTG",    name="Meeting", is_working=False),
    ])

    watch_cycle_days = {"A": 6, "B": 4, "C": 2, "D": 10, "E": 8}
    anchor_date = date(2025, 9, 1)

    demo_names = [
        ["Alex McLean", "Bethany Kerr", "Callum Reid", "Donna Fraser", "Euan Boyd"],
        ["Fiona Watt", "Gordon Bryce", "Harris Quinn",
            "Isla Morton", "Jamie Lindsay"],
        ["Kara Drummond", "Lewis Pratt", "Maya Allan", "Noah Cairns", "Orla McAdam"],
        ["Poppy Neill", "Quinn Murray", "Robbie Hogg", "Sophie Duff", "Tommy Craig"],
        ["Una McKay", "Viktor Shaw", "Will Findlay", "Xander Kerr", "Yasmin Doyle"],
    ]

    staff = []
    staff_no = 2001
    for wi, w in enumerate(watches):
        label = w.name.replace("Watch ", "")
        if label == "NOPS":
            continue
        for nm in demo_names[wi]:
            username = "admin" if staff_no == 2001 else f"user{staff_no}"
            s = Staff(
                username=username,
                name=nm,
                staff_no=str(staff_no),
                watch=w,
                is_operational=True,
                has_ojti=((staff_no % 3) == 0),
                is_trainee=((staff_no % 7) == 0),
                role=("admin" if staff_no == 2001 else "user"),
                leave_year_start_month=4,
                leave_entitlement_days=25,
                leave_public_holidays=8,
                leave_carryover_days=0,
            )
            s.set_password("password")
            cycle_day = watch_cycle_days[label]
            offset = cycle_day - 1
            s.pattern_anchor = anchor_date - timedelta(days=offset)
            staff.append(s)
            staff_no += 1

    db.session.add_all(staff)
    db.session.commit()

# -------------------- Small parse & AI helpers --------------------


def _is_empty_like(val) -> bool:
    """Treat '', '-', and em-dash as empty cells the AI may fill."""
    return str(val or "").strip() in {"", "-", "—"}


def _allocate_days_for_date(
    d: date,
    req,
    staff: list,                     # list[Staff]
    by_staff_day: dict,              # dict[int, dict[date, Assignment]]
    day_code_mon_sat: str,
    day_code_sun: str,
) -> int:
    """
    Fill Day shifts (D*) for a single date d to meet req.req_d.
    Respects leave/sick, OFF/manual-protected cells, and fatigue.
    Returns number of assignments created/changed.
    """
    rows_today = Assignment.query.filter_by(day=d).all()
    haveD = sum(1 for a in rows_today if _is_working_day_code((a.code or "")))
    needD = getattr(req, "req_d", 0) if req else 0
    short = max(0, needD - haveD)
    if short <= 0:
        return 0

    changes = 0
    dow = d.weekday()  # 0=Mon .. 6=Sun
    day_code = day_code_mon_sat if dow < 6 else day_code_sun

    for s in staff:
        if short <= 0:
            break

        # leave/sick guard
        if _has_leave_or_sick(s.id, d):
            continue

        a = by_staff_day[s.id].get(d)
        current_code = (a.code if a else "")

        # only fill if truly empty-like or explicitly OFF
        # only fill if truly empty-like (do NOT replace OFF)
        if not _is_empty_like(current_code):
            continue

        # fatigue gate
        if not _passes_fatigue_for(s, d, day_code):
            continue

        # ensure an Assignment row exists
        if a is None:
            a = Assignment(staff_id=s.id, day=d)
            db.session.add(a)
            by_staff_day[s.id][d] = a

        _set_code(a, day_code, source="ai", note="AI fill D")
        changes += 1
        short -= 1

    return changes


def _parse_hhmm(val: str):
    val = (val or "").strip()
    if not val:
        return None
    try:
        hh, mm = val.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return None


def _parse_date(val: str):
    val = (val or "").strip()
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except Exception:
        return None


def _normalise_phone_number(val: str | None) -> str:
    """Tidy phone numbers for SMS sending (keep digits and leading +)."""
    if not val:
        return ""
    cleaned = re.sub(r"[^0-9+]+", "", val.strip())
    if cleaned.startswith("00") and not cleaned.startswith("000"):
        cleaned = "+" + cleaned[2:]
    return cleaned


def parse_annotation(s: str):
    """Return {'type':'A6','suffix':'M'} or {'type':'EXTL'}, else None."""
    if not s:
        return None
    value = s.strip().upper()
    info = get_annotation_config(value)
    if info:
        return {"type": info["code"], "suffix": None}

    snap = _annotation_snapshot(int(_current_unit_id() or 1))["items"]
    for item in snap:
        if not item["allow_suffix"]:
            continue
        code = item["code"]
        if not code:
            continue
        if value.startswith(code) and len(value) == len(code) + 1:
            suffix = value[len(code):]
            if suffix and suffix in set(item["suffixes"]):
                return {"type": code, "suffix": suffix}
    return None


def _context_month_for_date(d: date | None) -> str | None:
    return None if not d else f"{d.year:04d}-{d.month:02d}"


def log_change(entity_type: str, entity_id: int, field: str, old, new, note: str = "", context_day: date | None = None):
    try:
        entry = ChangeLog(
            when=utcnow(),
            who_user_id=getattr(current_user, "id", None),
            entity_type=entity_type,
            entity_id=entity_id,
            field=field,
            old_value=str(old) if old is not None else None,
            new_value=str(new) if new is not None else None,
            context_month=_context_month_for_date(context_day),
            note=note or ""
        )
        db.session.add(entry)
        db.session.commit()
    except Exception:
        db.session.rollback()

# --- Month math (no dateutil) ---


def _month_add(y: int, m: int, delta: int) -> Tuple[int, int]:
    return add_months(y, m, delta)


def lock_date_for_month(y: int, m: int) -> date:
    return roster_lock_date(y, m)


def is_month_locked(y: int, m: int, today: Optional[date] = None) -> bool:
    return roster_month_is_locked(y, m, today)


# Source protection: we never overwrite these
LOCKED_SOURCES = {"manual", "leave", "sickness"}

def _assignment(staff_id: int, d: date) -> "Assignment":
    a = Assignment.query.filter_by(staff_id=staff_id, day=d).first()
    if not a:
        a = Assignment(staff_id=staff_id, day=d)
        db.session.add(a)
    return a


def _cell_is_protected(a: "Assignment") -> bool:
    return (a.code and (a.source in LOCKED_SOURCES))


def _set_code(a: "Assignment", code: str, source: str, note: str = "", ctx_month: Optional[str] = None):
    old = a.code
    if old == code and a.source == source:
        return a

    a.code = code
    a.annotation = None
    a.source = source

    # Invalidate month cache for this day
    _invalidate_month_cache_for_day(a.day)

    try:
        # log using day; function computes month string internally
        log_change("Assignment", a.id, "code", old,
                   code, note=note, context_day=a.day)
    except Exception:
        # don’t break generator if logging fails
        pass

    return a


def _has_leave_or_sick(staff_id: int, d: date) -> bool:
    return bool(
        Leave.query.filter(Leave.staff_id == staff_id, Leave.start <= d, Leave.end >= d).first() or
        Sickness.query.filter(Sickness.staff_id == staff_id,
                              Sickness.start <= d, Sickness.end >= d).first()
    )


def _fatigue_ok(staff: "Staff", day: date, code: str) -> bool:
    """True if assigning `code` on `day` would NOT create new fatigue flags."""
    try:
        flags = would_trigger_fatigue(staff, day, code)
    except Exception:
        # If the analysis fails for any reason, be safe and block the assignment
        return False
    return len(flags) == 0

# Back-compat shim so all AI code can call the same name


def _passes_fatigue_for(staff: "Staff", day: date, code: str) -> bool:
    return _fatigue_ok(staff, day, code)


def _weekday_is_sun(d: date) -> bool:
    return d.weekday() == 6  # Monday=0 ... Sunday=6

# ---------- Shift code helpers ----------


def _normalize_code(code) -> str:
    return str(code or "").strip().upper()


def _is_non_working(code: str) -> bool:
    return _normalize_code(code) in get_non_working_codes()


def _is_working_code_prefix(code: str, prefix: str) -> bool:
    """
    True if:
      - code is not in the non-working list
      - ShiftType says it's working (when known)
      - AND the normalized code startswith the given prefix
    Falls back to just the prefix check if ShiftType is unknown.
    """
    cu = _normalize_code(code)
    if not cu or cu in get_non_working_codes():
        return False

    # Prefer cached lookup if present in your app
    try:
        sh = get_shift(cu)
    except NameError:
        sh = None
    if sh is None:
        try:
            sh = ShiftType.query.filter_by(code=cu).first()
        except Exception:
            sh = None

    if sh is not None:
        return bool(getattr(sh, "is_working", False)) and cu.startswith(prefix)

    return cu.startswith(prefix)


def _is_working_day_code(code: str) -> bool:
    return _is_working_code_prefix(code, "D")


def _is_working_m_code(code: str) -> bool:
    return _is_working_code_prefix(code, "M")


def _is_working_n_code(code: str) -> bool:
    return _is_working_code_prefix(code, "N")

def is_admin_user(u) -> bool:
    return bool(getattr(u, "is_admin", False) or getattr(u, "role", "") == "admin")


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not is_admin_user(current_user):
            abort(403)
        return f(*args, **kwargs)
    return wrapper

# -------------------- Password management --------------------


@app.route("/password", methods=["GET", "POST"])
@login_required
def password_change():
    """Allow any logged-in user to reset OWN password."""
    if request.method == "POST":
        _validate_csrf()
        cur = request.form.get("current_password", "")
        new1 = request.form.get("new_password", "")
        new2 = request.form.get("confirm_password", "")
        if not current_user.check_password(cur):
            flash("Current password is incorrect.", "error")
            return redirect(
                url_for("password_change")
                if current_user.role == "superadmin"
                else url_for("staff_profile", sid=current_user.id) + "#security"
            )
        if not new1 or new1 != new2:
            flash("New passwords do not match.", "error")
            return redirect(
                url_for("password_change")
                if current_user.role == "superadmin"
                else url_for("staff_profile", sid=current_user.id) + "#security"
            )
        if len(new1) < 12:
            flash("Use a password of at least 12 characters.", "error")
            return redirect(
                url_for("password_change")
                if current_user.role == "superadmin"
                else url_for("staff_profile", sid=current_user.id) + "#security"
            )
        if new1 == cur:
            flash("Choose a password different from the current password.", "error")
            return redirect(
                url_for("password_change")
                if current_user.role == "superadmin"
                else url_for("staff_profile", sid=current_user.id) + "#security"
            )
        if current_user.role == "superadmin":
            new_hash = generate_password_hash(new1)
            identity = PlatformIdentity.query.filter_by(
                username=current_user.username
            ).first_or_404()
            identity.password_hash = new_hash
            db.session.commit()
        else:
            u = tenant_get(Staff, current_user.id)
            if not u:
                abort(404)
            u.set_password(new1)
            identity = PlatformIdentity.query.filter_by(
                username=u.username
            ).first()
            if identity:
                identity.password_hash = u.password_hash
            db.session.commit()
        flash("Password updated.", "ok")
        return redirect(
            url_for("platform_admin")
            if current_user.role == "superadmin"
            else url_for("staff_profile", sid=current_user.id) + "#security"
        )
    return render_template("password.html")

# -------------------- Main / Roster --------------------


@app.route("/")
@login_required
def index():
    if is_admin_user(current_user):
        unit = db.session.get(Unit, _current_unit_id())
        if unit and int(unit.onboarding_step or 0) < 100:
            return redirect(url_for("unit_onboarding"))
    t = date.today()
    return redirect(url_for("roster_month", ym=f"{t.year}-{t.month:02d}"))


def _roster_snapshot(year: int, month: int) -> dict:
    start = date(year, month, 1)
    ny, nm = _month_add(year, month, 1)
    end = date(ny, nm, 1)
    assignments = (
        Assignment.query.filter(
            Assignment.unit_id == _current_unit_id(),
            Assignment.day >= start, Assignment.day < end
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
        published_rows = json.loads(
            publication.snapshot_json or "{}"
        ).get("assignments", [])
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
                shift and shift.is_working and shift.required_qualification
                and not _staff_has_shift_qualification(person, shift, day)
            ):
                qualification_gaps.append({
                    "staff": person, "day": day, "shift": shift,
                    "qualification": shift.required_qualification,
                })
            if (
                shift and shift.is_working and not shift.is_training
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
                if group == "N" and not _night_active_on(
                    _current_unit_id(), day
                )
                else int(
                    getattr(requirement, f"req_{group.lower()}", 0) or 0
                )
            )
            available = counts[day][group]
            if available < needed:
                coverage_gaps.append({
                    "day": day, "group": group,
                    "available": available, "needed": needed,
                    "shortfall": needed - available,
                })

    fatigue = _compliance_findings(year, month)
    position_rows = _position_assurance(year, month)
    position_shortfalls = [row for row in position_rows if row["shortfall"]]
    approved_rule = RosterRuleVersion.query.filter(
        RosterRuleVersion.state == "approved",
        db.or_(
            RosterRuleVersion.effective_from.is_(None),
            RosterRuleVersion.effective_from <= days[0],
        ),
    ).order_by(RosterRuleVersion.version.desc()).first()
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
        configuration_blocks.append("No position requirements configured for the month.")
    if not approved_rule:
        configuration_blocks.append("No approved rostering rule version governs the month.")
    if not BreakPlan.query.filter(
        BreakPlan.day >= days[0], BreakPlan.day <= days[-1]
    ).first():
        configuration_blocks.append("No operational break plan is recorded for the month.")
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
            fatigue["total"] + len(coverage_gaps)
            + len(position_shortfalls) + len(critical_reports)
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
        Staff.query
        .filter(
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


def _clamp_prev_next(year, month):
    """Clamp navigation so you cannot go earlier than MIN_MONTH."""
    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)
    prev_allowed = date(prev_y, prev_m, 1) >= date(
        MIN_MONTH.year, MIN_MONTH.month, 1)
    return (f"{prev_y}-{prev_m:02d}" if prev_allowed else None,
            f"{next_y}-{next_m:02d}")


@app.context_processor
def inject_perms():
    au = current_user if getattr(
        current_user, "is_authenticated", False) else None
    current_unit = (
        db.session.get(Unit, int(getattr(au, "unit_id", 0) or 0))
        if au and getattr(au, "role", "") != "superadmin" else None
    )
    branding = {}
    if current_unit:
        try:
            candidate = json.loads(current_unit.branding_json or "{}")
            if isinstance(candidate, dict):
                branding = candidate
        except (TypeError, ValueError, json.JSONDecodeError):
            branding = {}
    primary_colour = branding.get("primary_colour", "")
    accent_colour = branding.get("accent_colour", "")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", primary_colour):
        primary_colour = ""
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", accent_colour):
        accent_colour = ""
    pending_request_count = 0
    unread_notification_count = 0
    unread_briefing_count = 0
    has_briefing_module = False
    has_training_module = False
    has_competency_module = False
    active_admin_count = 0
    if current_unit and au and is_admin_user(au):
        active_admin_count = Staff.query.filter(
            Staff.unit_id == current_unit.id,
            Staff.membership_status == "active",
            db.or_(Staff.role == "admin", Staff.is_admin.is_(True)),
        ).count()
        pending_request_count = ShiftRequest.query.filter(
            ShiftRequest.unit_id == current_unit.id,
            ShiftRequest.status.in_(("pending", "approved")),
        ).count()
    if current_unit and au:
        unread_notification_count = Notification.query.filter_by(
            unit_id=current_unit.id,
            recipient_id=au.id,
            read_at=None,
        ).count()
        has_briefing_module = briefing_enabled(current_unit.id)
        has_training_module = training_enabled(current_unit.id)
        has_competency_module = competency_enabled(current_unit.id)
        if (
            has_briefing_module
            and (
                request.endpoint == "module_home"
                or (
                    request.endpoint
                    and request.endpoint.startswith("briefing.")
                )
            )
        ):
            briefing_now = briefing_local_now(current_unit.id)
            unread_briefing_count = (
                db.session.query(BriefingDelivery.id)
                .join(
                    BriefingItem,
                    BriefingItem.id == BriefingDelivery.briefing_id,
                )
                .filter(
                    BriefingDelivery.unit_id == current_unit.id,
                    BriefingDelivery.recipient_id == au.id,
                    BriefingDelivery.acknowledged_at.is_(None),
                    BriefingDelivery.archived_at.is_(None),
                    BriefingDelivery.deleted_at.is_(None),
                    BriefingItem.status == "published",
                    BriefingItem.kind != "daily",
                    BriefingItem.effective_at <= briefing_now,
                    BriefingItem.expires_at >= briefing_now,
                )
                .count()
            )
    return {
        "is_admin":  bool(au) and is_admin_user(au),
        "is_editor": bool(au) and is_editor_user(au),
        "pending_request_count": pending_request_count,
        "unread_notification_count": unread_notification_count,
        "unread_briefing_count": unread_briefing_count,
        "has_briefing_module": has_briefing_module,
        "has_training_module": has_training_module,
        "has_competency_module": has_competency_module,
        "active_admin_count": active_admin_count,
        "current_unit": current_unit,
        "unit_branding": {
            "primary_colour": primary_colour,
            "accent_colour": accent_colour,
            "display_name": (
                branding.get("display_name") or (
                    current_unit.name if current_unit else ""
                )
            )[:120],
        },
    }


@app.get("/modules")
@login_required
def module_home():
    """Select between subscribed airport modules after authentication."""
    if current_user.role == "superadmin":
        return redirect(url_for("platform_admin"))
    if not (
        briefing_enabled(current_user.unit_id)
        or training_enabled(current_user.unit_id)
        or competency_enabled(current_user.unit_id)
    ):
        return redirect(url_for("index"))
    return render_template(
        "module_home.html",
        show_briefing=briefing_enabled(current_user.unit_id),
        show_training=training_enabled(current_user.unit_id),
        show_competency=competency_enabled(current_user.unit_id),
    )



@app.route("/__can")
@login_required
def __can():
    # replicate the same logic the roster uses
    can_edit = (
        is_admin_user(current_user) or
        bool(getattr(current_user, "is_wm", False)) or
        bool(getattr(current_user, "is_dwm", False))
    )
    return {
        "is_admin_user": is_admin_user(current_user),
        "is_wm": bool(getattr(current_user, "is_wm", False)),
        "is_dwm": bool(getattr(current_user, "is_dwm", False)),
        "final_can_edit": can_edit,
    }


# -------------------- Admin --------------------


@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():

    if request.method == "POST":
        form = request.form.get("form", "")

        if form == "sms_settings":
            _validate_csrf()
            senders, sender_errors = _parse_sms_number_lines(
                request.form.get("sms_sender_numbers") or ""
            )
            destinations, destination_errors = _parse_sms_number_lines(
                request.form.get("sms_operational_numbers") or ""
            )
            default_sender = _normalise_sms_number(
                request.form.get("sms_default_sender")
            )
            default_destination = _normalise_sms_number(
                request.form.get("sms_default_operational_number")
            )
            allowed_senders = {item["number"] for item in senders}
            allowed_destinations = {item["number"] for item in destinations}
            if sender_errors or destination_errors:
                invalid = ", ".join(
                    [f"sender {item}" for item in sender_errors]
                    + [f"destination {item}" for item in destination_errors]
                )
                flash(
                    f"Use international numbers such as +447700900123. "
                    f"Check {invalid}.",
                    "error",
                )
            elif default_sender and default_sender not in allowed_senders:
                flash("The default sender must be in the sender list.", "error")
            elif (
                default_destination
                and default_destination not in allowed_destinations
            ):
                flash(
                    "The default operational number must be in its list.",
                    "error",
                )
            else:
                _save_roster_setting(
                    "sms_sender_numbers",
                    json.dumps(senders, separators=(",", ":")),
                )
                _save_roster_setting(
                    "sms_operational_numbers",
                    json.dumps(destinations, separators=(",", ":")),
                )
                _save_roster_setting(
                    "sms_default_sender",
                    default_sender or (senders[0]["number"] if senders else ""),
                )
                _save_roster_setting(
                    "sms_default_operational_number",
                    default_destination or (
                        destinations[0]["number"] if destinations else ""
                    ),
                )
                db.session.commit()
                flash("SMS numbers saved for this airport.", "ok")
            return redirect(url_for("admin") + "#sms")

        if form == "unit_roster_setup":
            pattern = _validated_pattern(request.form.get("base_pattern_csv"))
            anchor = _parse_date(request.form.get("base_pattern_anchor"))
            if not pattern or not anchor:
                flash(
                    "Choose at least one valid base-pattern duty and a start date.",
                    "error",
                )
            else:
                active_nights = [
                    str(day) for day in range(7)
                    if request.form.get(f"night_day_{day}")
                ]
                _save_roster_setting("base_pattern_csv", ",".join(pattern))
                _save_roster_setting(
                    "base_pattern_anchor", anchor.isoformat()
                )
                _save_roster_setting(
                    "night_active_weekdays", ",".join(active_nights)
                )
                db.session.commit()
                flash("Unit roster setup saved.", "ok")
            return redirect(url_for("admin") + "#roster-setup")

        if form == "watch_new":
            name = (request.form.get("name") or "").strip()
            pattern = _validated_pattern(request.form.get("pattern_csv"))
            anchor = _parse_date(request.form.get("pattern_anchor"))
            if not name:
                flash("Enter a watch name.", "error")
            elif Watch.query.filter_by(
                unit_id=_current_unit_id(), name=name
            ).first():
                flash("That watch name already exists.", "error")
            else:
                max_order = db.session.query(
                    db.func.max(Watch.order_index)
                ).filter(Watch.unit_id == _current_unit_id()).scalar() or 0
                db.session.add(Watch(
                    unit_id=_current_unit_id(),
                    name=name[:32],
                    order_index=max_order + 1,
                    pattern_csv=",".join(pattern),
                    pattern_anchor=anchor,
                ))
                db.session.commit()
                flash(f"{name} created.", "ok")
            return redirect(url_for("admin") + "#roster-setup")

        if form == "watch_edit":
            watch = Watch.query.filter_by(
                id=int(request.form.get("watch_id") or 0),
                unit_id=_current_unit_id(),
            ).first_or_404()
            name = (request.form.get("name") or "").strip()
            pattern = _validated_pattern(request.form.get("pattern_csv"))
            anchor = _parse_date(request.form.get("pattern_anchor"))
            duplicate = Watch.query.filter(
                Watch.unit_id == _current_unit_id(),
                Watch.name == name,
                Watch.id != watch.id,
            ).first()
            if not name:
                flash("Enter a watch name.", "error")
            elif duplicate:
                flash("That watch name already exists.", "error")
            else:
                watch.name = name[:32]
                watch.pattern_csv = ",".join(pattern)
                watch.pattern_anchor = anchor
                db.session.commit()
                flash(f"{watch.name} updated.", "ok")
            return redirect(url_for("admin") + "#roster-setup")

        if form == "watch_delete":
            watch = Watch.query.filter_by(
                id=int(request.form.get("watch_id") or 0),
                unit_id=_current_unit_id(),
            ).first_or_404()
            in_use = (
                Staff.query.filter_by(
                    unit_id=_current_unit_id(), watch_id=watch.id
                ).first()
                or StaffWatchHistory.query.filter_by(
                    unit_id=_current_unit_id(), watch_id=watch.id
                ).first()
            )
            if in_use:
                flash(
                    "Move staff and remove scheduled moves before deleting this watch.",
                    "error",
                )
            else:
                name = watch.name
                db.session.delete(watch)
                db.session.commit()
                flash(f"{name} deleted.", "ok")
            return redirect(url_for("admin") + "#roster-setup")

        if form == "counter_mapping":
            mapping = {}
            for shift in ShiftType.query.filter_by(
                unit_id=_current_unit_id()
            ).all():
                group = (
                    request.form.get(f"counter_group_{shift.id}") or ""
                ).strip().upper()
                if group not in {"", "M", "D", "A", "N"}:
                    abort(400, "Invalid roster counter group.")
                mapping[shift.code.upper()] = group
            _save_roster_setting(
                "shift_counter_map",
                json.dumps(mapping, sort_keys=True),
            )
            db.session.commit()
            flash("Shift counter mapping saved.", "ok")
            return redirect(url_for("admin") + "#shifts")

        # Create staff
        if form == "staff_new":
            name = request.form.get("name", "").strip()
            staff_no = request.form.get("staff_no", "").strip()
            username = request.form.get("username", "").strip()
            watch_id = request.form.get("watch_id")
            role = request.form.get("role", "user")

            # NEW flags
            is_wm = bool(request.form.get("is_wm"))
            is_dwm = bool(request.form.get("is_dwm"))
            exclude_from_ot = bool(request.form.get("exclude_from_ot"))
            permissions = {
                "edit_roster": bool(
                    request.form.get("permission_edit_roster")
                ),
                "apply_annotations": bool(
                    request.form.get("permission_apply_annotations")
                ),
            }

            # Leave/TOIL config
            leave_year_start_month = int(
                request.form.get("leave_year_start_month", 4) or 4)
            leave_entitlement_days = int(
                request.form.get("leave_entitlement_days", 0) or 0)
            leave_public_holidays = int(
                request.form.get("leave_public_holidays", 0) or 0)
            leave_carryover_days = int(
                request.form.get("leave_carryover_days", 0) or 0)

            if not all([name, staff_no, watch_id]):
                flash("Name, staff number and watch are required.", "error")
            elif Staff.query.filter_by(
                unit_id=_current_unit_id(), staff_no=staff_no
            ).first() or (
                username and Staff.query.filter(
                    Staff.unit_id == _current_unit_id(),
                    db.func.lower(Staff.username) == username.lower(),
                ).first()
            ):
                flash("Username or Staff # already exists.", "error")
            else:
                username = username or (
                    f"person-{_current_unit_id()}-{secrets.token_hex(8)}"
                )
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
                )
                s.set_password("password")
                if not s.calendar_token:
                    s.calendar_token = secrets.token_hex(16)
                db.session.add(s)
                db.session.commit()
                flash(
                    "Roster profile created. Complete the profile, then "
                    "issue account access when ready.",
                    "ok",
                )
                return redirect(url_for("admin_staff_edit", sid=s.id))

        # Create / edit / delete shifts
        if form == "shift_new":
            code = request.form.get("code", "").strip().upper()
            name = request.form.get("name", "").strip()
            start = _parse_hhmm(request.form.get("start"))
            end = _parse_hhmm(request.form.get("end"))
            is_working = bool(request.form.get("is_working"))
            is_training = bool(request.form.get("is_training"))
            is_active = bool(request.form.get("is_active"))
            is_requestable = bool(request.form.get("is_requestable"))
            required_qualification = (
                request.form.get("required_qualification") or ""
            ).strip().upper()
            allowed_qualifications = {
                row.code
                for row in QualificationType.query.filter_by(
                    unit_id=_current_unit_id(), is_active=True
                ).all()
            } | {""}
            if not code:
                flash("Shift code is required.", "error")
            elif required_qualification not in allowed_qualifications:
                flash("Unknown required qualification.", "error")
            elif is_requestable and (not is_active or not is_working):
                flash("Only active working shifts can be requestable.", "error")
            elif ShiftType.query.filter_by(code=code).first():
                flash("Shift code already exists.", "error")
            else:
                sh = ShiftType(code=code, name=name or code, start_time=start, end_time=end,
                               is_working=is_working, is_training=is_training,
                               is_active=is_active, is_requestable=is_requestable,
                               required_qualification=required_qualification)
                db.session.add(sh)
                db.session.commit()
                refresh_shift_cache()
                _shift_groups_snapshot.cache_clear()
                flash("Shift added.", "ok")
                return redirect(url_for("admin"))

        if form == "shift_edit":
            sid = int(request.form.get("shift_id"))
            sh = ShiftType.query.filter_by(
                id=sid, unit_id=_current_unit_id()
            ).first_or_404()
            sh.name = request.form.get("name", "").strip() or sh.name
            sh.start_time = _parse_hhmm(request.form.get("start"))
            sh.end_time = _parse_hhmm(request.form.get("end"))
            sh.is_working = bool(request.form.get("is_working"))
            sh.is_training = bool(request.form.get("is_training"))
            sh.is_active = bool(request.form.get("is_active"))
            requested = bool(request.form.get("is_requestable"))
            required_qualification = (
                request.form.get("required_qualification") or ""
            ).strip().upper()
            allowed_qualifications = {
                row.code
                for row in QualificationType.query.filter_by(
                    unit_id=_current_unit_id(), is_active=True
                ).all()
            } | {""}
            if required_qualification not in allowed_qualifications:
                flash("Unknown required qualification.", "error")
                return redirect(url_for("admin"))
            if requested and (not sh.is_active or not sh.is_working):
                flash("Only active working shifts can be requestable.", "error")
                return redirect(url_for("admin"))
            sh.is_requestable = requested
            sh.required_qualification = required_qualification
            db.session.commit()
            refresh_shift_cache()
            _shift_groups_snapshot.cache_clear()

            flash("Shift updated.", "ok")
            return redirect(url_for("admin"))

        if form == "shift_delete":
            sid = int(request.form.get("shift_id"))
            sh = ShiftType.query.filter_by(
                id=sid, unit_id=_current_unit_id()
            ).first_or_404()
            db.session.delete(sh)
            db.session.flush()
            _prune_roster_code_settings(_current_unit_id())
            db.session.commit()
            refresh_shift_cache()
            _shift_groups_snapshot.cache_clear()
            flash("Shift deleted.", "ok")
            return redirect(url_for("admin"))

        # Save requirements grid (includes req_d)
        if form == "req":
            yms = request.form.getlist("ym")
            field_names = [
                f"req_{prefix}{code}"
                for prefix in ("", "sat_", "sun_")
                for code in ("m", "d", "a", "n")
            ]
            values = {
                field: request.form.getlist(field)
                for field in field_names
            }
            for i in range(len(yms)):
                y, m = [int(x) for x in yms[i].split("-")]
                r = Requirement.query.filter_by(year=y, month=m).first()
                if not r:
                    r = Requirement(year=y, month=m)
                    db.session.add(r)
                for field in field_names:
                    try:
                        value = int(values[field][i] or 0)
                    except (ValueError, IndexError):
                        abort(400, f"Invalid staffing value for {field}.")
                    setattr(r, field, max(0, value))
            db.session.commit()
            flash("Requirements saved.", "ok")
            return redirect(url_for("admin") + "#requirements")

        if form == "special_requirement":
            try:
                selected_day = date.fromisoformat(
                    request.form.get("special_day") or ""
                )
            except ValueError:
                flash("Choose a valid date for the special requirement.", "error")
                return redirect(url_for("admin") + "#requirements")
            label = (request.form.get("special_label") or "").strip()[:80]
            if not label:
                flash(
                    "Describe the reason, for example Christmas Day.",
                    "error",
                )
                return redirect(url_for("admin") + "#requirements")
            row = SpecialRequirement.query.filter_by(day=selected_day).first()
            if not row:
                row = SpecialRequirement(day=selected_day)
                db.session.add(row)
            row.label = label
            for code in ("m", "d", "a", "n"):
                try:
                    value = int(
                        request.form.get(f"special_req_{code}") or 0
                    )
                except ValueError:
                    abort(400, "Special staffing values must be numbers.")
                setattr(row, f"req_{code}", max(0, value))
            db.session.commit()
            flash(
                f"Special requirements saved for "
                f"{selected_day.strftime('%d %B %Y')}.",
                "ok",
            )
            return redirect(url_for("admin") + "#requirements")

        if form == "special_requirement_delete":
            row = SpecialRequirement.query.filter_by(
                id=int(request.form.get("special_requirement_id") or 0)
            ).first_or_404()
            db.session.delete(row)
            db.session.commit()
            flash("Special requirement removed.", "ok")
            return redirect(url_for("admin") + "#requirements")

        # (Legacy) Bulk TOIL seed still accepted server-side, but you won't use it in UI.
        if form == "toil_seed":
            lines = (request.form.get("toil_seed_lines")
                     or "").strip().splitlines()
            updated = 0
            errors = 0
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    staff_no, val = [x.strip() for x in ln.split(",", 1)]
                    s = Staff.query.filter_by(staff_no=staff_no).first()
                    if not s:
                        errors += 1
                        continue
                    txt = val.lower().replace("days", "d").replace("day", "d").replace(
                        "hrs", "h").replace("hr", "h").replace("hours", "h").replace("hour", "h")
                    half = 0
                    if txt.endswith("d"):
                        days = float(txt[:-1])
                        half = int(round(days * 2))
                    elif txt.endswith("h"):
                        hours = float(txt[:-1])
                        # 8h = 1 day = 2 half-days
                        half = int(round((hours / 8.0) * 2))
                    else:
                        # bare number => days
                        days = float(txt)
                        half = int(round(days * 2))
                    s.toil_half_days = half
                    updated += 1
                except Exception:
                    errors += 1
            db.session.commit()
            flash(
                f"TOIL balances updated: {updated} staff; {errors} error(s).", "ok" if errors == 0 else "error")
            return redirect(url_for("admin"))

    # GET render
    watches = Watch.query.order_by(Watch.order_index).all()
    shifts = ShiftType.query.order_by(ShiftType.code).all()
    qualification_types = QualificationType.query.filter_by(
        unit_id=_current_unit_id(), is_active=True
    ).order_by(QualificationType.code).all()
    staff = (Staff.query
             .outerjoin(Watch, Staff.watch_id == Watch.id)
             .order_by(Watch.order_index, Staff.name).all())
    # Keep the staffing screen focused on a useful planning horizon instead of
    # making administrators scan fixed calendar years (and eventually stale
    # historic months). Show the current month plus the next 23 months.
    planning_start = date.today().replace(day=1)
    months = []
    cursor = planning_start
    for _ in range(24):
        months.append((cursor.year, cursor.month))
        cursor = (
            cursor.replace(year=cursor.year + 1, month=1)
            if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
        )
    requirements_by_month = {
        (r.year, r.month): r for r in Requirement.query.all()}
    special_requirements = SpecialRequirement.query.order_by(
        SpecialRequirement.day
    ).all()
    leaves = Leave.query.order_by(Leave.start.desc()).all()
    roster_settings = _roster_settings_snapshot(_current_unit_id())
    base_pattern = ",".join(_validated_pattern(
        roster_settings.get("base_pattern_csv") or DEFAULT_BASE_PATTERN
    ))
    base_anchor = (
        roster_settings.get("base_pattern_anchor") or "2025-01-01"
    )
    night_active_days = {
        int(value)
        for value in roster_settings.get(
            "night_active_weekdays", "0,1,2,3,4,5,6"
        ).split(",")
        if value.strip().isdigit()
    }
    shift_counter_mapping = {
        shift.code: shift_counter_group(shift.code, _current_unit_id())
        for shift in shifts
    }
    sms_senders = _sms_number_options("sms_sender_numbers")
    sms_operational_numbers = _sms_operational_options()
    sms_default_sender = _sms_default_number(
        "sms_default_sender", sms_senders
    )
    sms_default_operational_number = _sms_default_number(
        "sms_default_operational_number", sms_operational_numbers
    )
    current_unit = db.session.get(Unit, _current_unit_id())
    return render_template("admin.html",
                           shifts=shifts, staff=staff, watches=watches,
                           months=months, requirements_by_month=requirements_by_month,
                           special_requirements=special_requirements,
                           leaves=leaves,
                           qualification_types=qualification_types,
                           base_pattern=base_pattern,
                           base_anchor=base_anchor,
                           night_active_days=night_active_days,
                           pattern_codes=PATTERN_CODES,
                           shift_counter_mapping=shift_counter_mapping,
                           sms_senders=sms_senders,
                           sms_operational_numbers=sms_operational_numbers,
                           sms_default_sender=sms_default_sender,
                           sms_default_operational_number=sms_default_operational_number,
                           current_unit=current_unit)


@app.route("/admin/reference", methods=["GET", "POST"])
@login_required
@admin_required
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
                db.session.add(AnnotationAudit(
                    unit_id=unit_id, annotation_type_id=ann.id,
                    actor_id=current_user.id, action="definition_created",
                    new_value=json.dumps({"code": code, "label": label}),
                ))
                db.session.commit()
                refresh_annotation_cache()
                flash("Annotation added.", "ok")
                return redirect(url_for("admin_reference"))

            if form == "annotation_edit":
                try:
                    aid = int(request.form.get("annotation_id") or "")
                except ValueError:
                    abort(400, "Invalid annotation ID.")
                ann = AnnotationType.query.filter_by(id=aid, unit_id=unit_id).first_or_404()
                new_code = (request.form.get("code") or ann.code).strip().upper()
                if not re.fullmatch(r"[A-Z0-9]{1,10}", new_code):
                    abort(400, "Invalid annotation code.")
                if ann.has_been_used and new_code != ann.code:
                    abort(409, "A used annotation code is immutable; deactivate it and create a new definition.")
                duplicate = AnnotationType.query.filter(
                    AnnotationType.unit_id == unit_id,
                    AnnotationType.code == new_code,
                    AnnotationType.id != ann.id,
                ).first()
                if duplicate:
                    abort(409, "That annotation code already exists.")
                old_value = {
                    "code": ann.code, "label": ann.label, "category": ann.category,
                    "active": ann.is_active,
                }
                ann.code = new_code or ann.code
                ann.label = (request.form.get("label") or ann.label or new_code).strip() or new_code
                ann.category = (request.form.get("category") or ann.category or "Other").strip() or "Other"
                requested_colour = request.form.get("colour") or ann.colour
                if not re.fullmatch(r"#[0-9A-Fa-f]{6}", requested_colour or ""):
                    abort(400, "Invalid annotation colour.")
                ann.colour = requested_colour
                ann.description = (request.form.get("description") or ann.description or "")[:1000]
                try:
                    ann.toil_half_days = int(request.form.get("toil_half_days") or 0)
                except ValueError:
                    ann.toil_half_days = 0
                ann.toil_half_days = max(-200, min(ann.toil_half_days, 200))
                ann.note_required = bool(request.form.get("note_required"))
                ann.admin_only = bool(request.form.get("admin_only"))
                ann.is_active = bool(request.form.get("is_active"))
                db.session.add(AnnotationAudit(
                    unit_id=unit_id, annotation_type_id=ann.id,
                    actor_id=current_user.id, action="definition_updated",
                    old_value=json.dumps(old_value, sort_keys=True),
                    new_value=json.dumps({
                        "code": ann.code, "label": ann.label, "category": ann.category,
                        "active": ann.is_active,
                    }, sort_keys=True),
                ))
                db.session.commit()
                refresh_annotation_cache()
                flash("Annotation updated.", "ok")
                return redirect(url_for("admin_reference"))

            if form == "annotation_delete":
                try:
                    aid = int(request.form.get("annotation_id") or "")
                except ValueError:
                    abort(400, "Invalid annotation ID.")
                ann = AnnotationType.query.filter_by(id=aid, unit_id=unit_id).first_or_404()
                used = Assignment.query.filter(
                    Assignment.unit_id == unit_id,
                    Assignment.annotation.like(f"{ann.code}%"),
                ).first() is not None
                ann.has_been_used = ann.has_been_used or used
                ann.is_active = False
                db.session.add(AnnotationAudit(
                    unit_id=unit_id, annotation_type_id=ann.id,
                    actor_id=current_user.id, action="definition_deactivated",
                    old_value="active", new_value="inactive",
                ))
                db.session.commit()
                refresh_annotation_cache()
                flash("Annotation deactivated; historical use remains readable.", "ok")
                return redirect(url_for("admin_reference"))

            if form == "settings_codes":
                key = request.form.get("key", "")
                if key not in settings_meta:
                    flash("Unknown setting.", "error")
                    return redirect(url_for("admin_reference"))
                values = _normalise_codes(request.form.getlist("values"))
                valid_codes = {
                    str(code or "").strip().upper()
                    for (code,) in db.session.query(
                        ShiftType.code
                    ).filter_by(unit_id=unit_id).all()
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

    annotations = (AnnotationType.query.filter_by(unit_id=unit_id)
                   .order_by(AnnotationType.code)
                   .all())

    settings_view = []
    roster_codes = ShiftType.query.filter_by(
        unit_id=unit_id
    ).order_by(ShiftType.code).all()
    current_values = {
        "working_codes": sorted(get_working_codes()),
        "banned_codes": sorted(get_banned_roster_codes()),
        "exclude_from_counters": sorted(get_exclude_from_counters()),
        "non_working_codes": sorted(get_non_working_codes()),
    }
    for key, meta in settings_meta.items():
        settings_view.append({
            "key": key,
            "label": meta["label"],
            "help": meta["help"],
            "selected_codes": set(current_values.get(key, [])),
        })

    return render_template("admin_reference.html",
                           annotations=annotations,
                           settings=settings_view,
                           roster_codes=roster_codes)

# Keep your dedicated staff edit route (ATCO edit)


@app.route("/admin/staff/<int:sid>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_staff_edit(sid):
    # remove: if not is_admin_user(current_user): ...
    ...

    s = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
    if request.method == "POST":
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
            request.form.get("phone_number", s.phone_number))
        s.watch_id = int(request.form.get("watch_id", s.watch_id or 0)) or None

        s.is_operational = bool(request.form.get("operational"))
        s.is_trainee = bool(request.form.get("trainee"))
        s.has_ojti = bool(request.form.get("ojti"))

        # NEW flags
        s.is_wm = bool(request.form.get("is_wm"))
        s.is_dwm = bool(request.form.get("is_dwm"))
        s.exclude_from_ot = bool(request.form.get("exclude_from_ot"))
        s.permissions_json = json.dumps({
            "edit_roster": bool(
                request.form.get("permission_edit_roster")
            ),
            "apply_annotations": bool(
                request.form.get("permission_apply_annotations")
            ),
        }, sort_keys=True)

        # update role
        s.role = request.form.get("role", s.role)

        s.pattern_override = bool(request.form.get("pattern_override"))
        requested_pattern = _validated_pattern(
            request.form.get("pattern_csv")
        )
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
            "MEDICAL": s.medical_expiry, "ADI": s.tower_ue_expiry,
            "APS": s.radar_ue_expiry, "MET": s.met_ue_expiry,
        }.items():
            qtype = QualificationType.query.filter_by(
                unit_id=s.unit_id, code=code
            ).first()
            if not qtype:
                continue
            qualification = PersonQualification.query.filter_by(
                unit_id=s.unit_id, person_id=s.id,
                qualification_type_id=qtype.id,
            ).first()
            if not qualification and expiry:
                qualification = PersonQualification(
                    unit_id=s.unit_id, person_id=s.id,
                    qualification_type_id=qtype.id, status="valid",
                )
                db.session.add(qualification)
                db.session.flush()
            if qualification:
                qualification.expires_on = expiry
                qualification.updated_at = utcnow()
                _record_qualification_history(qualification, "roster_profile_updated")

        s.tower_ut = bool(request.form.get("tower_ut"))
        s.radar_ut = bool(request.form.get("radar_ut"))
        s.met_ut = bool(request.form.get("met_ut"))

        # Leave-year config
        s.leave_year_start_month = int(request.form.get(
            "leave_year_start_month", s.leave_year_start_month or 4) or 4)
        s.leave_entitlement_days = int(request.form.get(
            "leave_entitlement_days", s.leave_entitlement_days or 0) or 0)
        s.leave_public_holidays = int(request.form.get(
            "leave_public_holidays", s.leave_public_holidays or 0) or 0)
        s.leave_carryover_days = int(request.form.get(
            "leave_carryover_days", s.leave_carryover_days or 0) or 0)

        if request.form.get("reset_password"):
            s.set_password("password")

        if request.form.get("reset_calendar_token"):
            s.calendar_token = secrets.token_hex(16)

        try:
            membership = UnitMembership.query.filter_by(
                unit_id=s.unit_id, person_id=s.id, status="active"
            ).first()
            if membership:
                identity = db.session.get(
                    PlatformIdentity, membership.identity_id
                )
                if identity:
                    identity.email = s.email
            db.session.commit()
            flash("Staff updated.", "ok")
        except Exception as e:
            db.session.rollback()
            flash(f"Update failed: {e}", "error")

        return redirect(url_for("admin"))

    watches = Watch.query.order_by(Watch.order_index).all()
    account_membership = UnitMembership.query.filter_by(
        unit_id=_current_unit_id(), person_id=s.id
    ).order_by(UnitMembership.id.desc()).first()
    pending_access_invitation = SecureInvitation.query.filter_by(
        unit_id=_current_unit_id(), target_person_id=s.id,
        accepted_at=None, disabled_at=None,
    ).order_by(SecureInvitation.id.desc()).first()
    return render_template(
        "staff_edit.html", s=s, watches=watches,
        permissions=user_permissions(s),
        pattern_codes=PATTERN_CODES,
        account_membership=account_membership,
        pending_access_invitation=pending_access_invitation,
    )


@app.route("/admin/staff/<int:sid>/watch-move", methods=["POST"])
@login_required
def admin_watch_move(sid):
    if not is_admin_user(current_user):
        abort(403)
    s = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
    watch_id_val = request.form.get("watch_id")
    eff = (request.form.get("effective_date") or "").strip()

    if not watch_id_val or not eff:
        flash("Watch and effective date are required.", "error")
        return redirect(url_for("admin_staff_edit", sid=s.id))

    try:
        new_watch_id = int(watch_id_val)
    except (TypeError, ValueError):
        flash("Invalid watch selection.", "error")
        return redirect(url_for("admin_staff_edit", sid=s.id))

    try:
        eff_d = date.fromisoformat(eff)
    except ValueError:
        flash("Invalid effective date.", "error")
        return redirect(url_for("admin_staff_edit", sid=s.id))

    new_watch = Watch.query.filter_by(
        id=new_watch_id, unit_id=_current_unit_id()
    ).first()
    if not new_watch:
        flash("Invalid watch selection.", "error")
        return redirect(url_for("admin_staff_edit", sid=s.id))
    existing = StaffWatchHistory.query.filter_by(
        unit_id=_current_unit_id(),
        staff_id=s.id,
        effective_date=eff_d,
    ).first()
    if existing:
        existing.watch_id = new_watch_id
    else:
        db.session.add(StaffWatchHistory(
            unit_id=_current_unit_id(), staff_id=s.id,
            watch_id=new_watch_id, effective_date=eff_d,
        ))
    old_watch_id = s.watch_id
    if eff_d <= date.today():
        s.watch_id = new_watch_id
    db.session.commit()

    log_change("Staff", s.id, "watch_id", old_watch_id,
               new_watch_id, note=f"effective {eff_d.isoformat()}")
    flash(
        f"Watch move recorded. {s.name} follows {new_watch.name}'s "
        f"pattern from {eff_d.strftime('%d %b %Y')}.",
        "ok",
    )
    return redirect(url_for("admin_staff_edit", sid=s.id))


@app.route("/admin/staff/watch-move/<int:hid>/edit", methods=["POST"])
@login_required
def admin_watch_move_edit(hid):
    if not is_admin_user(current_user):
        abort(403)

    hist = StaffWatchHistory.query.filter_by(
        id=hid, unit_id=_current_unit_id()
    ).first_or_404()
    watch_id_val = request.form.get("watch_id")
    eff = (request.form.get("effective_date") or "").strip()

    if not watch_id_val or not eff:
        flash("Watch and effective date are required.", "error")
        return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

    try:
        new_watch_id = int(watch_id_val)
    except (TypeError, ValueError):
        flash("Invalid watch selection.", "error")
        return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

    try:
        eff_d = date.fromisoformat(eff)
    except ValueError:
        flash("Invalid effective date.", "error")
        return redirect(url_for("admin_staff_edit", sid=hist.staff_id))
    if not Watch.query.filter_by(
        id=new_watch_id, unit_id=_current_unit_id()
    ).first():
        flash("Invalid watch selection.", "error")
        return redirect(url_for("admin_staff_edit", sid=hist.staff_id))

    old_watch_id = hist.watch_id
    old_eff = hist.effective_date

    hist.watch_id = new_watch_id
    hist.effective_date = eff_d
    db.session.commit()

    if old_watch_id != new_watch_id:
        log_change("StaffWatchHistory", hist.id, "watch_id",
                   old_watch_id, new_watch_id)
    if old_eff != eff_d:
        log_change("StaffWatchHistory", hist.id, "effective_date",
                   old_eff, eff_d)

    flash("Watch move updated.", "ok")
    return redirect(url_for("admin_staff_edit", sid=hist.staff_id))


@app.route("/admin/staff/watch-move/<int:hid>/delete", methods=["POST"])
@login_required
def admin_watch_move_delete(hid):
    if not is_admin_user(current_user):
        abort(403)

    hist = StaffWatchHistory.query.filter_by(
        id=hid, unit_id=_current_unit_id()
    ).first_or_404()
    sid = hist.staff_id
    old_watch_id = hist.watch_id
    old_eff = hist.effective_date

    db.session.delete(hist)
    db.session.commit()

    log_change("StaffWatchHistory", hid, "delete", old_watch_id, None,
               note=f"effective {old_eff.isoformat()}")
    flash("Watch move deleted.", "ok")
    return redirect(url_for("admin_staff_edit", sid=sid))


@app.route("/admin/change-log")
@login_required
@admin_required
def change_log_page():
    ...

    ym = request.args.get("ym", "").strip() or None
    et = request.args.get("entity_type", "").strip() or None
    who = request.args.get("who", "").strip() or None

    q = ChangeLog.query.order_by(ChangeLog.when.desc())
    if ym:
        q = q.filter(ChangeLog.context_month == ym)
    if et:
        q = q.filter(ChangeLog.entity_type == et)
    if who and who.isdigit():
        q = q.filter(ChangeLog.who_user_id == int(who))

    rows = q.limit(500).all()
    return render_template("change_log.html", rows=rows, ym=ym, entity_type=et, who=who)


# -------------------- Leave / Sickness / TOIL --------------------


def _group_sickness_instances(assignments, month_start=None, month_end=None):
    return group_sickness_instances(assignments, month_start, month_end)


# -------------------- Staff profile --------------------


def _training_profile_allowed(person):
    return bool(
        person.id == current_user.id
        or is_editor_user(current_user)
        or can_manage_training(current_user)
        or can_record_training(current_user)
    )



@app.route("/staff/<int:sid>", methods=["GET", "POST"])
@login_required
def staff_profile(sid):
    s = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
    if s.id != current_user.id and not is_editor_user(current_user):
        abort(403)
    if request.method == "POST":
        _validate_csrf()
        if s.id != current_user.id:
            abort(403)
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
            identity = db.session.get(
                PlatformIdentity, membership.identity_id
            )
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
        mfa_recovery_codes = session.pop(
            "_new_mfa_recovery_codes", []
        )
        if not mfa_enabled:
            mfa_secret = session.get("_pending_mfa_secret")
            if not mfa_secret:
                mfa_secret = pyotp.random_base32()
                session["_pending_mfa_secret"] = mfa_secret
            mfa_provisioning_uri = pyotp.TOTP(
                mfa_secret
            ).provisioning_uri(
                name=current_user.username, issuer_name="ATCRoster"
            )
            mfa_qr_data_uri = _totp_qr_data_uri(
                mfa_provisioning_uri
            )

    # ensure_month_requirement(today.year, today.month)
    # generate_month(today.year, today.month)

    yr_ago = today - timedelta(days=365)

    al_days = sum((lv.end - lv.start).days + 1 for lv in s.leaves
                  if lv.leave_type == "AL" and lv.end >= yr_ago and lv.start <= today)

    # Sickness categories configured for this airport, counted via assignments.
    sickness_codes = {
        item["code"] for item in get_absence_types(
            "sickness", active_only=False
        )
    }
    q = (Assignment.query
         .filter(Assignment.staff_id == s.id,
                 Assignment.day >= yr_ago,
                 Assignment.day <= today))
    sick_days = sum(1 for a in q.all() if a.code in sickness_codes)

    month_start, days = month_range(today.year, today.month)
    month_end = days[-1]
    assigns = (Assignment.query
               .filter(Assignment.staff_id == s.id,
                       Assignment.day >= month_start,
                       Assignment.day <= month_end)
               .all())
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
        cal_link = url_for("calendar_feed", sid=s.id,
                           token=s.calendar_token, _external=True)
        # Apple uses webcal:// for subscription
        apple_link = cal_link.replace(
            "http://", "webcal://").replace("https://", "webcal://")
        # Google "Add by URL" link
        from urllib.parse import quote
        google_link = f"https://calendar.google.com/calendar/r?cid={quote(cal_link)}"

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
    return render_template(
        "staff_profile.html", staff=s,
        al_days=al_days, sick_days=sick_days,
        hours_this_month=hours_this_month,
        cal_link=cal_link, apple_link=apple_link, google_link=google_link,
        upcoming=upcoming[:10], next_duty=next_duty,
        notifications=notifications,
        unread_notifications=sum(1 for item in notifications if not item.read_at),
        mfa_enabled=mfa_enabled,
        mfa_secret=mfa_secret,
        mfa_provisioning_uri=mfa_provisioning_uri,
        mfa_qr_data_uri=mfa_qr_data_uri,
        mfa_recovery_codes=mfa_recovery_codes,
    )


@app.route("/staff/<int:sid>/calendar-token", methods=["POST"])
@login_required
def calendar_token_create(sid):
    _validate_csrf()
    staff = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
    if staff.id != current_user.id and not is_admin_user(current_user):
        abort(403)
    staff.calendar_token = secrets.token_hex(24)
    db.session.commit()
    flash("A new private calendar subscription link was generated.", "ok")
    return redirect(url_for("staff_profile", sid=staff.id) + "#calendar")


@app.post("/notifications/read")
@login_required
def notifications_read():
    _validate_csrf()
    Notification.query.filter_by(
        unit_id=_current_unit_id(),
        recipient_id=current_user.id, read_at=None
    ).update({"read_at": utcnow()}, synchronize_session=False)
    db.session.commit()
    return redirect(
        url_for("staff_profile", sid=current_user.id) + "#notifications"
    )


@app.post("/notifications/<int:notification_id>/read")
@login_required
def notification_read(notification_id):
    _validate_csrf()
    item = Notification.query.filter_by(
        id=notification_id,
        unit_id=_current_unit_id(),
        recipient_id=current_user.id,
    ).first_or_404()
    if not item.read_at:
        item.read_at = utcnow()
        db.session.commit()
        flash("Notification marked as read.", "ok")
    return redirect(
        url_for("staff_profile", sid=current_user.id) + "#notifications"
    )


@app.post("/notifications/<int:notification_id>/delete")
@login_required
def notification_delete(notification_id):
    _validate_csrf()
    item = Notification.query.filter_by(
        id=notification_id,
        unit_id=_current_unit_id(),
        recipient_id=current_user.id,
    ).first_or_404()
    if not item.read_at:
        flash("Mark the notification as read before deleting it.", "error")
        return redirect(
            url_for("staff_profile", sid=current_user.id) + "#notifications"
        )
    db.session.delete(item)
    db.session.commit()
    flash("Notification deleted.", "ok")
    return redirect(
        url_for("staff_profile", sid=current_user.id) + "#notifications"
    )

# -------------------- Metrics + CSV (date range; FYTD default) --------------------
# (… unchanged metrics functions from your file …)


def _compute_metrics_range(start_day: date, end_day: date):
    return compute_annotation_metrics(
        start_day,
        end_day,
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        annotation_items=_annotation_snapshot(
            int(_current_unit_id() or 1)
        )["items"],
        parse_annotation=parse_annotation,
    )


def _fy_start_for(d: date) -> date:
    return financial_year_start(d)


def _count_aava_soal_since_prev_april(staff_id: int, upto: date):
    start = date(upto.year if upto.month >= 4 else upto.year - 1, 4, 1)
    q = (Assignment.query
         .filter(Assignment.staff_id == staff_id,
                 Assignment.day >= start,
                 Assignment.day <= upto))
    aava = 0
    soal = 0
    for a in q.all():
        p = parse_annotation(a.annotation)
        if not p:
            continue
        tags = annotation_tags_for(p["type"])
        if "aava" in tags:
            aava += 1
        if "soal" in tags:
            soal += 1
    return aava, soal


def _worked_like_consecutive_days(staff: Staff, upto_day: date, lookback_days: int = 10) -> int:
    count = 0
    cur = upto_day
    for _ in range(lookback_days):
        a = Assignment.query.filter_by(staff_id=staff.id, day=cur).first()
        code = a.code if a else None
        if not code:
            break
        if code in get_working_codes():
            count += 1
            cur = cur - timedelta(days=1)
        else:
            break
    return count


def _had_sc_within_48h(staff: Staff, ref_day: date, ref_shift: ShiftType) -> bool:
    ref_start, _ = _span(ref_day, ref_shift) if ref_shift else (
        datetime.combine(ref_day, time(0, 0)), None)
    start_window = ref_start - timedelta(hours=48)
    end_window = ref_start

    q = (Assignment.query
         .filter(Assignment.staff_id == staff.id,
                 Assignment.day >= (start_window.date() - timedelta(days=1)),
                 Assignment.day <= end_window.date()))
    for a in q.all():
        if a.code in ("SC", "SSC"):
            sh = get_shift(a.code)
            sdt, edt = _span(a.day, sh) if sh else (None, None)
            if sdt and edt:
                if edt > start_window and sdt < end_window:
                    return True
    return False


def _has_in_date_ue(s: Staff, ref_day: date) -> bool:
    def valid(expiry: date, ut_flag: bool):
        return (not ut_flag) and (expiry is not None) and (expiry >= ref_day)
    tower_ok = valid(s.tower_ue_expiry, s.tower_ut)
    radar_ok = valid(s.radar_ue_expiry, s.radar_ut)
    return tower_ok or radar_ok


# -------------------- Overtime finder (admin/editor) --------------------
# (… unchanged from your file …)


def _count_ot_since_prev_april(staff_id: int, upto: date):
    start = date(upto.year if upto.month >= 4 else upto.year - 1, 4, 1)
    q = (Assignment.query
         .filter(Assignment.staff_id == staff_id,
                 Assignment.day >= start,
                 Assignment.day <= upto))
    total = 0
    for a in q.all():
        p = parse_annotation(a.annotation)
        if p and ("ot" in annotation_tags_for(p["type"])):
            total += 1
    return total

# … keep the rest of your overtime helpers exactly as pasted …


def _compute_overtime_candidates(chosen_date: date | None, chosen_shift_code: str):
    shift_code = (chosen_shift_code or "").upper().strip()
    unit_id = _current_unit_id()
    sh = get_shift(shift_code, unit_id)
    if not (chosen_date and sh and sh.is_working):
        return [], [], "Please select a valid date and working shift."

    lookahead_days = 14
    ensure_assignments_for_range(chosen_date - timedelta(days=30),
                                 chosen_date + timedelta(days=lookahead_days))

    staff_members = (
        Staff.query
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .filter(
            Staff.unit_id == unit_id,
            Staff.is_operational.is_(True),
        )
        .order_by(Watch.order_index, Staff.name)
        .all()
    )

    soal_codes = annotation_codes_for_tag("soal", active_only=False)
    soal_display = "SOAL"
    if soal_codes:
        first = soal_codes[0]
        info = get_annotation_config(first)
        soal_display = (info.get("label") if info else first) or first

    results = []
    excluded = []
    for s in staff_members:
        reasons = []
        if s.exclude_from_ot:
            reasons.append("Opted out of overtime")

        if not reasons and not _staff_has_shift_qualification(s, sh, chosen_date):
            qualification = (sh.required_qualification or "").strip().upper()
            reasons.append(
                f"Missing or expired {qualification} qualification"
                if qualification else "Missing required shift qualification"
            )

        a_today = Assignment.query.filter_by(
            unit_id=unit_id, staff_id=s.id, day=chosen_date
        ).first()
        code_today = a_today.code if a_today else "OFF"
        sh_today = get_shift(code_today, unit_id)
        if sh_today and sh_today.is_working:
            reasons.append(f"Already rostered for {code_today}")

        if code_today in ("SC", "SSC"):
            reasons.append(f"Rostered {code_today}")

        if not _has_in_date_ue(s, chosen_date):
            reasons.append("No in-date tower or radar endorsement")

        if _worked_like_consecutive_days(s, chosen_date - timedelta(days=1), lookback_days=6) >= 6:
            reasons.append("Already worked six consecutive duties")

        future_issues = would_create_new_fatigue_issues(
            s, chosen_date, shift_code, lookback_days=30, lookahead_days=lookahead_days
        )

        d24_warnings = []
        blocking_issues = {}
        for _d, _lst in future_issues.items():
            keep = []
            for _f in _lst:
                if _f.startswith(("D24:", "D24 rest deficit")):
                    d24_warnings.append(f"{_d.isoformat()}: {_f}")
                else:
                    keep.append(_f)
            if keep:
                blocking_issues[_d] = keep

        if any(blocking_issues.values()):
            fatigue_reasons = sorted({
                issue
                for issues in blocking_issues.values()
                for issue in issues
            })
            reasons.append("Blocking fatigue rule: " + "; ".join(fatigue_reasons))

        if reasons:
            excluded.append({
                "staff": s,
                "watch": s.watch.name.replace("Watch ", "") if s.watch else "-",
                "rostered_code": code_today,
                "reasons": reasons,
            })
            continue

        count_upto = chosen_date - timedelta(days=1)
        aava_to_date, soal_to_date = _count_aava_soal_since_prev_april(
            s.id, count_upto)
        total_to_date = aava_to_date + soal_to_date

        flags = []
        if code_today == "AL":
            flags.append(f"On AL that day — {soal_display} required")
        if _had_sc_within_48h(s, chosen_date, sh):
            flags.append(
                "SC/SSC within 48h — managerial approval required")
        flags.extend(d24_warnings)

        results.append({
            "staff": s,
            "watch": s.watch.name.replace("Watch ", "") if s.watch else "-",
            "aava_to_date": aava_to_date,
            "soal_to_date": soal_to_date,
            "total_to_date": total_to_date,
            "score": total_to_date,
            "flags": flags
        })

    results.sort(key=lambda r: (
        r["aava_to_date"], r["soal_to_date"], r["staff"].name.lower()))
    excluded.sort(key=lambda row: row["staff"].name.lower())
    return results, excluded, None


@app.route("/overtime", methods=["GET", "POST"])
@login_required
def overtime():
    if request.method == "POST" and not _consume_rate_limit(
        "overtime-search", current_user.id, limit=60,
        window=timedelta(hours=1),
    ):
        abort(429)
    if not (
        is_editor_user(current_user)
        or getattr(current_user, "is_wm", False)
        or getattr(current_user, "is_dwm", False)
    ):
        abort(403)

    unit_id = _current_unit_id()
    shifts = ShiftType.query.filter_by(
        unit_id=unit_id, is_working=True
    ).order_by(ShiftType.code).all()
    overtime_staff = (
        Staff.query
        .filter_by(unit_id=unit_id, is_operational=True)
        .order_by(Staff.name)
        .all()
    )
    results = []
    excluded = []
    what_if_result = None
    chosen_date = None
    chosen_shift = None
    what_if_staff_id = None
    selected_staff_ids: set[str] = set()
    sms_body = ""
    searched = request.method == "POST"

    if request.method == "POST":
        _validate_csrf()
        action = request.form.get("action", "find")
        chosen_date = _parse_date(request.form.get("date"))
        chosen_shift = (request.form.get("shift_code") or "").upper().strip()
        raw_what_if_staff_id = request.form.get("what_if_staff_id", "")
        what_if_staff_id = (
            int(raw_what_if_staff_id)
            if raw_what_if_staff_id.isdigit()
            else None
        )
        selected_staff_ids = {sid for sid in request.form.getlist("staff_ids")}
        sms_body = (request.form.get("message") or "").strip()

        results, excluded, error_msg = _compute_overtime_candidates(
            chosen_date, chosen_shift
        )

        if action == "what_if":
            searched = False
            selected_staff = next(
                (
                    person for person in overtime_staff
                    if person.id == what_if_staff_id
                ),
                None,
            )
            if error_msg:
                flash(error_msg, "error")
            elif selected_staff is None:
                flash("Select an ATCO to check.", "error")
            else:
                eligible_result = next(
                    (
                        row for row in results
                        if row["staff"].id == selected_staff.id
                    ),
                    None,
                )
                excluded_result = next(
                    (
                        row for row in excluded
                        if row["staff"].id == selected_staff.id
                    ),
                    None,
                )
                if eligible_result:
                    what_if_result = {
                        "eligible": True,
                        "staff": selected_staff,
                        "flags": eligible_result["flags"],
                    }
                else:
                    what_if_result = {
                        "eligible": False,
                        "staff": selected_staff,
                        "reasons": (
                            excluded_result["reasons"]
                            if excluded_result
                            else ["Eligibility could not be determined"]
                        ),
                    }
        elif action == "send_sms":
            if not can_send_unit_messages(current_user):
                abort(403)
            if error_msg:
                flash(error_msg, "error")
                results = []
            else:
                if not sms_body:
                    flash("Enter a message to send.", "error")
                elif len(sms_body) > 480:
                    flash("Message is too long (limit 480 characters).", "error")
                else:
                    eligible_map = {r["staff"].id: r["staff"] for r in results}
                    selected_staff = [eligible_map[int(sid)]
                                      for sid in selected_staff_ids
                                      if sid.isdigit() and int(sid) in eligible_map]
                    missing_ids = [sid for sid in selected_staff_ids
                                    if sid.isdigit() and int(sid) not in eligible_map]
                    if not selected_staff:
                        flash("Select at least one eligible staff member.", "error")
                    else:
                        if missing_ids:
                            flash("Some selected staff are no longer eligible; please refresh the list.", "error")
                        sent, failures = _send_overtime_sms_notifications(selected_staff, sms_body)
                        if sent:
                            plural = "s" if sent != 1 else ""
                            flash(f"SMS sent to {sent} staff member{plural}.", "ok")
                        if failures:
                            parts = []
                            for staff, msg in failures:
                                name = staff.name if staff else "System"
                                parts.append(f"{name}: {msg}")
                            flash("SMS failed for " + "; ".join(parts), "error")

        else:  # action == find or unknown
            if error_msg:
                flash(error_msg, "error")
                results = []

        if not sms_body:
            sms_body = _default_overtime_sms_body(chosen_date, chosen_shift)

    sms_ready = _sms_service_configured()

    return render_template("overtime.html",
                           shifts=shifts, results=results,
                           overtime_staff=overtime_staff,
                           what_if_result=what_if_result,
                           what_if_staff_id=what_if_staff_id,
                           chosen_date=chosen_date, chosen_shift=chosen_shift,
                           sms_body=sms_body, sms_ready=sms_ready,
                           selected_staff_ids=selected_staff_ids,
                           searched=searched, excluded=excluded)


@app.route("/messages", methods=["GET", "POST"])
@login_required
def unit_messages():
    if not can_send_unit_messages(current_user):
        abort(403)
    people = Staff.query.filter_by(
        membership_status="active"
    ).order_by(Staff.name).all()
    watches = Watch.query.order_by(Watch.order_index, Watch.name).all()
    selected_scope = request.form.get("scope", "all")
    selected_recipient = request.form.get("recipient_id", "")
    selected_watch = request.form.get("watch_id", "")
    sender_options = _sms_sender_options()
    operational_options = _sms_operational_options()
    default_sender = _sms_default_number(
        "sms_default_sender", sender_options
    )
    default_operational = _sms_default_number(
        "sms_default_operational_number", operational_options
    )
    selected_sender = _normalise_sms_number(
        request.form.get("sender_number") or default_sender
    )
    selected_operational = _normalise_sms_number(
        request.form.get("operational_number") or default_operational
    )
    template = request.form.get("template", "custom")
    message = (request.form.get("message") or "").strip()
    preview = []

    if request.method == "POST":
        _validate_csrf()
        recipients = []
        direct_recipient = None
        allowed_senders = {item["number"] for item in sender_options}
        allowed_operational = {
            item["number"] for item in operational_options
        }
        if selected_sender not in allowed_senders:
            abort(400, "Choose an approved sender number for this airport.")
        if selected_scope == "all":
            recipients = people
        elif selected_scope == "watch" and selected_watch.isdigit():
            recipients = [
                person for person in people
                if person.watch_id == int(selected_watch)
            ]
        elif selected_scope == "individual" and selected_recipient.isdigit():
            recipients = [
                person for person in people
                if person.id == int(selected_recipient)
            ]
        elif (
            selected_scope == "operational"
            and selected_operational in allowed_operational
        ):
            direct_recipient = selected_operational
        if not recipients and not direct_recipient:
            flash("Choose at least one recipient.", "error")
        elif direct_recipient and template == "today_shift":
            flash(
                "Shift reminders can only be sent to rostered people.",
                "error",
            )
        elif template == "today_shift":
            today = date.today()
            assignment_map = {
                row.staff_id: row for row in Assignment.query.filter(
                    Assignment.day == today,
                    Assignment.staff_id.in_([person.id for person in recipients]),
                ).all()
            }
            failures = []
            sent = 0
            for person in recipients:
                assignment = assignment_map.get(person.id)
                body = (
                    f"Hello {person.name}, you are rostered for "
                    f"{assignment.code if assignment else 'no assigned'} shift today "
                    f"({today.strftime('%d %b %Y')})."
                )
                ok, detail = _send_sms_via_twilio(
                    person.phone_number, body, from_number=selected_sender
                )
                preview.append((person.name, body))
                if ok:
                    _record_sms_audit(
                        sender_number=selected_sender,
                        recipient_number=person.phone_number,
                        recipient_label=person.name,
                        body=body,
                        message_type="shift_reminder",
                        provider_message_id=detail,
                    )
                    sent += 1
                else:
                    failures.append((person, detail))
            _flash_sms_result(sent, failures)
        elif not message:
            flash("Enter a custom message.", "error")
        elif len(message) > 480:
            flash("Message is too long (limit 480 characters).", "error")
        elif direct_recipient:
            ok, detail = _send_sms_via_twilio(
                direct_recipient, message, from_number=selected_sender
            )
            label = next(
                (
                    item["label"] for item in operational_options
                    if item["number"] == direct_recipient
                ),
                direct_recipient,
            )
            preview = [(label, message)]
            if ok:
                _record_sms_audit(
                    sender_number=selected_sender,
                    recipient_number=direct_recipient,
                    recipient_label=label,
                    body=message,
                    message_type="operational",
                    provider_message_id=detail,
                )
            _flash_sms_result(
                1 if ok else 0,
                [] if ok else [(None, detail)],
            )
        else:
            sent = 0
            failures = []
            for person in recipients:
                ok, detail = _send_sms_via_twilio(
                    person.phone_number,
                    message,
                    from_number=selected_sender,
                )
                if ok:
                    _record_sms_audit(
                        sender_number=selected_sender,
                        recipient_number=person.phone_number,
                        recipient_label=person.name,
                        body=message,
                        message_type="unit",
                        provider_message_id=detail,
                    )
                    sent += 1
                else:
                    failures.append((person, detail))
            preview = [(person.name, message) for person in recipients]
            _flash_sms_result(sent, failures)

    return render_template(
        "messages.html", people=people, watches=watches,
        sms_ready=_sms_service_configured(), template=template,
        message=message, selected_scope=selected_scope,
        selected_recipient=selected_recipient, selected_watch=selected_watch,
        sender_options=sender_options,
        operational_options=operational_options,
        selected_sender=selected_sender,
        selected_operational=selected_operational,
        preview=preview,
    )


@app.route("/admin/sms-audit")
@login_required
@admin_required
def admin_sms_audit():
    rows = SmsAudit.query.order_by(
        SmsAudit.sent_at.desc(), SmsAudit.id.desc()
    ).limit(1000).all()
    return render_template("admin_sms_audit.html", rows=rows)

# -------------------- Calendar subscription --------------------


def _calendar_window_today():
    today = date.today()
    cur_start = date(today.year, today.month, 1)
    nxt_month = (today.month % 12) + 1
    nxt_year = today.year + (1 if today.month == 12 else 0)
    include_next = today.day >= 20
    cur_end = (cur_start.replace(day=28) + timedelta(days=10)
               ).replace(day=1) - timedelta(days=1)
    nxt_end = (date(nxt_year, nxt_month, 1).replace(day=28) +
               timedelta(days=10)).replace(day=1) - timedelta(days=1)
    start = cur_start
    end = nxt_end if include_next else cur_end
    return start, end


def _ical_escape(txt: str) -> str:
    return (txt or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


@app.route("/calendar/<int:sid>/<token>.ics")
def calendar_feed(sid, token):
    # The unguessable token is the credential. Calendar clients are not
    # logged in, so this public feed must not depend on session tenancy.
    s = Staff.query.filter_by(id=sid, calendar_token=token).first_or_404()
    if not s.calendar_token or token != s.calendar_token:
        abort(403)

    start, end = _calendar_window_today()
    q = (Assignment.query
         .filter(Assignment.staff_id == s.id,
                 Assignment.day >= start,
                 Assignment.day <= end)
         .order_by(Assignment.day.asc()))

    lines = []
    lines.append("BEGIN:VCALENDAR")
    lines.append("VERSION:2.0")
    lines.append("PRODID:-//ATC Roster//EN")
    lines.append("CALSCALE:GREGORIAN")
    lines.append(f"X-WR-CALNAME:{_ical_escape(s.name)} Roster")

    for a in q.all():
        sh = get_shift(a.code, unit_id=s.unit_id)
        uid = f"{s.id}-{a.day.isoformat()}-{a.code}@atcroster"
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{uid}")
        summary = f"{a.code}"
        if a.annotation:
            summary += f" ({a.annotation})"
        lines.append(f"SUMMARY:{_ical_escape(summary)}")
        if sh and sh.start_time and sh.end_time and sh.is_working:
            dt0 = datetime.combine(a.day, sh.start_time)
            dt1 = datetime.combine(a.day, sh.end_time)
            if sh.end_time <= sh.start_time:
                dt1 += timedelta(days=1)
            lines.append(f"DTSTART:{dt0.strftime('%Y%m%dT%H%M%S')}")
            lines.append(f"DTEND:{dt1.strftime('%Y%m%dT%H%M%S')}")
        else:
            lines.append(f"DTSTART;VALUE=DATE:{a.day.strftime('%Y%m%d')}")
            lines.append(
                f"DTEND;VALUE=DATE:{(a.day + timedelta(days=1)).strftime('%Y%m%d')}")
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    ics = "\r\n".join(lines).encode("utf-8")
    return Response(ics, mimetype="text/calendar; charset=utf-8")

# ===== Leave Report (HTML + CSV) =====
# (unchanged core; monthly AL-only kept to endpoints)


def _leave_summary_for_month(year: int, month: int, watch_id: int | None = None):
    return leave_summary_for_month(
        year,
        month,
        watch_id,
        unit_id=_current_unit_id(),
        Assignment=Assignment,
        Staff=Staff,
        Watch=Watch,
        month_range=month_range,
        active_leave_types=get_absence_types("leave", active_only=True),
    )


# ===== Leave-Year report (per-person config; AL only; includes TOIL days) =====
# (unchanged from your post)

def _current_leave_year_window(s: Staff, today: date | None = None):
    return current_leave_year_window(s, today)


def _toil_accrual_half_days_from_annotation(parsed):
    if not parsed:
        return 0
    info = get_annotation_config(parsed.get("type"))
    if not info:
        return 0
    try:
        return int(info.get("toil_half_days", 0) or 0)
    except Exception:
        return 0


def _apply_toil_annotation_delta(staff: Staff, old_annot: str, new_annot: str):
    old_half = _toil_accrual_half_days_from_annotation(
        parse_annotation(old_annot))
    new_half = _toil_accrual_half_days_from_annotation(
        parse_annotation(new_annot))
    delta = new_half - old_half
    if delta:
        s = tenant_get(Staff, staff.id)
        s.toil_half_days = int((s.toil_half_days or 0) + delta)


def _toil_accrued_used_in_range_half_days(staff_id: int, start_day: date, end_day: date):
    acc = use = 0
    q = (Assignment.query
         .filter(Assignment.staff_id == staff_id,
                 Assignment.day >= start_day,
                 Assignment.day <= end_day))
    for a in q.all():
        pa = parse_annotation(a.annotation)
        acc += _toil_accrual_half_days_from_annotation(pa)
        if a.code == "TOU8":
            use += 2
        elif a.code == "TOUI":
            use += 1
    return acc, use


# ===== Sickness Report (unchanged) =====


def _group_consecutive_days(days_set):
    return group_consecutive_days(days_set)


# -------------------- Request Sheets (shift requests) --------------------


def _unit_request_rules(unit_id: int | None = None) -> tuple[int, int]:
    unit = db.session.get(Unit, unit_id or _current_unit_id())
    return normalise_request_rules(
        getattr(unit, "request_months_ahead", 3),
        getattr(unit, "request_lock_day", 20),
    )


def _lock_date_for_target_month(y: int, m: int, unit_id: int | None = None):
    _, lock_day = _unit_request_rules(unit_id)
    return request_lock_date(y, m, lock_day)


def _is_month_locked(y: int, m: int, today: date | None = None, unit_id: int | None = None):
    _, lock_day = _unit_request_rules(unit_id)
    return request_month_is_locked(y, m, lock_day, today)


def _add_months(first: date, count: int) -> date:
    return add_request_months(first, count)


def _request_date_bounds(today: date, unit_id: int) -> tuple[date, date]:
    months, _ = _unit_request_rules(unit_id)
    return request_date_bounds(today, months)


def _request_audit(req: ShiftRequest, actor_id: int, transition: str,
                   old_value: object, new_value: object, reason: str = "") -> None:
    db.session.add(RequestAudit(
        unit_id=req.unit_id,
        request_id=req.id,
        actor_id=actor_id,
        transition=transition,
        old_value=json.dumps(old_value, default=str, sort_keys=True),
        new_value=json.dumps(new_value, default=str, sort_keys=True),
        reason=(reason or "")[:500],
    ))


def _notify_requester(req: ShiftRequest) -> None:
    if req.status not in {"pending", "approved", "rejected", "fulfilled"}:
        return
    if req.status == "fulfilled":
        outcome = "was approved and added to the roster"
    elif req.status == "rejected":
        outcome = "was refused"
    else:
        outcome = f"is now {req.status}"
    comment = (req.admin_response or "").strip()
    comment_text = f" Manager comment: {comment}" if comment else ""
    db.session.add(Notification(
        unit_id=req.unit_id,
        recipient_id=req.staff_id,
        kind=f"shift_request_{req.status}",
        message=(
            f"Your {req.code} shift request for "
            f"{req.day.strftime('%d %B %Y')} {outcome}.{comment_text}"
        ),
    ))


def _safe_request_admin_month(raw_value: str | None, fallback: date) -> str:
    return safe_admin_month(raw_value, fallback)


def staff_has_qualification(
    staff: Staff, qualification_code: str, duty_date: date
) -> bool:
    """Evaluate authoritative, tenant-scoped competence on the duty date."""
    code = (qualification_code or "").strip().upper()
    if not code:
        return True
    unit_id = int(getattr(staff, "unit_id", 0) or 0)
    try:
        context_unit_id = authenticated_unit_id()
    except RuntimeError:
        return False
    if not unit_id or unit_id != context_unit_id:
        return False
    qualification_type = QualificationType.query.filter_by(
        unit_id=unit_id, code=code, is_active=True
    ).first()
    if not qualification_type:
        return False
    record = PersonQualification.query.filter_by(
        unit_id=unit_id,
        person_id=staff.id,
        qualification_type_id=qualification_type.id,
    ).first()
    if not record or record.status != "valid":
        return False
    if record.valid_from and record.valid_from > duty_date:
        return False
    if qualification_type.expiry_required:
        return bool(record.expires_on and record.expires_on >= duty_date)
    return not record.expires_on or record.expires_on >= duty_date


def _staff_has_shift_qualification(
    staff: Staff, shift: ShiftType, duty_date: date | None = None
) -> bool:
    return staff_has_qualification(
        staff,
        shift.required_qualification,
        duty_date or date.today(),
    )




# >>> Admin can respond to a specific request




@app.route("/platform/admin", methods=["GET", "POST"])
@login_required
def platform_admin():
    """Privacy-preserving control plane: aggregates and unit metadata only."""
    if getattr(current_user, "role", "") != "superadmin":
        abort(403)
    platform_actor = PlatformIdentity.query.filter_by(
        username=current_user.username
    ).first()
    if not platform_actor:
        abort(403, "Super Admin identity is not provisioned in the control plane.")
    if request.method == "POST":
        _validate_csrf()
        action = (request.form.get("action") or "").strip()
        if action == "create_unit":
            code = (request.form.get("code") or "").strip().upper()
            name = (request.form.get("name") or "").strip()
            plan = (request.form.get("plan") or "starter").strip()[:40]
            try:
                limit = int(request.form.get("active_user_limit") or 10)
            except ValueError:
                limit = 0
            if not re.fullmatch(r"[A-Z0-9]{2,12}", code):
                flash("Airport code must be 2–12 letters or numbers.", "error")
            elif not name:
                flash("Airport name is required.", "error")
            elif not 1 <= limit <= 10000:
                flash("Active-user limit must be between 1 and 10,000.", "error")
            elif Unit.query.filter_by(code=code).first():
                flash("That airport code already exists.", "error")
            else:
                try:
                    unit = Unit(
                        code=code, name=name, plan=plan,
                        active_user_limit=limit, onboarding_step=0,
                        status="provisioning",
                    )
                    db.session.add(unit)
                    db.session.flush()
                    db.session.add(DatabaseRoutingMetadata(
                        unit_id=unit.id,
                        secret_name=f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL",
                        provisioning_state="pending",
                    ))
                    db.session.add(PlanHistory(
                        unit_id=unit.id, plan=plan,
                        active_user_limit=limit,
                    ))
                    db.session.add(SuperAdminAudit(
                        actor_identity_id=platform_actor.id, unit_id=unit.id,
                        action="airport_created",
                        safe_summary=f"Created airport {code} on {plan} plan with limit {limit}",
                    ))
                    db.session.commit()
                    flash(
                        f"{name} metadata created. Configure its database "
                        "secret, then run provisioning.",
                        "ok",
                    )
                    return redirect(url_for("platform_admin"))
                except Exception:
                    db.session.rollback()
                    raise
        elif action == "provision_unit":
            if not _consume_rate_limit(
                "airport-provisioning", platform_actor.id,
                limit=10, window=timedelta(hours=1),
            ):
                abort(429, "Too many provisioning requests.")
            unit_id = int(request.form.get("unit_id") or 0)
            unit = db.session.get(Unit, unit_id)
            routing = db.session.get(DatabaseRoutingMetadata, unit_id)
            if not unit or unit.status == "platform_control" or not routing:
                abort(404)
            active_accounts = UnitMembership.query.filter_by(
                unit_id=unit_id, status="active"
            ).count()
            if active_accounts:
                flash(
                    "This airport already has active accounts. Manage access "
                    "from the airport's Accounts page; bootstrap provisioning "
                    "is only for a new airport.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            existing_invitation = SecureInvitation.query.filter_by(
                unit_id=unit_id,
                role="UnitAdmin",
                active_bootstrap_key="active",
            ).first()
            if existing_invitation:
                flash(
                    "A bootstrap invitation is already active. Show that "
                    "one-time link or use Revoke and replace.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            active = ProvisioningJob.query.filter(
                ProvisioningJob.unit_id == unit_id,
                ProvisioningJob.state.in_(("queued", "running", "retry_wait")),
            ).with_for_update().first()
            if not active:
                active = ProvisioningJob(
                    unit_id=unit_id,
                    idempotency_key=hashlib.sha256(
                        f"{unit_id}:{secrets.token_hex(16)}".encode()
                    ).hexdigest(),
                    state="queued", active_key="active",
                    next_attempt_at=utcnow(),
                )
                db.session.add(active)
            elif active.state == "retry_wait":
                active.state = "queued"
                active.next_attempt_at = utcnow()
            routing.provisioning_state = "queued"
            try:
                db.session.commit()
            except IntegrityError:
                # A concurrent request won the database uniqueness race.
                # Treat this request as an idempotent resume.
                db.session.rollback()
                active = ProvisioningJob.query.filter_by(
                    unit_id=unit_id, active_key="active"
                ).first()
                if not active:
                    raise
            flash(
                "Provisioning was queued. The worker will migrate and check "
                "the airport database before issuing an invitation.",
                "ok",
            )
            return redirect(url_for("platform_admin"))
        elif action == "cancel_provisioning":
            job = ProvisioningJob.query.filter_by(
                id=int(request.form.get("job_id") or 0)
            ).with_for_update().first_or_404()
            job.cancel_requested = True
            job.updated_at = utcnow()
            db.session.commit()
            flash("Provisioning cancellation requested.", "ok")
            return redirect(url_for("platform_admin"))
        elif action == "reveal_bootstrap":
            job = ProvisioningJob.query.filter_by(
                id=int(request.form.get("job_id") or 0), state="completed"
            ).first_or_404()
            from platform_provisioning import pop_one_time_token

            raw_token = pop_one_time_token(job.id, job.unit_id)
            if raw_token:
                invite_url = url_for(
                    "accept_invitation", token=raw_token, _external=True
                )
                flash(
                    "Copy this bootstrap link now; it will not be shown "
                    f"again: {invite_url}",
                    "ok",
                )
            else:
                flash(
                    "The one-time link is no longer available. Revoke the "
                    "pending bootstrap and deliberately issue a replacement.",
                    "error",
                )
            return redirect(url_for("platform_admin"))
        elif action == "replace_bootstrap":
            unit_id = int(request.form.get("unit_id") or 0)
            unit = db.session.get(Unit, unit_id)
            if not unit or unit.status == "platform_control":
                abort(404)
            if UnitMembership.query.filter_by(
                unit_id=unit_id, status="active"
            ).count():
                flash(
                    "This airport already has active accounts, so a bootstrap "
                    "invitation cannot be replaced.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            invitation = SecureInvitation.query.filter_by(
                unit_id=unit_id, role="UnitAdmin",
                active_bootstrap_key="active",
            ).with_for_update().first()
            if invitation:
                invitation.disabled_at = utcnow()
                invitation.active_bootstrap_key = None
            active = ProvisioningJob.query.filter_by(
                unit_id=unit_id, active_key="active"
            ).first()
            if active:
                flash("Provisioning is already in progress.", "error")
            else:
                db.session.add(ProvisioningJob(
                    unit_id=unit_id,
                    idempotency_key=hashlib.sha256(
                        f"{unit_id}:{secrets.token_hex(16)}".encode()
                    ).hexdigest(),
                    state="queued", active_key="active",
                    next_attempt_at=utcnow(),
                ))
                routing = db.session.get(DatabaseRoutingMetadata, unit_id)
                if routing:
                    routing.provisioning_state = "queued"
                flash("Replacement bootstrap generation was queued.", "ok")
            db.session.commit()
            return redirect(url_for("platform_admin"))
        elif action == "update_limit":
            try:
                unit_id = int(request.form.get("unit_id") or 0)
                new_limit = int(request.form.get("active_user_limit") or 0)
            except ValueError:
                abort(400)
            unit = db.session.get(Unit, unit_id)
            if not unit or unit.status == "platform_control":
                abort(404)
            if not 1 <= new_limit <= 10000:
                flash("Active-user limit must be between 1 and 10,000.", "error")
            else:
                active_count = UnitMembership.query.filter_by(
                    unit_id=unit.id, status="active"
                ).count()
                if new_limit < active_count:
                    flash(
                        f"Limit cannot be below the {active_count} active accounts.",
                        "error",
                    )
                else:
                    old_limit = unit.active_user_limit
                    unit.active_user_limit = new_limit
                    db.session.add(PlanHistory(
                        unit_id=unit.id, plan=unit.plan,
                        active_user_limit=new_limit,
                    ))
                    db.session.add(SuperAdminAudit(
                        actor_identity_id=platform_actor.id, unit_id=unit.id,
                        action="account_limit_changed",
                        safe_summary=f"Changed active-user limit from {old_limit} to {new_limit}",
                    ))
                    db.session.commit()
                    flash(f"{unit.code} account limit updated.", "ok")
                    return redirect(url_for("platform_admin"))
        elif action == "toggle_suspension":
            unit_id = int(request.form.get("unit_id") or 0)
            unit = db.session.get(Unit, unit_id)
            if not unit or unit.status == "platform_control":
                abort(404)
            if unit.status == "suspended":
                unit.status = "active"
                unit.suspended_at = None
                action_name = "airport_restored"
            else:
                unit.status = "suspended"
                unit.suspended_at = utcnow()
                action_name = "airport_suspended"
            db.session.add(SuperAdminAudit(
                actor_identity_id=platform_actor.id, unit_id=unit.id,
                action=action_name, safe_summary=f"{action_name}: {unit.code}",
            ))
            db.session.commit()
            return redirect(url_for("platform_admin"))
        elif action == "delete_unit":
            unit_id = int(request.form.get("unit_id") or 0)
            confirmation = (
                request.form.get("confirmation_code") or ""
            ).strip().upper()
            database_acknowledged = (
                request.form.get("database_retained") == "yes"
            )
            unit = db.session.get(Unit, unit_id)
            if not unit or unit.status == "platform_control":
                abort(404)
            if confirmation != unit.code.upper():
                flash(
                    f"Type {unit.code} exactly to confirm airport deletion.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            if not database_acknowledged:
                flash(
                    "Confirm that the separate airport database will be "
                    "retained for deliberate backup and decommissioning.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            active_accounts = UnitMembership.query.filter_by(
                unit_id=unit.id, status="active"
            ).count()
            active_job = ProvisioningJob.query.filter(
                ProvisioningJob.unit_id == unit.id,
                ProvisioningJob.state.in_((
                    "queued", "running", "retry_wait",
                )),
            ).first()
            if active_accounts:
                flash(
                    "Suspend or remove every active airport account before "
                    "deleting the airport.",
                    "error",
                )
                return redirect(url_for("platform_admin"))
            if active_job:
                flash(
                    "Cancel and finish the active provisioning job before "
                    "deleting the airport.",
                    "error",
                )
                return redirect(url_for("platform_admin"))

            invitation_ids = [
                row.id for row in SecureInvitation.query.filter_by(
                    unit_id=unit.id
                ).all()
            ]
            membership_ids = [
                row.id for row in UnitMembership.query.filter_by(
                    unit_id=unit.id
                ).all()
            ]
            workflow_filters = []
            if invitation_ids:
                workflow_filters.append(
                    SignupWorkflow.invitation_id.in_(invitation_ids)
                )
            if membership_ids:
                workflow_filters.append(
                    SignupWorkflow.membership_id.in_(membership_ids)
                )
            if workflow_filters:
                db.session.query(SignupWorkflow).filter(
                    db.or_(*workflow_filters)
                ).delete(synchronize_session=False)

            job_ids = [
                row.id for row in ProvisioningJob.query.filter_by(
                    unit_id=unit.id
                ).all()
            ]
            db.session.query(SuperAdminAudit).filter_by(
                unit_id=unit.id
            ).update({"unit_id": None}, synchronize_session=False)
            for model in (
                SecureInvitation,
                ProvisioningJob,
                FeatureFlag,
                PlanHistory,
                AggregateUsageEvent,
                DatabaseRoutingMetadata,
                UnitMembership,
            ):
                db.session.query(model).filter_by(
                    unit_id=unit.id
                ).delete(synchronize_session=False)
            deleted_code = unit.code
            db.session.delete(unit)
            db.session.add(SuperAdminAudit(
                actor_identity_id=platform_actor.id,
                unit_id=None,
                action="airport_deleted",
                safe_summary=(
                    f"Deleted airport metadata for {deleted_code}; "
                    "operational database retained for decommissioning."
                ),
            ))
            db.session.commit()
            if job_ids and os.environ.get("REDIS_URL"):
                try:
                    import redis

                    cache = redis.from_url(
                        os.environ["REDIS_URL"],
                        socket_connect_timeout=2,
                        socket_timeout=2,
                        decode_responses=True,
                    )
                    cache.delete(*[
                        f"atcroster:provisioning-token:{job_id}"
                        for job_id in job_ids
                    ])
                except Exception:
                    _security_event(
                        "airport_token_cleanup_failed",
                        unit_digest=hashlib.sha256(
                            str(unit_id).encode()
                        ).hexdigest()[:16],
                    )
            flash(
                f"{deleted_code} airport metadata deleted. Its separate "
                "database was retained and must be backed up or destroyed "
                "through the database provider.",
                "ok",
            )
            return redirect(url_for("platform_admin"))
        elif action == "set_feature":
            try:
                unit_id = int(request.form.get("unit_id") or 0)
            except ValueError:
                abort(400)
            key = (request.form.get("key") or "").strip()
            if key not in PLATFORM_FEATURE_FLAGS:
                abort(400, "Unknown feature flag.")
            unit = db.session.get(Unit, unit_id)
            if not unit or unit.status == "platform_control":
                abort(404)
            row = FeatureFlag.query.filter_by(
                unit_id=unit.id, key=key
            ).first()
            if not row:
                row = FeatureFlag(unit_id=unit.id, key=key)
                db.session.add(row)
            old_enabled = bool(row.enabled)
            row.enabled = request.form.get("enabled") == "yes"
            db.session.add(SuperAdminAudit(
                actor_identity_id=platform_actor.id,
                unit_id=unit.id,
                action="feature_flag_changed",
                safe_summary=(
                    f"Changed {key} from {old_enabled} to {row.enabled}"
                ),
            ))
            db.session.commit()
            return redirect(url_for("platform_admin"))
        else:
            abort(400)
    rows = []
    now = utcnow()
    for unit in Unit.query.filter(
        Unit.status != "platform_control"
    ).order_by(Unit.code).all():
        active_accounts = UnitMembership.query.filter_by(
            unit_id=unit.id, status="active"
        ).count()
        flags = {
            row.key: row.enabled
            for row in FeatureFlag.query.filter_by(unit_id=unit.id).all()
        }
        if "competency_module" not in flags:
            flags["competency_module"] = bool(flags.get("training_module"))
        routing = db.session.get(DatabaseRoutingMetadata, unit.id)
        activity = db.session.query(
            db.func.coalesce(db.func.sum(AggregateUsageEvent.count), 0)
        ).filter(AggregateUsageEvent.unit_id == unit.id).scalar()
        bootstrap = SecureInvitation.query.filter_by(
            unit_id=unit.id, role="UnitAdmin"
        ).order_by(SecureInvitation.id.desc()).first()
        latest_job = ProvisioningJob.query.filter_by(
            unit_id=unit.id
        ).order_by(ProvisioningJob.id.desc()).first()
        if (
            latest_job
            and latest_job.state == "completed"
            and latest_job.last_error_code == "bootstrap_already_issued"
        ):
            latest_job = ProvisioningJob.query.filter_by(
                unit_id=unit.id,
                state="completed",
                last_error_code="",
            ).order_by(ProvisioningJob.id.desc()).first()
        if not bootstrap:
            bootstrap_status = (
                "established" if active_accounts else "not issued"
            )
        elif bootstrap.accepted_at:
            bootstrap_status = "accepted"
        elif bootstrap.disabled_at:
            bootstrap_status = "revoked"
        else:
            comparison_now = (
                now.replace(tzinfo=None)
                if bootstrap.expires_at.tzinfo is None else now
            )
            bootstrap_status = (
                "expired" if bootstrap.expires_at <= comparison_now else "unused"
            )
        rows.append({
            "unit": unit,
            "active_accounts": active_accounts,
            "flags": flags,
            "database_health": routing.health if routing else "unknown",
            "provisioning_state": (
                routing.provisioning_state if routing else "pending"
            ),
            "provisioning_error": (
                routing.last_error_code if routing else ""
            ),
            "migration_version": routing.migration_version if routing else "",
            "storage_bytes": routing.storage_bytes if routing else 0,
            "activity_count": int(activity or 0),
            "bootstrap_status": bootstrap_status,
            "provisioning_job": latest_job,
        })
    return render_template(
        "platform_admin.html", rows=rows,
        feature_keys=sorted(PLATFORM_FEATURE_FLAGS),
    )


@app.get("/platform/worker-health")
@login_required
def platform_worker_health():
    if getattr(current_user, "role", "") != "superadmin":
        abort(403)
    cutoff = utcnow() - timedelta(
        seconds=max(
            60,
            int(os.environ.get("ATCROSTER_PROVISIONING_LEASE_SECONDS", "120"))
            * 2,
        )
    )
    active = WorkerHeartbeat.query.filter(
        WorkerHeartbeat.last_seen_at >= cutoff
    ).count()
    stale = WorkerHeartbeat.query.filter(
        WorkerHeartbeat.last_seen_at < cutoff
    ).count()
    return jsonify({
        "status": "ready" if active else "unavailable",
        "active_workers": active,
        "stale_workers": stale,
    }), 200 if active else 503


@app.route("/unit/accounts", methods=["GET", "POST"])
@login_required
def unit_accounts():
    if not is_admin_user(current_user):
        abort(403)
    unit_id = _current_unit_id()
    unit = db.session.get(Unit, unit_id)
    if not unit:
        abort(404)
    if request.method == "POST":
        _validate_csrf()
        action = (request.form.get("action") or "").strip()
        if action == "create_invitation":
            role = (request.form.get("role") or "StaffUser").strip()
            allowed_roles = {
                "UnitAdmin", "RosterEditor", "WatchManager",
                "StaffUser", "ReadOnlyAuditor",
            }
            if role not in allowed_roles:
                abort(400, "Invalid invitation role.")
            try:
                person_id = int(request.form.get("person_id") or 0)
            except ValueError:
                abort(400, "Invalid roster person.")
            person = Staff.query.filter_by(
                id=person_id, unit_id=unit_id
            ).first()
            if not person:
                flash(
                    "Select an existing roster person before issuing access.",
                    "error",
                )
                return redirect(url_for("unit_accounts"))
            if UnitMembership.query.filter_by(
                unit_id=unit_id, person_id=person.id
            ).filter(
                UnitMembership.status.in_(("active", "invited"))
            ).first():
                flash(
                    "That roster person already has account access or a pending membership.",
                    "error",
                )
                return redirect(url_for("unit_accounts"))
            existing_invitation = SecureInvitation.query.filter_by(
                unit_id=unit_id, target_person_id=person.id,
                accepted_at=None, disabled_at=None,
            ).first()
            if existing_invitation:
                flash(
                    "That roster person already has a pending invitation. "
                    "Disable it before issuing another.",
                    "error",
                )
                return redirect(url_for("unit_accounts"))
            try:
                from account_limits import lock_unit_capacity
                lock_unit_capacity(db, Unit, UnitMembership, unit_id)
                raw_token = secrets.token_urlsafe(32)
                invitation = SecureInvitation(
                    unit_id=unit_id,
                    token_digest=hashlib.sha256(
                        raw_token.encode()
                    ).hexdigest(),
                    role=role, target_person_id=person.id,
                    expires_at=utcnow() + timedelta(days=7),
                )
                db.session.add(invitation)
                db.session.commit()
            except ValueError as exc:
                db.session.rollback()
                flash(str(exc), "error")
                return redirect(url_for("unit_accounts"))
            invite_url = url_for(
                "accept_invitation", token=raw_token, _external=True
            )
            flash(
                f"Invitation for {person.name} created. Copy this link now; it is shown only "
                f"once: {invite_url}",
                "ok",
            )
            return redirect(url_for("unit_accounts"))
        if action == "create_account":
            name = (request.form.get("name") or "").strip()
            username = _normalized_login(
                request.form.get("username") or ""
            )
            password = request.form.get("password") or ""
            if not name or not username or len(password) < 12:
                flash("Name, username, and a 12-character password are required.", "error")
                return redirect(url_for("unit_accounts"))
            central_duplicate = PlatformIdentity.query.filter(
                db.func.lower(PlatformIdentity.username) == username
            ).first()
            local_duplicate = Staff.query.filter(
                db.func.lower(Staff.username) == username
            ).first()
            if central_duplicate or local_duplicate:
                flash("That login identifier is unavailable.", "error")
                return redirect(url_for("unit_accounts"))
            identity = None
            staff = None
            try:
                password_hash = generate_password_hash(password)
                identity = PlatformIdentity(
                    public_id=f"member-{secrets.token_hex(12)}",
                    username=username, password_hash=password_hash,
                )
                db.session.add(identity)
                db.session.commit()
                staff = Staff(
                    unit_id=unit_id, username=username, name=name,
                    staff_no=f"{unit.code}-LOGIN-{secrets.token_hex(3).upper()}",
                    role="user", is_operational=False,
                    membership_status="pending",
                )
                staff.password_hash = password_hash
                db.session.add(staff)
                db.session.commit()
                membership = UnitMembership(
                    identity_id=identity.id, unit_id=unit_id,
                    person_id=staff.id, role="StaffUser", status="invited",
                )
                db.session.add(membership)
                db.session.flush()
                from account_limits import activate_membership
                activate_membership(db, Unit, UnitMembership, membership.id)
                membership.activated_at = utcnow()
                staff.membership_status = "active"
                db.session.commit()
                flash("Account activated.", "ok")
            except (ValueError, IntegrityError) as exc:
                db.session.rollback()
                if staff and staff.id:
                    pending_staff = db.session.get(Staff, staff.id)
                    if pending_staff and pending_staff.membership_status != "active":
                        db.session.delete(pending_staff)
                        db.session.commit()
                if identity and identity.id:
                    orphan = db.session.get(PlatformIdentity, identity.id)
                    has_membership = UnitMembership.query.filter_by(
                        identity_id=identity.id
                    ).first()
                    if orphan and not has_membership:
                        db.session.delete(orphan)
                        db.session.commit()
                message = (
                    str(exc) if isinstance(exc, ValueError)
                    else "That login identifier is unavailable."
                )
                flash(message, "error")
            return redirect(url_for("unit_accounts"))
        if action == "deactivate":
            membership_id = int(request.form.get("membership_id") or 0)
            membership = UnitMembership.query.filter_by(
                id=membership_id, unit_id=unit_id, status="active"
            ).first_or_404()
            if membership.person_id == current_user.id:
                flash("You cannot deactivate your own account.", "error")
            else:
                membership.status = "suspended"
                membership.suspended_at = utcnow()
                linked = (
                    tenant_get(Staff, membership.person_id)
                    if membership.person_id else None
                )
                if linked:
                    linked.membership_status = "suspended"
                db.session.commit()
                flash("Account deactivated.", "ok")
            return redirect(url_for("unit_accounts"))
        if action == "restore":
            try:
                membership_id = int(
                    request.form.get("membership_id") or 0
                )
            except ValueError:
                abort(400)
            membership = UnitMembership.query.filter_by(
                id=membership_id, unit_id=unit_id, status="suspended"
            ).first_or_404()
            try:
                from account_limits import activate_membership
                activate_membership(
                    db, Unit, UnitMembership, membership.id
                )
                membership.suspended_at = None
                membership.activated_at = (
                    membership.activated_at or utcnow()
                )
                linked = (
                    tenant_get(Staff, membership.person_id)
                    if membership.person_id else None
                )
                if linked:
                    linked.membership_status = "active"
                db.session.commit()
                flash("Account restored.", "ok")
            except ValueError as exc:
                db.session.rollback()
                flash(str(exc), "error")
            return redirect(url_for("unit_accounts"))
        if action == "disable_invitation":
            try:
                invitation_id = int(
                    request.form.get("invitation_id") or 0
                )
            except ValueError:
                abort(400)
            invitation = SecureInvitation.query.filter_by(
                id=invitation_id, unit_id=unit_id,
                accepted_at=None, disabled_at=None,
            ).first_or_404()
            invitation.disabled_at = utcnow()
            invitation.active_bootstrap_key = None
            db.session.commit()
            flash("Invitation disabled.", "ok")
            return redirect(url_for("unit_accounts"))
        abort(400)
    memberships = UnitMembership.query.filter_by(unit_id=unit_id).order_by(
        UnitMembership.id
    ).all()
    active_count = sum(1 for row in memberships if row.status == "active")
    current_time = utcnow()
    pending_invitations = SecureInvitation.query.filter(
        SecureInvitation.unit_id == unit_id,
        SecureInvitation.accepted_at.is_(None),
        SecureInvitation.disabled_at.is_(None),
        SecureInvitation.expires_at > current_time,
    ).order_by(SecureInvitation.expires_at).all()
    unavailable_person_ids = {
        row.person_id for row in memberships
        if row.person_id and row.status in {"active", "invited"}
    } | {
        row.target_person_id for row in pending_invitations
        if row.target_person_id
    }
    roster_people = Staff.query.filter_by(
        unit_id=unit_id
    ).order_by(Staff.name).all()
    eligible_people = [
        person for person in roster_people
        if person.id not in unavailable_person_ids
    ]
    return render_template(
        "unit_accounts.html", unit=unit, memberships=memberships,
        active_count=active_count,
        pending_invitations=pending_invitations,
        eligible_people=eligible_people,
        staff_by_id={person.id: person for person in roster_people},
    )


class SignupWorkflowError(RuntimeError):
    pass


def _normalized_login(value: str) -> str:
    return value.strip().casefold()


def _run_invitation_signup(
    invitation, unit, name, username, password, email="", fail_after=None,
):
    """Resume an invitation saga without claiming cross-DB atomicity."""
    # Every airport begins with a dependable rest-day code, even when it was
    # provisioned after the original application seed ran.
    off_shift = ShiftType.query.filter_by(unit_id=unit.id, code="OFF").first()
    if not off_shift:
        db.session.add(ShiftType(
            unit_id=unit.id,
            code="OFF",
            name="Day off",
            is_working=False,
            is_active=True,
            is_requestable=False,
        ))
        db.session.commit()
    normalized = _normalized_login(username)
    workflow = SignupWorkflow.query.filter_by(
        invitation_id=invitation.id
    ).first()
    if not workflow:
        workflow = SignupWorkflow(
            invitation_id=invitation.id,
            idempotency_key=hashlib.sha256(
                f"signup:{invitation.id}:{invitation.token_digest}".encode()
            ).hexdigest(),
            normalized_username=normalized,
            state="pending",
        )
        db.session.add(workflow)
        try:
            db.session.commit()
        except IntegrityError as exc:
            db.session.rollback()
            workflow = SignupWorkflow.query.filter_by(
                invitation_id=invitation.id
            ).first()
            if not workflow:
                raise SignupWorkflowError(
                    "Account setup could not be started safely."
                ) from exc
    if workflow.normalized_username != normalized:
        retryable_username_validation = (
            workflow.state == "failed"
            and workflow.compensation_state == "pending"
            and not workflow.identity_id
            and workflow.last_error_code == "validation_failed"
        )
        if not retryable_username_validation:
            raise SignupWorkflowError(
                "This invitation already has an incomplete setup attempt."
            )
        # No central identity or operational account exists yet. The previous
        # username was rejected before any durable account data was created,
        # so the same invitation can safely be corrected and retried.
        workflow.normalized_username = normalized
        workflow.updated_at = utcnow()
        db.session.commit()
    if workflow.state == "completed":
        return workflow
    if workflow.state == "failed" and workflow.compensation_state:
        workflow.state = workflow.compensation_state
    workflow.attempt_count = int(workflow.attempt_count or 0) + 1
    workflow.last_error_code = ""
    workflow.updated_at = utcnow()
    db.session.commit()
    try:
        if workflow.state == "pending":
            duplicate = PlatformIdentity.query.filter(
                db.func.lower(PlatformIdentity.username) == normalized
            ).first()
            if duplicate:
                raise SignupWorkflowError(
                    "That login identifier is unavailable."
                )
            identity = PlatformIdentity(
                public_id=f"member-{secrets.token_hex(12)}",
                username=normalized,
                password_hash=generate_password_hash(password),
                email=_valid_email(email),
            )
            db.session.add(identity)
            try:
                db.session.commit()
            except IntegrityError as exc:
                db.session.rollback()
                raise SignupWorkflowError(
                    "That login identifier is unavailable."
                ) from exc
            workflow = db.session.get(SignupWorkflow, workflow.id)
            workflow.identity_id = identity.id
            workflow.state = "identity_created"
            workflow.updated_at = utcnow()
            db.session.commit()
            if fail_after == "identity_created":
                raise RuntimeError("injected_identity_created")
        if workflow.state == "identity_created":
            role_map = {
                "UnitAdmin": "admin", "RosterEditor": "editor",
                "WatchManager": "user", "StaffUser": "user",
                "ReadOnlyAuditor": "auditor",
            }
            if invitation.target_person_id:
                staff = Staff.query.filter_by(
                    id=invitation.target_person_id, unit_id=unit.id
                ).first()
                if not staff:
                    raise SignupWorkflowError(
                        "The linked roster person is no longer available."
                    )
                duplicate_staff = Staff.query.filter(
                    Staff.unit_id == unit.id,
                    db.func.lower(Staff.username) == normalized,
                    Staff.id != staff.id,
                ).first()
                if duplicate_staff:
                    raise SignupWorkflowError(
                        "That login identifier is unavailable."
                    )
                staff.username = normalized
                staff.email = _valid_email(email) or staff.email
                staff.role = role_map[invitation.role]
                staff.is_wm = invitation.role == "WatchManager"
                staff.set_password(password)
                staff.membership_status = "pending"
                db.session.commit()
            else:
                marker = f"{unit.code}-SIGNUP-{workflow.id}"
                staff = Staff.query.filter_by(staff_no=marker).first()
            if not staff:
                if Staff.query.filter(
                    db.func.lower(Staff.username) == normalized
                ).first():
                    raise SignupWorkflowError(
                        "That login identifier is unavailable."
                    )
                staff = Staff(
                    unit_id=unit.id, username=normalized, name=name[:80],
                    staff_no=marker, role=role_map[invitation.role],
                    is_wm=invitation.role == "WatchManager",
                    is_operational=False, membership_status="pending",
                    email=_valid_email(email),
                )
                staff.set_password(password)
                db.session.add(staff)
                try:
                    db.session.commit()
                except IntegrityError as exc:
                    db.session.rollback()
                    raise SignupWorkflowError(
                        "That login identifier is unavailable."
                    ) from exc
            workflow = db.session.get(SignupWorkflow, workflow.id)
            workflow.operational_person_id = staff.id
            workflow.state = "operational_account_created"
            workflow.updated_at = utcnow()
            db.session.commit()
            if fail_after == "operational_account_created":
                raise RuntimeError("injected_operational_account_created")
        if workflow.state == "operational_account_created":
            membership = UnitMembership.query.filter_by(
                identity_id=workflow.identity_id, unit_id=unit.id
            ).first()
            if not membership:
                membership = UnitMembership(
                    identity_id=workflow.identity_id, unit_id=unit.id,
                    person_id=workflow.operational_person_id,
                    role=invitation.role, status="invited",
                )
                db.session.add(membership)
                db.session.flush()
                from account_limits import activate_membership
                activate_membership(
                    db, Unit, UnitMembership, membership.id
                )
                membership.activated_at = utcnow()
                db.session.commit()
            workflow = db.session.get(SignupWorkflow, workflow.id)
            workflow.membership_id = membership.id
            workflow.state = "membership_created"
            workflow.updated_at = utcnow()
            db.session.commit()
            if fail_after == "membership_created":
                raise RuntimeError("injected_membership_created")
        if workflow.state == "membership_created":
            staff = db.session.get(
                Staff, workflow.operational_person_id
            )
            if not staff:
                raise SignupWorkflowError(
                    "Operational account requires reconciliation."
                )
            staff.membership_status = "active"
            db.session.commit()
            workflow = db.session.get(SignupWorkflow, workflow.id)
            invitation = db.session.get(
                SecureInvitation, workflow.invitation_id
            )
            invitation.accepted_at = utcnow()
            invitation.active_bootstrap_key = None
            if invitation.role == "UnitAdmin":
                unit.status = "active"
                routing = db.session.get(
                    DatabaseRoutingMetadata, unit.id
                )
                routing.provisioning_state = "active"
            workflow.state = "completed"
            workflow.compensation_state = ""
            workflow.updated_at = utcnow()
            db.session.commit()
        return workflow
    except SignupWorkflowError:
        db.session.rollback()
        workflow = db.session.get(SignupWorkflow, workflow.id)
        workflow.compensation_state = workflow.state
        workflow.state = "failed"
        workflow.last_error_code = "validation_failed"
        workflow.updated_at = utcnow()
        db.session.commit()
        raise
    except Exception as exc:
        db.session.rollback()
        workflow = db.session.get(SignupWorkflow, workflow.id)
        workflow.compensation_state = workflow.state
        workflow.state = "failed"
        workflow.last_error_code = (
            str(exc) if str(exc).startswith("injected_")
            else "stage_interrupted"
        )
        workflow.updated_at = utcnow()
        db.session.commit()
        raise SignupWorkflowError(
            "Account setup was interrupted safely. Retry this invitation "
            "or ask an administrator to reconcile it."
        ) from exc


@app.route("/invite/<token>", methods=["GET", "POST"])
def accept_invitation(token):
    """Accept a one-time, expiring invitation without trusting tenant input."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token or ""):
        abort(404)
    digest = hashlib.sha256(token.encode()).hexdigest()
    if not _consume_rate_limit(
        "invitation-acceptance", digest, limit=20,
        window=timedelta(hours=1),
    ):
        abort(429, "Too many invitation attempts.")
    invitation = SecureInvitation.query.filter_by(
        token_digest=digest
    ).first_or_404()
    expiry_now = utcnow()
    if invitation.expires_at.tzinfo is None:
        expiry_now = expiry_now.replace(tzinfo=None)
    if (
        invitation.accepted_at
        or invitation.disabled_at
        or invitation.expires_at <= expiry_now
    ):
        abort(410, "This invitation has expired or has already been used.")
    unit = db.session.get(Unit, invitation.unit_id)
    routing = (
        db.session.get(DatabaseRoutingMetadata, invitation.unit_id)
        if unit else None
    )
    if (
        not unit
        or unit.status not in {"active", "provisioning"}
        or (
            unit.status == "provisioning"
            and (
                not routing
                or routing.provisioning_state != "invitation_issued"
            )
        )
    ):
        abort(409, "This airport account is not accepting invitations.")
    if DEPLOYMENT_ENV == "production" and not routing:
        abort(503, "Operational database routing is unavailable.")
    # A targeted invitation refers to a person in the airport's operational
    # database. Establish that trusted route before resolving or displaying
    # the roster profile, including on the initial anonymous GET.
    g.tenant_context_token = bind_authenticated_unit(
        invitation.unit_id,
        routing.secret_name if routing else None,
    )
    target_person = None
    if invitation.target_person_id:
        target_person = Staff.query.filter_by(
            id=invitation.target_person_id,
            unit_id=invitation.unit_id,
        ).first()
        if not target_person:
            abort(410, "The linked roster person is no longer available.")
    if request.method == "POST":
        _validate_csrf()
        name = (
            target_person.name
            if target_person
            else (request.form.get("name") or "").strip()
        )
        username = (request.form.get("username") or "").strip().lower()
        email = _valid_email(request.form.get("email") or "")
        password = request.form.get("password") or ""
        if not name or not re.fullmatch(r"[a-z0-9._-]{3,120}", username):
            flash("Enter a name and a valid username.", "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit,
                target_person=target_person,
            ), 400
        if len(password) < 12:
            flash("Use a password of at least 12 characters.", "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit,
                target_person=target_person,
            ), 400
        if not email:
            flash("Enter a valid email address.", "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit,
                target_person=target_person,
            ), 400
        try:
            from signup_locking import invitation_signup_lock

            with invitation_signup_lock(db, invitation.id):
                locked_invitation = SecureInvitation.query.filter_by(
                    id=invitation.id,
                    accepted_at=None,
                    disabled_at=None,
                ).with_for_update().first()
                if not locked_invitation:
                    abort(410, "This invitation has already been used.")
                _run_invitation_signup(
                    locked_invitation, unit, name, username, password,
                    email=email,
                )
        except (SignupWorkflowError, ValueError) as exc:
            flash(str(exc), "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit,
                target_person=target_person,
            ), 409
        flash("Account created. Sign in and configure MFA.", "ok")
        return redirect(url_for("login"))
    return render_template(
        "invitation_accept.html", invitation=invitation, unit=unit,
        target_person=target_person,
    )


@app.route("/unit/onboarding", methods=["GET", "POST"])
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
            unit.timezone = (request.form.get("timezone") or unit.timezone).strip()[:64]
            unit.locale = (request.form.get("locale") or unit.locale).strip()[:20]
            unit.date_format = (request.form.get("date_format") or unit.date_format).strip()[:30]
            primary = request.form.get("primary_colour") or "#16283a"
            accent = request.form.get("accent_colour") or "#2c7be5"
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", primary) or not re.fullmatch(
                r"#[0-9A-Fa-f]{6}", accent
            ):
                abort(400, "Brand colours must be six-digit hex values.")
            unit.branding_json = json.dumps({
                "primary_colour": primary,
                "accent_colour": accent,
                "display_name": (
                    request.form.get("display_name") or unit.name
                ).strip()[:120],
            }, sort_keys=True)
            unit.onboarding_step = max(unit.onboarding_step, 2)
            db.session.commit()
            flash("Airport identity and branding saved.", "ok")
            return redirect(url_for("unit_onboarding"))
        if action == "request_rules":
            try:
                months = int(request.form.get("request_months_ahead") or 3)
                lock_day = int(request.form.get("request_lock_day") or 20)
            except ValueError:
                abort(400, "Request rules must be whole numbers.")
            if not 1 <= months <= 24 or not 1 <= lock_day <= 28:
                abort(400, "Request window must be 1–24 months and lock day 1–28.")
            unit.request_months_ahead = months
            unit.request_lock_day = lock_day
            unit.onboarding_step = max(unit.onboarding_step, 8)
            db.session.commit()
            flash("Fatigue and request rules saved.", "ok")
            return redirect(url_for("unit_onboarding"))
        if action == "seed_qualifications":
            defaults = (
                "MEDICAL", "ADI", "APP", "APS", "OJTI", "UCA",
                "ENGLISH_LANGUAGE",
            )
            for code in defaults:
                if not QualificationType.query.filter_by(
                    unit_id=unit.id, code=code
                ).first():
                    db.session.add(QualificationType(
                        unit_id=unit.id, code=code,
                        label=code.replace("_", " ").title(),
                        warning_days_csv="180,90,60,30",
                    ))
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
                if staff_no.lower() in seen_numbers or Staff.query.filter_by(
                    unit_id=unit.id, staff_no=staff_no
                ).first():
                    line_errors.append("duplicate staff_no")
                if not watch:
                    line_errors.append("unknown watch")
                if line_errors:
                    errors.append(
                        f"Line {line_number}: {', '.join(line_errors)}"
                    )
                    continue
                seen_numbers.add(staff_no.lower())
                rows.append({
                    "name": name[:80], "staff_no": staff_no,
                    "watch_id": watch.id, "watch": watch.name,
                })
            nonce = secrets.token_urlsafe(18)
            if not errors:
                session["_onboarding_csv_preview"] = {
                    "unit_id": unit.id, "nonce": nonce, "rows": rows,
                }
            else:
                session.pop("_onboarding_csv_preview", None)
            csv_preview = {
                "rows": rows, "errors": errors, "nonce": nonce,
            }
        elif action == "csv_apply":
            saved = session.get("_onboarding_csv_preview") or {}
            nonce = request.form.get("nonce") or ""
            if (
                saved.get("unit_id") != unit.id
                or not secrets.compare_digest(
                    nonce, str(saved.get("nonce") or "")
                )
            ):
                abort(409, "The import preview has expired.")
            for row in saved.get("rows") or []:
                person = Staff(
                    unit_id=unit.id,
                    username=(
                        f"person-{unit.code.lower()}-"
                        f"{secrets.token_hex(8)}"
                    ),
                    name=row["name"], staff_no=row["staff_no"],
                    watch_id=row["watch_id"], role="user",
                    membership_status="no_login", is_operational=True,
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
    active = UnitMembership.query.filter_by(unit_id=unit.id, status="active").count()
    pending = SecureInvitation.query.filter_by(
        unit_id=unit.id, accepted_at=None, disabled_at=None
    ).count()
    readiness = [
        ("Airport identity", bool(unit.name and unit.code and unit.timezone), "unit_onboarding"),
        ("Watches configured", Watch.query.count() > 0, "admin"),
        ("Active shifts configured", ShiftType.query.filter_by(is_active=True).count() > 0, "admin"),
        ("Operational staff added", Staff.query.filter_by(is_operational=True).count() > 0, "admin"),
        ("Staffing requirements set", Requirement.query.count() > 0, "admin"),
        ("Qualification types set", QualificationType.query.count() > 0, "qualification_compliance"),
        ("Roster warning rules available", True, "admin_fatigue_rules"),
        ("Unit Admin access active", active > 0, "unit_accounts"),
    ]
    readiness_complete = sum(1 for _, complete, _ in readiness if complete)
    return render_template(
        "unit_onboarding.html", unit=unit, active_accounts=active,
        pending_invitations=pending, readiness=readiness,
        readiness_complete=readiness_complete,
        readiness_percent=round(readiness_complete / len(readiness) * 100),
        csv_preview=csv_preview,
    )


def _qualification_snapshot(record: PersonQualification) -> dict:
    return {
        "person_id": record.person_id,
        "qualification_type_id": record.qualification_type_id,
        "issued_on": record.issued_on,
        "valid_from": record.valid_from,
        "expires_on": record.expires_on,
        "status": record.status,
    }


def _record_qualification_history(
    record: PersonQualification, action: str
) -> None:
    db.session.add(PersonQualificationHistory(
        unit_id=record.unit_id,
        person_qualification_id=record.id,
        actor_id=current_user.id,
        action=action,
        snapshot_json=json.dumps(
            _qualification_snapshot(record), default=str, sort_keys=True
        ),
    ))


def _sync_qualification_to_roster_profile(
    person: Staff, qtype: QualificationType, expires_on: date | None
) -> None:
    legacy_field = {
        "MEDICAL": "medical_expiry", "ADI": "tower_ue_expiry",
        "APS": "radar_ue_expiry", "MET": "met_ue_expiry",
    }.get(qtype.code)
    if legacy_field:
        setattr(person, legacy_field, expires_on)


@app.route("/compliance", methods=["GET", "POST"])
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
                abort(400, "Configure at least one warning period from 0–3650 days.")
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
                if qtype.code != code and PersonQualification.query.filter_by(
                    unit_id=unit_id, qualification_type_id=qtype.id
                ).first():
                    abort(409, "A used qualification code cannot be changed.")
                qtype.code = code
            qtype.label = label[:100]
            qtype.warning_days_csv = ",".join(str(value) for value in warnings)
            qtype.expiry_required = (
                request.form.get("expiry_required") == "yes"
            )
            qtype.is_active = request.form.get("is_active") == "yes"
            db.session.commit()
            flash("Qualification type saved.", "ok")
            return redirect(url_for("qualification_compliance"))
        if action == "save_person":
            person = Staff.query.filter_by(
                id=int(request.form.get("person_id") or 0),
                unit_id=unit_id, is_operational=True,
            ).first_or_404()
            qtype = QualificationType.query.filter_by(
                id=int(request.form.get("type_id") or 0),
                unit_id=unit_id, is_active=True,
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
            except ValueError:
                abort(400, "Qualification dates must be valid ISO dates.")
            if qtype.expiry_required and status == "valid" and not expires_on:
                abort(400, "This qualification requires an expiry date.")
            record = PersonQualification.query.filter_by(
                unit_id=unit_id, person_id=person.id,
                qualification_type_id=qtype.id,
            ).first()
            action_name = "renewed" if record else "assigned"
            if not record:
                record = PersonQualification(
                    unit_id=unit_id, person_id=person.id,
                    qualification_type_id=qtype.id,
                )
                db.session.add(record)
                db.session.flush()
            record.issued_on = issued_on
            record.valid_from = valid_from
            record.expires_on = expires_on
            record.status = status
            record.updated_at = utcnow()
            _sync_qualification_to_roster_profile(
                person, qtype, expires_on
            )
            _record_qualification_history(record, action_name)
            db.session.commit()
            flash("Person qualification saved.", "ok")
            return redirect(url_for("qualification_compliance"))
        if action == "import_preview":
            upload = request.files.get("csv_file")
            if not upload:
                abort(400, "Choose a qualification CSV file.")
            try:
                reader = csv.DictReader(io.StringIO(
                    upload.read().decode("utf-8-sig")
                ))
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
                            if (raw.get(key) or "").strip() else None
                        )
                        for key in ("issued_on", "valid_from", "expires_on")
                    }
                except ValueError:
                    errors.append(f"Line {line}: invalid date.")
                    continue
                if not person or not qtype or status not in {
                    "valid", "suspended", "revoked", "inactive"
                }:
                    errors.append(f"Line {line}: unknown person/type/status.")
                    continue
                if qtype.expiry_required and status == "valid" and not parsed["expires_on"]:
                    errors.append(f"Line {line}: expiry is required.")
                    continue
                rows.append({
                    "person_id": person.id, "person": person.name,
                    "type_id": qtype.id, "type": qtype.code,
                    "status": status,
                    **{key: value.isoformat() if value else "" for key, value in parsed.items()},
                })
            nonce = secrets.token_urlsafe(18)
            if not errors:
                session["_qualification_import_preview"] = {
                    "unit_id": unit_id, "nonce": nonce, "rows": rows,
                }
            import_preview = {"rows": rows, "errors": errors, "nonce": nonce}
        elif action == "import_apply":
            saved = session.get("_qualification_import_preview") or {}
            if (
                saved.get("unit_id") != unit_id
                or not secrets.compare_digest(
                    request.form.get("nonce") or "",
                    saved.get("nonce") or "",
                )
            ):
                abort(409, "The qualification preview has expired.")
            for row in saved.get("rows") or []:
                record = PersonQualification.query.filter_by(
                    unit_id=unit_id, person_id=row["person_id"],
                    qualification_type_id=row["type_id"],
                ).first()
                if not record:
                    record = PersonQualification(
                        unit_id=unit_id, person_id=row["person_id"],
                        qualification_type_id=row["type_id"],
                    )
                    db.session.add(record)
                    db.session.flush()
                for key in ("issued_on", "valid_from", "expires_on"):
                    setattr(
                        record, key,
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
            db.session.commit()
            session.pop("_qualification_import_preview", None)
            flash("Qualification import applied.", "ok")
            return redirect(url_for("qualification_compliance"))
        else:
            abort(400, "Unknown qualification action.")
    today = date.today()
    qualification_types = QualificationType.query.filter_by(
        unit_id=unit_id
    ).order_by(QualificationType.code).all()
    people = Staff.query.filter_by(
        unit_id=unit_id, is_operational=True
    ).order_by(Staff.name).all()
    qualifications = PersonQualification.query.filter_by(
        unit_id=unit_id
    ).all()
    by_person_type = {
        (row.person_id, row.qualification_type_id): row
        for row in qualifications
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
            rows.append({
                "person": person, "type": qtype,
                "qualification": qual, "expires_on": expires_on,
                "days": days, "state": state,
            })
    history = PersonQualificationHistory.query.filter_by(
        unit_id=unit_id
    ).order_by(PersonQualificationHistory.occurred_at.desc()).limit(100).all()
    return render_template(
        "qualification_compliance.html",
        rows=rows, qualification_types=qualification_types, people=people,
        history=history, import_preview=import_preview,
    )


def _valid_endorsement(person_id: int, position_id: int, on_day: date) -> bool:
    row = PositionEndorsement.query.filter_by(
        person_id=person_id, position_id=position_id, status="valid"
    ).first()
    return bool(
        row and row.valid_from <= on_day
        and (row.valid_until is None or row.valid_until >= on_day)
    )


def _position_assurance(year: int, month: int) -> list[dict]:
    _, days = month_range(year, month)
    requirements = PositionRequirement.query.filter(
        PositionRequirement.day >= days[0],
        PositionRequirement.day <= days[-1],
    ).order_by(PositionRequirement.day, PositionRequirement.shift_code).all()
    positions = {
        row.id: row for row in OperationalPosition.query.all()
    }
    rows = []
    for requirement in requirements:
        assignments = Assignment.query.filter_by(
            day=requirement.day, code=requirement.shift_code
        ).all()
        eligible = [
            assignment for assignment in assignments
            if _valid_endorsement(
                assignment.staff_id, requirement.position_id, requirement.day
            )
        ]
        target = requirement.required_count + requirement.contingency_count
        rows.append({
            "requirement": requirement,
            "position": positions.get(requirement.position_id),
            "eligible": len(eligible),
            "target": target,
            "shortfall": max(0, target - len(eligible)),
        })
    return rows




@app.route("/admin/toil/new", methods=["GET", "POST"])
@login_required
@admin_required
def admin_toil_new():
    atcos = Staff.query.filter_by(
        is_operational=True).order_by(Staff.name.asc()).all()
    if request.method == "POST":
        _validate_csrf()
        try:
            sid = int(request.form["staff_id"])
            amount = float(request.form.get("amount", "0") or 0)
        except (KeyError, TypeError, ValueError):
            flash("Choose an ATCO and enter a valid adjustment.", "error")
            return redirect(url_for("admin_toil_new"))
        unit = request.form.get("unit", "days").lower()
        note = (request.form.get("note") or "").strip()
        s = Staff.query.filter_by(
            id=sid, unit_id=_current_unit_id()
        ).first_or_404()
        # Convert to half-days
        direction = -1 if request.form.get("direction") == "subtract" else 1
        if unit.startswith("day"):
            half = int(round(amount * 2))
        else:  # hours
            half = int(round((amount / 8.0) * 2))
        if amount <= 0 or half <= 0:
            flash("Enter an adjustment greater than zero.", "error")
            return redirect(url_for("admin_toil_new"))
        s.toil_half_days = int((s.toil_half_days or 0) + direction * half)
        db.session.commit()
        verb = "added to" if direction > 0 else "deducted from"
        flash(
            f"{amount:g} {unit} {verb} {s.name}'s TOIL balance.",
            "ok",
        )
        return redirect(url_for("admin_toil_new"))
    return render_template("admin_toil_new.html", atcos=atcos)

LOGIN_RATE_WINDOW = timedelta(minutes=15)
LOGIN_RATE_LIMIT = 10


def _login_rate_key(username: str) -> str:
    remote = request.remote_addr or "unknown"
    return privacy_key(
        str(app.config["SECRET_KEY"]), "login", remote, username.lower()
    )


def _consume_rate_limit(
    scope: str, subject: object, limit: int = LOGIN_RATE_LIMIT,
    window: timedelta = LOGIN_RATE_WINDOW, fail_closed: bool = True,
) -> bool:
    key = privacy_key(
        str(app.config["SECRET_KEY"]), scope,
        request.remote_addr or "unknown", subject,
    )
    try:
        return _rate_limiter.consume(
            key, limit, max(1, int(window.total_seconds()))
        )
    except LimiterUnavailable:
        _security_event("rate_limiter_unavailable", scope=scope)
        if fail_closed:
            abort(503, "Security service is temporarily unavailable.")
        return True


def _reset_rate_limit(scope: str, subject: object) -> None:
    key = privacy_key(
        str(app.config["SECRET_KEY"]), scope,
        request.remote_addr or "unknown", subject,
    )
    try:
        _rate_limiter.reset(key)
    except LimiterUnavailable:
        _security_event("rate_limiter_unavailable", scope=scope)


def _security_event(event: str, **safe_fields) -> None:
    payload = {
        "event": event,
        "request_id": getattr(g, "request_id", ""),
        "occurred_at": utcnow().isoformat(),
        **safe_fields,
    }
    app.logger.warning("security_event %s", json.dumps(
        payload, sort_keys=True, default=str
    ))


def _current_auth_stamp(user) -> str:
    """Bind a session to mutable authentication and authorisation state."""
    parts = [
        str(getattr(user, "password_hash", "")),
        str(getattr(user, "role", "")),
        str(getattr(user, "membership_status", "")),
    ]
    if getattr(user, "role", "") == "superadmin":
        credential = PlatformMfaCredential.query.filter_by(
            identity_id=user.id
        ).first()
    else:
        # Airport MFA lives in the operational database and is deliberately
        # excluded from this control-plane stamp. Its reset workflow revokes
        # the affected login directly; querying it here would make the stamp
        # dependent on request routing state.
        credential = None
    parts.extend([
        str(bool(credential and credential.enabled)),
        str(bool(getattr(credential, "reset_required", False))),
    ])
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _initialize_authenticated_session(user, *, platform_mfa=False) -> None:
    """Regenerate authenticated session state after the final auth factor."""
    session.permanent = True
    session["_session_nonce"] = secrets.token_urlsafe(24)
    session["_session_started_at"] = utcnow().isoformat()
    session["_last_seen_epoch"] = int(utcnow().timestamp())
    session["_auth_stamp"] = _current_auth_stamp(user)
    if platform_mfa:
        session["_platform_mfa_verified"] = True


def _central_security_event(
    event_type: str, outcome: str, identity_id: int | None = None,
    principal: str = "", detail: str = "",
) -> None:
    db.session.add(CentralSecurityAudit(
        identity_id=identity_id, event_type=event_type[:80],
        outcome=outcome[:20], principal_digest=principal[:32],
        safe_detail=detail[:200],
    ))


def _record_successful_login(user: Staff) -> None:
    now = utcnow()
    identity = PlatformIdentity.query.filter_by(
        username=user.username
    ).first()
    if identity:
        identity.last_active_at = now
    unit = db.session.get(Unit, user.unit_id)
    if unit:
        unit.last_active_at = now
    if user.role != "superadmin":
        db.session.add(AggregateUsageEvent(
            unit_id=user.unit_id, event_type="login", count=1,
        ))
    db.session.commit()


def _active_recovery_from_digest(
    field_name: str, raw_token: str, expected_state: str
):
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", raw_token or ""):
        abort(404)
    digest = hashlib.sha256(raw_token.encode()).hexdigest()
    field = getattr(RecoveryRequest, field_name)
    row = RecoveryRequest.query.filter(
        field == digest, RecoveryRequest.state == expected_state
    ).first_or_404()
    comparison_now = utcnow()
    if row.expires_at.tzinfo is None:
        comparison_now = comparison_now.replace(tzinfo=None)
    if row.expires_at <= comparison_now:
        abort(410, "This recovery link has expired.")
    return row


@app.route("/recover", methods=["GET", "POST"])
def account_recovery():
    mode = (request.values.get("mode") or "password").strip()
    if mode not in {"password", "username"}:
        mode = "password"
    if request.method == "POST":
        _validate_csrf()
        if not _consume_rate_limit(
            "account-recovery",
            hashlib.sha256(
                f"{request.remote_addr}:{mode}".encode()
            ).hexdigest(),
            limit=8,
            window=timedelta(hours=1),
        ):
            abort(429, "Too many recovery attempts. Try again later.")
        if mode == "username":
            email = _valid_email(request.form.get("email") or "")
            airport_code = (
                request.form.get("airport_code") or ""
            ).strip().upper()
            if email:
                identities = PlatformIdentity.query.filter(
                    db.func.lower(PlatformIdentity.email) == email
                ).order_by(PlatformIdentity.username).all()
                if identities:
                    usernames = "\n".join(
                        f"- {identity.username}" for identity in identities
                    )
                    _send_account_email(
                        email,
                        "Your ATCRoster username",
                        "The username(s) registered to this email address are:\n"
                        f"{usernames}\n\nIf you did not request this, contact "
                        "your Unit Administrator.",
                    )
                elif airport_code:
                    unit = Unit.query.filter_by(code=airport_code).first()
                    if unit:
                        for admin_email in _unit_admin_emails(unit.id):
                            _send_account_email(
                                admin_email,
                                f"Username recovery assistance for {unit.code}",
                                "A user requested username assistance for "
                                f"{email}. Verify their identity using your "
                                "approved local process before disclosing or "
                                "updating account details.",
                            )
        else:
            username = _normalized_login(
                request.form.get("username") or ""
            )
            identity = PlatformIdentity.query.filter_by(
                username=username
            ).first()
            if identity:
                membership = UnitMembership.query.filter_by(
                    identity_id=identity.id, status="active"
                ).first()
                unit_id = membership.unit_id if membership else None
                approvers = (
                    _platform_support_emails()
                    if identity.role == "superadmin"
                    or (membership and membership.role == "UnitAdmin")
                    else _unit_admin_emails(unit_id)
                )
                if not approvers:
                    approvers = _platform_support_emails()
                raw_token = secrets.token_urlsafe(32)
                row = RecoveryRequest(
                    unit_id=unit_id,
                    identity_id=identity.id,
                    person_id=membership.person_id if membership else None,
                    approval_token_digest=hashlib.sha256(
                        raw_token.encode()
                    ).hexdigest(),
                    state="pending_approval",
                    expires_at=utcnow() + timedelta(hours=24),
                )
                db.session.add(row)
                db.session.commit()
                approval_url = url_for(
                    "approve_account_recovery",
                    token=raw_token,
                    _external=True,
                )
                for approver in approvers:
                    _send_account_email(
                        approver,
                        "ATCRoster password reset approval required",
                        f"A password reset was requested for {username}.\n\n"
                        f"Review and approve it here:\n{approval_url}\n\n"
                        "The link expires in 24 hours. Do not forward it.",
                    )
        flash(
            "If the supplied details match an account, the recovery process "
            "has started. Check email or contact your administrator.",
            "ok",
        )
        return redirect(url_for("account_recovery", mode=mode))
    return render_template("account_recovery.html", mode=mode)


@app.route("/recover/approve/<token>", methods=["GET", "POST"])
@login_required
def approve_account_recovery(token):
    row = _active_recovery_from_digest(
        "approval_token_digest", token, "pending_approval"
    )
    permitted = (
        getattr(current_user, "role", "") == "superadmin"
        or (
            is_admin_user(current_user)
            and int(getattr(current_user, "unit_id", 0) or 0)
            == int(row.unit_id or 0)
        )
    )
    if not permitted:
        abort(403)
    identity = db.session.get(PlatformIdentity, row.identity_id)
    if not identity:
        abort(410, "The account is no longer available.")
    if request.method == "POST":
        _validate_csrf()
        if not identity.email:
            flash(
                "This account has no registered email. Add and verify an "
                "email address before approving the reset.",
                "error",
            )
            return redirect(request.url)
        raw_reset = secrets.token_urlsafe(32)
        row.reset_token_digest = hashlib.sha256(
            raw_reset.encode()
        ).hexdigest()
        row.state = "reset_sent"
        row.approved_at = utcnow()
        row.expires_at = utcnow() + timedelta(hours=1)
        reset_url = url_for(
            "complete_account_recovery",
            token=raw_reset,
            _external=True,
        )
        delivered = _send_account_email(
            identity.email,
            "Reset your ATCRoster password",
            "Your password reset was approved.\n\n"
            f"Choose a new password here:\n{reset_url}\n\n"
            "The link expires in one hour and can only be used once.",
        )
        if not delivered:
            db.session.rollback()
            flash(
                "The reset email could not be delivered. Check the SMTP "
                "configuration and the registered email address.",
                "error",
            )
            return redirect(request.url)
        db.session.commit()
        flash("Reset approved and emailed to the account holder.", "ok")
        return redirect(url_for("index"))
    return render_template(
        "recovery_approve.html", recovery=row, identity=identity
    )


@app.route("/recover/reset/<token>", methods=["GET", "POST"])
def complete_account_recovery(token):
    row = _active_recovery_from_digest(
        "reset_token_digest", token, "reset_sent"
    )
    identity = db.session.get(PlatformIdentity, row.identity_id)
    if not identity:
        abort(410, "The account is no longer available.")
    if request.method == "POST":
        _validate_csrf()
        password = request.form.get("password") or ""
        confirmation = request.form.get("password_confirmation") or ""
        if len(password) < 12:
            flash("Use a password of at least 12 characters.", "error")
        elif password != confirmation:
            flash("The password confirmation does not match.", "error")
        else:
            password_hash = generate_password_hash(password)
            identity.password_hash = password_hash
            membership = UnitMembership.query.filter_by(
                identity_id=identity.id, status="active"
            ).first()
            if membership and membership.person_id:
                routing = db.session.get(
                    DatabaseRoutingMetadata, membership.unit_id
                )
                g.tenant_context_token = bind_authenticated_unit(
                    membership.unit_id,
                    routing.secret_name if routing else None,
                )
                person = db.session.get(Staff, membership.person_id)
                if person:
                    person.password_hash = password_hash
            row.state = "completed"
            row.completed_at = utcnow()
            row.reset_token_digest = None
            db.session.commit()
            flash(
                "Password updated. Sign in with your new password.", "ok"
            )
            return redirect(url_for("login"))
    return render_template("recovery_reset.html")


def _decrypt_mfa_secret(credential) -> str:
    try:
        return _decrypt_field(credential.encrypted_secret)
    except ValueError as exc:
        raise RuntimeError("MFA credential cannot be decrypted.") from exc


def _matching_totp_step(secret: str, code: str) -> int | None:
    totp = pyotp.TOTP(secret)
    now = utcnow()
    for offset in (-1, 0, 1):
        candidate_time = now + timedelta(seconds=offset * 30)
        candidate = totp.at(candidate_time)
        if secrets.compare_digest(candidate, code):
            return int(candidate_time.timestamp() // 30)
    return None


def _pending_platform_login():
    identity_id = int(session.get("_platform_mfa_identity_id") or 0)
    user_id = int(session.get("_platform_mfa_user_id") or 0)
    if not identity_id or user_id != identity_id:
        return None, None
    identity = db.session.get(PlatformIdentity, identity_id)
    if not identity or identity.role != "superadmin":
        return None, None
    return identity, identity


def _complete_platform_login(identity, user, recovery_used=False):
    next_url = session.get("_platform_mfa_next", "")
    session.clear()
    login_user(user)
    _initialize_authenticated_session(user, platform_mfa=True)
    identity.last_active_at = utcnow()
    _central_security_event(
        "platform_recovery_code_used" if recovery_used
        else "platform_mfa_verified",
        "success", identity.id,
        hashlib.sha256(identity.username.lower().encode()).hexdigest()[:16],
    )
    db.session.commit()
    return redirect(_canonical_login_redirect(
        next_url,
        default_endpoint="platform_admin",
        user_id=user.id,
    ))


def _totp_qr_data_uri(provisioning_uri: str) -> str:
    """Render a TOTP URI locally so MFA secrets never leave the application."""
    qr_buffer = io.BytesIO()
    qrcode.make(
        provisioning_uri,
        image_factory=qrcode.image.svg.SvgPathImage,
        box_size=8,
        border=4,
    ).save(qr_buffer)
    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(qr_buffer.getvalue()).decode("ascii")
    )


@app.route("/login/platform-mfa/setup", methods=["GET", "POST"])
def platform_mfa_setup():
    identity, user = _pending_platform_login()
    if not identity or not user:
        session.clear()
        return redirect(url_for("login"))
    existing = PlatformMfaCredential.query.filter_by(
        identity_id=identity.id, enabled=True, reset_required=False,
    ).first()
    if existing:
        return redirect(url_for("platform_mfa_challenge"))
    pending = session.get("_pending_platform_mfa_secret")
    if not pending:
        pending = pyotp.random_base32()
        session["_pending_platform_mfa_secret"] = pending
    provisioning_uri = pyotp.TOTP(pending).provisioning_uri(
        name=identity.username, issuer_name="ATCRoster Platform"
    )
    qr_data_uri = _totp_qr_data_uri(provisioning_uri)
    if request.method == "POST":
        _validate_csrf()
        if not _consume_rate_limit(
            "platform-mfa-enrolment", identity.id, limit=10,
            window=timedelta(minutes=15),
        ):
            abort(429)
        code = re.sub(r"\s", "", request.form.get("code") or "")
        if not pyotp.TOTP(pending).verify(code, valid_window=1):
            _central_security_event(
                "platform_mfa_enrolment", "denied", identity.id
            )
            db.session.commit()
            flash("The verification code is not valid.", "error")
            return redirect(url_for("platform_mfa_setup"))
        recovery_codes = [secrets.token_hex(5).upper() for _ in range(10)]
        credential = PlatformMfaCredential.query.filter_by(
            identity_id=identity.id
        ).first()
        if not credential:
            credential = PlatformMfaCredential(
                identity_id=identity.id, encrypted_secret=""
            )
            db.session.add(credential)
        credential.encrypted_secret = _encrypt_field(pending)
        credential.enabled = True
        credential.reset_required = False
        credential.enrolled_at = utcnow()
        credential.recovery_codes_digest = json.dumps([
            hashlib.sha256(value.encode()).hexdigest()
            for value in recovery_codes
        ])
        _central_security_event(
            "platform_mfa_enrolment", "success", identity.id
        )
        db.session.commit()
        session.pop("_pending_platform_mfa_secret", None)
        return render_template(
            "mfa_setup.html", enabled=True,
            recovery_codes=recovery_codes, platform_enrolment=True,
        )
    return render_template(
        "mfa_setup.html", enabled=False, secret=pending,
        provisioning_uri=provisioning_uri, qr_data_uri=qr_data_uri,
        platform_enrolment=True,
    )


@app.route("/login/platform-mfa", methods=["GET", "POST"])
def platform_mfa_challenge():
    identity, user = _pending_platform_login()
    if not identity or not user:
        session.clear()
        return redirect(url_for("login"))
    credential = PlatformMfaCredential.query.filter_by(
        identity_id=identity.id, enabled=True, reset_required=False,
    ).first()
    if not credential:
        return redirect(url_for("platform_mfa_setup"))
    if request.method == "POST":
        _validate_csrf()
        if not _consume_rate_limit("platform-mfa", identity.id):
            _central_security_event(
                "platform_mfa_rate_limited", "denied", identity.id
            )
            db.session.commit()
            abort(429, "Too many verification attempts. Try again later.")
        code = re.sub(
            r"[\s-]", "", request.form.get("code") or ""
        ).upper()
        accepted = False
        recovery_used = False
        if re.fullmatch(r"\d{6}", code):
            step = _matching_totp_step(
                _decrypt_mfa_secret(credential), code
            )
            if step is not None and (
                credential.last_used_step is None
                or step > credential.last_used_step
            ):
                credential.last_used_step = step
                accepted = True
        elif re.fullmatch(r"[A-Z0-9]{10}", code):
            digests = json.loads(
                credential.recovery_codes_digest or "[]"
            )
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest in digests:
                digests.remove(digest)
                credential.recovery_codes_digest = json.dumps(digests)
                accepted = True
                recovery_used = True
        if accepted:
            return _complete_platform_login(
                identity, user, recovery_used=recovery_used
            )
        _central_security_event(
            "platform_mfa_verification", "denied", identity.id
        )
        db.session.commit()
        flash("Invalid, expired or already-used verification code.", "error")
    return render_template("mfa_challenge.html", platform_challenge=True)


@app.route("/login/mfa", methods=["GET", "POST"])
def mfa_challenge():
    user_id = int(session.get("_mfa_user_id") or 0)
    unit_id = int(session.get("_mfa_unit_id") or 0)
    if not user_id or not unit_id:
        return redirect(url_for("login"))
    routing = db.session.get(DatabaseRoutingMetadata, unit_id)
    if DEPLOYMENT_ENV == "production" and not routing:
        session.clear()
        abort(503, "Operational database routing is unavailable.")
    g.tenant_context_token = bind_authenticated_unit(
        unit_id, routing.secret_name if routing else None
    )
    user = Staff.query.filter_by(
        id=user_id, unit_id=unit_id
    ).first()
    credential = MfaCredential.query.filter_by(
        person_id=user_id, enabled=True
    ).first()
    if not user or not credential:
        session.clear()
        return redirect(url_for("login"))
    if request.method == "POST":
        _validate_csrf()
        if not _consume_rate_limit("airport-mfa", f"{unit_id}:{user_id}"):
            abort(429, "Too many verification attempts. Try again later.")
        code = re.sub(r"[\s-]", "", request.form.get("code") or "").upper()
        accepted = False
        if re.fullmatch(r"\d{6}", code):
            step = _matching_totp_step(_decrypt_mfa_secret(credential), code)
            if step is not None and (
                credential.last_used_step is None
                or step > credential.last_used_step
            ):
                credential.last_used_step = step
                accepted = True
        elif re.fullmatch(r"[A-Z0-9]{10}", code):
            digests = json.loads(credential.recovery_codes_digest or "[]")
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest in digests:
                digests.remove(digest)
                credential.recovery_codes_digest = json.dumps(digests)
                accepted = True
        if accepted:
            next_url = session.get("_mfa_next", "")
            session.clear()
            login_user(user)
            _initialize_authenticated_session(user)
            _security_event(
                "mfa_login_succeeded",
                principal=hashlib.sha256(
                    user.username.lower().encode()
                ).hexdigest()[:16],
                unit_id=user.unit_id,
            )
            _record_successful_login(user)
            db.session.commit()
            return redirect(_canonical_login_redirect(
                next_url, user_id=user.id,
            ))
        flash("Invalid, expired or already-used verification code.", "error")
    return render_template("mfa_challenge.html")


@app.route("/security/mfa", methods=["GET", "POST"])
@login_required
def mfa_setup():
    if getattr(current_user, "role", "") == "superadmin":
        abort(
            403,
            "Platform administrator MFA is managed by the deployment identity "
            "control and cannot open an airport credential store.",
        )
    credential = MfaCredential.query.filter_by(
        person_id=current_user.id
    ).first()
    if credential and credential.enabled:
        return render_template("mfa_setup.html", enabled=True)
    pending = session.get("_pending_mfa_secret")
    if not pending:
        pending = pyotp.random_base32()
        session["_pending_mfa_secret"] = pending
    issuer = "ATCRoster"
    provisioning_uri = pyotp.TOTP(pending).provisioning_uri(
        name=current_user.username, issuer_name=issuer
    )
    qr_data_uri = _totp_qr_data_uri(provisioning_uri)
    if request.method == "POST":
        _validate_csrf()
        if not _consume_rate_limit(
            "airport-mfa-enrolment",
            f"{_current_unit_id()}:{current_user.id}",
            limit=10, window=timedelta(minutes=15),
        ):
            abort(429)
        code = re.sub(r"\s", "", request.form.get("code") or "")
        if not pyotp.TOTP(pending).verify(code, valid_window=1):
            flash("The verification code is not valid.", "error")
            return redirect(
                url_for("staff_profile", sid=current_user.id) + "#mfa"
            )
        recovery_codes = [
            secrets.token_hex(5).upper() for _ in range(10)
        ]
        if not credential:
            credential = MfaCredential(
                unit_id=_current_unit_id(), person_id=current_user.id,
                encrypted_secret="",
            )
            db.session.add(credential)
        credential.encrypted_secret = _encrypt_field(pending)
        credential.enabled = True
        credential.enrolled_at = utcnow()
        credential.recovery_codes_digest = json.dumps([
            hashlib.sha256(code.encode()).hexdigest()
            for code in recovery_codes
        ])
        db.session.commit()
        session.pop("_pending_mfa_secret", None)
        session["_auth_stamp"] = _current_auth_stamp(current_user)
        session["_new_mfa_recovery_codes"] = recovery_codes
        return redirect(
            url_for("staff_profile", sid=current_user.id) + "#mfa"
        )
    return render_template(
        "mfa_setup.html", enabled=False, secret=pending,
        provisioning_uri=provisioning_uri, qr_data_uri=qr_data_uri,
    )


@app.cli.command("bootstrap-platform")
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def bootstrap_platform(username, password):
    """Create the one-time platform control unit and Super Admin."""
    username = username.strip().lower()
    if len(password) < 12:
        raise click.ClickException("Password must contain at least 12 characters.")
    if PlatformIdentity.query.filter_by(username=username).first():
        raise click.ClickException("That platform identity already exists.")
    control = Unit.query.filter_by(status="platform_control").first()
    if not control:
        control = Unit(
            code="PLATFORM", name="ATCRoster Platform",
            status="platform_control", plan="internal", active_user_limit=5,
        )
        db.session.add(control)
        db.session.flush()
    password_hash = generate_password_hash(password)
    db.session.add(PlatformIdentity(
        public_id=f"platform-{secrets.token_hex(12)}",
        username=username, password_hash=password_hash,
    ))
    db.session.commit()
    click.echo(f"Platform Super Admin {username} created.")


@app.cli.command("reset-platform-mfa")
@click.option("--username", prompt=True)
def reset_platform_mfa(username):
    """Invalidate platform MFA and require trusted re-enrolment."""
    normalized = username.strip().lower()
    identity = PlatformIdentity.query.filter(
        db.func.lower(PlatformIdentity.username) == normalized
    ).first()
    if not identity:
        raise click.ClickException("Platform identity was not found.")
    credential = PlatformMfaCredential.query.filter_by(
        identity_id=identity.id
    ).first()
    if credential:
        credential.enabled = False
        credential.reset_required = True
        credential.encrypted_secret = ""
        credential.recovery_codes_digest = "[]"
        credential.last_used_step = None
    _central_security_event(
        "platform_mfa_reset", "success", identity.id,
        hashlib.sha256(normalized.encode()).hexdigest()[:16],
        "Re-enrolment required by trusted operator.",
    )
    db.session.commit()
    click.echo("Platform MFA reset; re-enrolment is required at next login.")


@app.cli.command("reconcile-signups")
@click.option("--apply", "apply_changes", is_flag=True)
@click.option(
    "--confirm", default="",
    help="Required with --apply: enter RECONCILE-INCOMPLETE-SIGNUPS",
)
def reconcile_signups(apply_changes, confirm):
    """Report or safely reconcile interrupted cross-database signups."""
    if apply_changes and confirm != "RECONCILE-INCOMPLETE-SIGNUPS":
        raise click.UsageError(
            "--apply requires --confirm RECONCILE-INCOMPLETE-SIGNUPS"
        )
    rows = SignupWorkflow.query.filter(
        SignupWorkflow.state != "completed"
    ).order_by(SignupWorkflow.id).all()
    for row in rows:
        invitation = db.session.get(SecureInvitation, row.invitation_id)
        routing = (
            db.session.get(
                DatabaseRoutingMetadata, invitation.unit_id
            )
            if invitation else None
        )
        click.echo(
            f"workflow={row.id} state={row.state} "
            f"error={row.last_error_code or 'none'}"
        )
        if not apply_changes or not invitation or not routing:
            continue
        if row.membership_id and row.operational_person_id:
            with operational_unit_context(
                invitation.unit_id, routing.secret_name
            ):
                staff = db.session.get(
                    Staff, row.operational_person_id
                )
                if staff:
                    staff.membership_status = "active"
                    db.session.commit()
            invitation.accepted_at = invitation.accepted_at or utcnow()
            row.state = "completed"
            row.compensation_state = ""
            row.last_error_code = ""
            if invitation.role == "UnitAdmin":
                unit = db.session.get(Unit, invitation.unit_id)
                unit.status = "active"
                routing.provisioning_state = "active"
            db.session.commit()
        else:
            if row.operational_person_id:
                with operational_unit_context(
                    invitation.unit_id, routing.secret_name
                ):
                    staff = db.session.get(
                        Staff, row.operational_person_id
                    )
                    if staff and invitation.target_person_id:
                        staff.membership_status = "active"
                        db.session.commit()
                    elif staff and staff.membership_status != "active":
                        db.session.delete(staff)
                        db.session.commit()
                row.operational_person_id = None
            if row.identity_id:
                identity = db.session.get(
                    PlatformIdentity, row.identity_id
                )
                membership = UnitMembership.query.filter_by(
                    identity_id=row.identity_id
                ).first()
                if identity and not membership:
                    db.session.delete(identity)
                    row.identity_id = None
            row.state = "compensation_required"
            row.compensation_state = "pending"
            row.last_error_code = "compensated_retry_required"
            db.session.commit()
    click.echo(f"{len(rows)} incomplete signup workflow(s) inspected.")


@app.cli.command("rotate-field-encryption")
@click.option(
    "--confirm", default="",
    help="Required: enter ROTATE-FIELD-ENCRYPTION",
)
def rotate_field_encryption(confirm):
    """Re-encrypt MFA secrets with the first configured versioned key."""
    if confirm != "ROTATE-FIELD-ENCRYPTION":
        raise click.UsageError(
            "--confirm ROTATE-FIELD-ENCRYPTION is required"
        )
    rotated = 0
    for credential in PlatformMfaCredential.query.filter(
        PlatformMfaCredential.encrypted_secret != ""
    ).all():
        credential.encrypted_secret = _encrypt_field(
            _decrypt_field(credential.encrypted_secret)
        )
        rotated += 1
    db.session.commit()
    for routing in DatabaseRoutingMetadata.query.order_by(
        DatabaseRoutingMetadata.unit_id
    ).all():
        with operational_unit_context(routing.unit_id, routing.secret_name):
            for credential in MfaCredential.query.filter(
                MfaCredential.encrypted_secret != ""
            ).all():
                credential.encrypted_secret = _encrypt_field(
                    _decrypt_field(credential.encrypted_secret)
                )
                rotated += 1
            db.session.commit()
    click.echo(f"Rotated {rotated} encrypted credential(s).")


# -------------------- DB init (single, safe block) --------------------

with app.app_context():
    if (
        DEPLOYMENT_ENV != "production"
        and os.environ.get("ATCROSTER_SKIP_RUNTIME_SCHEMA") != "1"
    ):
        db.create_all()
        seed_once()
        refresh_shift_cache()

# Expose helpers & models needed by Jinja templates that refer to them directly
app.jinja_env.globals['month_range'] = month_range
app.jinja_env.globals['ShiftType'] = ShiftType

# -------------------- Run --------------------

from auth_blueprint import AuthDependencies, create_auth_blueprint
from absence_requests_blueprint import (
    AbsenceRequestDependencies,
    create_absence_requests_blueprint,
)
from reports_blueprint import ReportsDependencies, create_reports_blueprint
from roster_blueprint import RosterDependencies, create_roster_blueprint
from training_blueprint import TrainingDependencies, create_training_blueprint
from operations_blueprint import OperationsDependencies, create_operations_blueprint

app.register_blueprint(create_auth_blueprint(AuthDependencies(
    db=db,
    PlatformIdentity=PlatformIdentity,
    UnitMembership=UnitMembership,
    DatabaseRoutingMetadata=DatabaseRoutingMetadata,
    Staff=Staff,
    Unit=Unit,
    PlatformMfaCredential=PlatformMfaCredential,
    MfaCredential=MfaCredential,
    validate_csrf=_validate_csrf,
    normalized_login=_normalized_login,
    login_rate_key=_login_rate_key,
    consume_rate_limit=_consume_rate_limit,
    reset_rate_limit=_reset_rate_limit,
    security_event=_security_event,
    central_security_event=_central_security_event,
    bind_authenticated_unit=bind_authenticated_unit,
    canonical_login_redirect=_canonical_login_redirect,
    airport_login_endpoint=_airport_login_endpoint,
    initialize_authenticated_session=_initialize_authenticated_session,
    record_successful_login=_record_successful_login,
)))
app.register_blueprint(create_reports_blueprint(ReportsDependencies(
    Assignment=Assignment,
    Staff=Staff,
    Watch=Watch,
    is_admin_user=is_admin_user,
    current_unit_id=_current_unit_id,
    validate_csrf=_validate_csrf,
    consume_rate_limit=_consume_rate_limit,
    compute_metrics_range=_compute_metrics_range,
    financial_year_start=_fy_start_for,
    parse_year_month=parse_ym,
    ensure_month_requirement=ensure_month_requirement,
    generate_month=generate_month,
    leave_summary_for_month=_leave_summary_for_month,
    current_leave_year_window=_current_leave_year_window,
    toil_accrued_used=_toil_accrued_used_in_range_half_days,
    group_consecutive_days=_group_consecutive_days,
    get_absence_types=get_absence_types,
)))
app.register_blueprint(create_roster_blueprint(RosterDependencies(
    db=db,
    RosterPublication=RosterPublication,
    Staff=Staff,
    Notification=Notification,
    Assignment=Assignment,
    Watch=Watch,
    Requirement=Requirement,
    SpecialRequirement=SpecialRequirement,
    ShiftRequest=ShiftRequest,
    AnnotationType=AnnotationType,
    AnnotationAudit=AnnotationAudit,
    can_publish_roster=_can_publish_roster,
    validate_csrf=_validate_csrf,
    parse_year_month=parse_ym,
    current_unit_id=_current_unit_id,
    month_has_data=month_has_data,
    ensure_month_requirement=ensure_month_requirement,
    generate_month=generate_month,
    active_publication=_active_roster_publication,
    publication_matches_live=_publication_matches_live_roster,
    roster_snapshot=_roster_snapshot,
    utcnow=utcnow,
    send_publication_emails=_send_roster_publication_emails,
    log_change=log_change,
    consume_rate_limit=_consume_rate_limit,
    month_range=month_range,
    requirements_for_day=requirements_for_day,
    staff_is_countable_on=staff_is_countable_on,
    exclude_from_counters=get_exclude_from_counters,
    get_shift=get_shift,
    shift_counter_group_for_day=shift_counter_group_for_day,
    night_active_on=_night_active_on,
    can_edit_roster=can_edit_roster,
    banned_roster_codes=get_banned_roster_codes,
    can_apply_annotations=can_apply_annotations,
    parse_annotation=parse_annotation,
    is_admin_user=is_admin_user,
    apply_toil_annotation_delta=_apply_toil_annotation_delta,
    load_month_roster=_load_month_roster_fast,
    add_months=_month_add,
    shift_groups=_shift_groups_snapshot,
    watch_id_for_staff_on=watch_id_for_staff_on,
    roster_fatigue_flags=roster_fatigue_flags_for_range,
    get_annotation_groups=get_annotation_groups,
)))
app.register_blueprint(create_absence_requests_blueprint(
    AbsenceRequestDependencies(
        db=db,
        Staff=Staff,
        Leave=Leave,
        Assignment=Assignment,
        is_admin_user=is_admin_user,
        parse_year_month=parse_ym,
        month_range=month_range,
        clamp_prev_next=_clamp_prev_next,
        validate_csrf=_validate_csrf,
        get_absence_types=get_absence_types,
        save_absence_types=_save_absence_types,
        tenant_get=tenant_get,
        current_unit_id=_current_unit_id,
        refresh_day_from_pattern_and_leave=refresh_day_from_pattern_and_leave,
        group_sickness_instances=_group_sickness_instances,
        ShiftType=ShiftType,
        ShiftRequest=ShiftRequest,
        unit_request_rules=_unit_request_rules,
        request_date_bounds=_request_date_bounds,
        is_month_locked=_is_month_locked,
        request_audit=_request_audit,
        utcnow=utcnow,
        safe_request_admin_month=_safe_request_admin_month,
        request_statuses=REQUEST_STATUSES,
        request_transitions=REQUEST_TRANSITIONS,
        would_create_new_fatigue_issues=would_create_new_fatigue_issues,
        staff_has_shift_qualification=_staff_has_shift_qualification,
        can_override_roster_conflicts=can_override_roster_conflicts,
        notify_requester=_notify_requester,
    )
))
app.register_blueprint(create_training_blueprint(TrainingDependencies(
    db=db,
    Staff=Staff,
    TrainingLevel=TrainingLevel,
    TrainingSession=TrainingSession,
    TrainingScore=TrainingScore,
    current_unit_id=_current_unit_id,
    training_enabled=training_enabled,
    is_editor_user=is_editor_user,
    can_manage_training=can_manage_training,
    can_record_training=can_record_training,
    is_under_training=is_under_training,
    training_profile_allowed=_training_profile_allowed,
    validate_csrf=_validate_csrf,
    QualificationType=QualificationType,
    PersonQualification=PersonQualification,
    competency_enabled=competency_enabled,
    is_admin_user=is_admin_user,
    utcnow=utcnow,
    record_qualification_history=_record_qualification_history,
    sync_qualification_to_roster_profile=_sync_qualification_to_roster_profile,
    TrainingObjective=TrainingObjective,
)))
app.register_blueprint(create_operations_blueprint(OperationsDependencies(
    db=db,
    OperationalPosition=OperationalPosition,
    PositionEndorsement=PositionEndorsement,
    PositionRequirement=PositionRequirement,
    Staff=Staff,
    ShiftType=ShiftType,
    BreakPlan=BreakPlan,
    Assignment=Assignment,
    AchievedDuty=AchievedDuty,
    FatigueReport=FatigueReport,
    RosterRuleVersion=RosterRuleVersion,
    is_admin_user=is_admin_user,
    compliance_month=_compliance_month,
    validate_csrf=_validate_csrf,
    current_unit_id=_current_unit_id,
    utcnow=utcnow,
    log_change=log_change,
    month_add=_month_add,
    position_assurance=_position_assurance,
    can_edit_roster=can_edit_roster,
    parse_year_month=parse_ym,
    month_range=month_range,
    shift_counter_group_for_day=shift_counter_group_for_day,
    staff_has_shift_qualification=_staff_has_shift_qualification,
    Scenario=Scenario,
)))
app.register_blueprint(briefing_blueprint)

# -------------------- WSGI entry point --------------------
# Compatibility alias for WSGI servers that import ``application``.
application = app

# -------------------- Local dev server --------------------
if __name__ == "__main__":
    # bind explicitly & avoid debug reloader port conflicts
    app.run(host="127.0.0.1", port=5001, debug=False)
