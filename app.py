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
from cryptography.fernet import Fernet, InvalidToken

from flask_sqlalchemy import SQLAlchemy
from flask_sqlalchemy.session import Session as FlaskSqlAlchemySession
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
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
app = Flask(__name__)

# Writable ./instance folder (works locally & on PythonAnywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

# Secrets & DB config (env-overridable)
# On PythonAnywhere set: FLASK_SECRET_KEY (and optionally DATABASE_URL)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "fallback-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(INSTANCE_DIR, 'roster.db')}"
)
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
    "pool_size": 5,
    "max_overflow": 5,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("ATCROSTER_MAX_REQUEST_BYTES", 2 * 1024 * 1024)
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("ATCROSTER_SECURE_COOKIES", "").lower() in {
    "1", "true", "yes"
}
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)

# Make external URLs prefer https behind PA’s proxy
app.config["PREFERRED_URL_SCHEME"] = "https"
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

DEPLOYMENT_ENV = os.environ.get("ATCROSTER_ENVIRONMENT", "development").lower()
FIELD_ENCRYPTION_KEY = os.environ.get("ATCROSTER_FIELD_ENCRYPTION_KEY", "")
if DEPLOYMENT_ENV == "production":
    if app.config["SECRET_KEY"] == "fallback-change-me" or len(
        app.config["SECRET_KEY"]
    ) < 32:
        raise RuntimeError(
            "Production requires FLASK_SECRET_KEY with at least 32 characters."
        )
    if str(app.config["SQLALCHEMY_DATABASE_URI"]).startswith("sqlite"):
        raise RuntimeError("Production requires PostgreSQL; SQLite is not supported.")
    if not app.config["SESSION_COOKIE_SECURE"]:
        raise RuntimeError(
            "Production requires ATCROSTER_SECURE_COOKIES=true."
        )
    if not FIELD_ENCRYPTION_KEY:
        raise RuntimeError(
            "Production requires ATCROSTER_FIELD_ENCRYPTION_KEY."
        )
    try:
        Fernet(FIELD_ENCRYPTION_KEY.encode())
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "ATCROSTER_FIELD_ENCRYPTION_KEY must be a valid Fernet key."
        ) from exc
else:
    FIELD_ENCRYPTION_KEY = base64.urlsafe_b64encode(
        hashlib.sha256(str(app.config["SECRET_KEY"]).encode()).digest()
    ).decode()


def _field_cipher() -> Fernet:
    return Fernet(FIELD_ENCRYPTION_KEY.encode())

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
    "fatigue_reporting", "custom_branding",
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


@app.before_request
def _start_request_tenant_boundary():
    clear_request_context()


@app.before_request
def _enforce_authenticated_csrf():
    """Apply one default-deny CSRF boundary to every authenticated mutation."""
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and current_user.is_authenticated
    ):
        _validate_csrf()


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder, "favicon.svg", mimetype="image/svg+xml"
    )


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
        db.session.execute(text("SELECT 1"))
        required_tables = {
            "unit", "staff", "assignment", "roster_publication",
            "operational_position", "fatigue_report", "roster_rule_version",
        }
        from sqlalchemy import inspect
        present = set(inspect(db.engine).get_table_names())
        missing = sorted(required_tables - present)
        if missing:
            return jsonify({
                "status": "not_ready", "missing_tables": missing,
            }), 503
        return jsonify({"status": "ready", "database": "ok"})
    except Exception:
        app.logger.exception("readiness_check_failed")
        return jsonify({"status": "not_ready", "database": "unavailable"}), 503


@app.errorhandler(500)
def _internal_error(error):
    app.logger.error(
        "unhandled_request_error request_id=%s path=%s",
        getattr(g, "request_id", ""), request.path, exc_info=error,
    )
    return render_template("error.html", request_id=getattr(g, "request_id", "")), 500


@app.errorhandler(400)
def _bad_request(error):
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
    ), 400


@app.errorhandler(403)
def _forbidden(_error):
    is_platform_admin = (
        getattr(current_user, "is_authenticated", False)
        and getattr(current_user, "role", "") == "superadmin"
    )
    return render_template(
        "error.html",
        status_code=403,
        error_title="You do not have access to this area",
        error_message=(
            "Platform administrators cannot access airport personnel or "
            "operational roster data. Return to Platform Administration."
            if is_platform_admin
            else (
                "Your account role does not permit this action. Return to the "
                "roster or ask your Unit Administrator for access."
            )
        ),
        home_url=url_for("platform_admin") if is_platform_admin else url_for("index"),
        home_label=(
            "Return to Platform Administration"
            if is_platform_admin
            else "Return to roster"
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
    ), 404

OPERATIONAL_TABLE_NAMES = frozenset({
    "roster_setting", "annotation_type", "watch", "staff", "shift_type",
    "requirement", "leave", "sickness", "assignment", "shift_request",
    "request_audit", "notification", "annotation_audit", "ai_rule_set",
    "change_log", "staff_watch_history", "qualification_type",
    "person_qualification", "person_qualification_history",
    "roster_publication", "roster_acknowledgement", "scenario",
    "operational_position", "position_endorsement", "position_requirement",
    "break_plan", "achieved_duty", "fatigue_report",
    "roster_rule_version", "mfa_credential",
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
    g.tenant_context_token = None
    g.platform_control_token = None
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
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
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


def _is_safe_local_redirect(target: str | None) -> bool:
    if not target:
        return False
    parsed = urllib_parse.urlsplit(target)
    return not parsed.scheme and not parsed.netloc and target.startswith("/")


# ----- SQLite performance helpers (define only; run after db exists) -----
def _enable_sqlite_fast_mode():
    """Enable WAL and other pragmas when using SQLite."""
    try:
        if "sqlite" in str(db.engine.url.drivername).lower():
            with db.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.execute(text("PRAGMA temp_store=MEMORY"))
                conn.execute(text("PRAGMA mmap_size=268435456"))  # 256MB
    except Exception:
        pass


def migrate_add_more_perf_indexes():
    """Extra composite indexes that help roster month queries."""
    try:
        db.session.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_day_staff ON assignment(day, staff_id)"
        ))
        db.session.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_staff_day_code ON assignment(staff_id, day, code)"
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()


def _init_perf_once():
    try:
        _enable_sqlite_fast_mode()
    except Exception:
        pass
    try:
        migrate_add_more_perf_indexes()
    except Exception:
        pass


    # Run the performance tweaks once at import time (Flask 3.x safe)
try:
    with app.app_context():
        _init_perf_once()
except Exception:
    # Don’t block app import if pragmas/index creation fails
    pass

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


def _sms_service_configured() -> bool:
    account_sid, auth_token, from_number = _twilio_credentials()
    return bool(account_sid and auth_token and from_number)


def _send_sms_via_twilio(to_number: str, body: str,
                         creds: tuple[str, str, str] | None = None) -> tuple[bool, str]:
    account_sid, auth_token, from_number = creds or _twilio_credentials()
    if not (account_sid and auth_token and from_number):
        return False, "SMS credentials are not configured."

    if not to_number:
        return False, "Missing destination number."

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
        with urllib_request.urlopen(req, timeout=10) as resp:
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


def _send_overtime_sms_notifications(staff_list: list["Staff"], message: str) -> tuple[int, list[tuple[Optional["Staff"], str]]]:
    creds = _twilio_credentials()
    if not (creds[0] and creds[1] and creds[2]):
        return 0, [(None, "SMS sending is not configured." )]

    sent = 0
    failures: list[tuple[Optional["Staff"], str]] = []
    for staff in staff_list:
        if not (staff and staff.phone_number):
            failures.append((staff, "No phone number on file."))
            continue
        ok, detail = _send_sms_via_twilio(staff.phone_number, message, creds)
        if ok:
            sent += 1
        else:
            failures.append((staff, detail))
    return sent, failures


def _default_overtime_sms_body(chosen_date: date | None, shift_code: str | None) -> str:
    if not (chosen_date and shift_code):
        return ""
    return (f"Overtime available on {chosen_date.isoformat()} for {shift_code} shift. "
            "Please reply if interested.")


MIN_MONTH = date(2025, 4, 1)   # Start app from April 2025

# Reference defaults (used if DB rows missing)
DEFAULT_WORKING_CODES = ["M", "D", "A", "N", "SC", "SSC", "SBY"]
DEFAULT_LEAVE_CODES = ["AL", "PL", "SPL"]
DEFAULT_BANNED_ROSTER_CODES = ["SIC", "SC", "SSC", "AL", "SP", "SPL", "PL", "TOU8", "TOUI"]
DEFAULT_EXCLUDE_FROM_COUNTERS = ["OSS"]
DEFAULT_NON_WORKING_CODES = [
    "OFF", "AL", "PL", "SPL", "TOU8", "TOUI", "OSS", "OFFICE", "WFH", "CTB", "MTG"
]

DEFAULT_ANNOTATION_TYPES = [
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
    "leave_codes": DEFAULT_LEAVE_CODES,
    "banned_codes": DEFAULT_BANNED_ROSTER_CODES,
    "exclude_from_counters": DEFAULT_EXCLUDE_FROM_COUNTERS,
    "non_working_codes": DEFAULT_NON_WORKING_CODES,
}

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


def migrate_add_met_and_assessor():
    """Idempotent: add MET/Assessor columns to staff if missing."""
    with app.app_context():
        from sqlalchemy import inspect, text
        insp = inspect(db.engine)
        try:
            cols = {c["name"] for c in insp.get_columns("staff")}
        except Exception:
            # If table doesn't exist yet, create all then re-inspect
            db.create_all()
            cols = {c["name"] for c in inspect(db.engine).get_columns("staff")}

        alters = []
        if "met_ue_expiry" not in cols:
            alters.append("ALTER TABLE staff ADD COLUMN met_ue_expiry DATE")
        if "met_ut" not in cols:
            alters.append(
                "ALTER TABLE staff ADD COLUMN met_ut BOOLEAN DEFAULT 0")
        if "has_assessor" not in cols:
            alters.append(
                "ALTER TABLE staff ADD COLUMN has_assessor BOOLEAN DEFAULT 0")

        from sqlalchemy import text  # keep this at top of the function if not already there
        for sql in alters:
            db.session.execute(text(sql))

        if alters:
            db.session.commit()

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
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "year", "month", name="uniq_unit_year_month"),)


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


class AiRuleSet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, default=1, index=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    rules_json = db.Column(db.Text, nullable=False, default="{}")
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "year", "month", name="uniq_ai_ruleset_unit_month"),)


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
DatabaseRoutingMetadata = SaaS.DatabaseRoutingMetadata
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

# Enforce the authenticated airport on all legacy operational SELECTs and
# stamp new rows. This protects older routes while they move to repositories.
from sqlalchemy import event
from sqlalchemy.orm import Session as OrmSession, with_loader_criteria
from tenancy import authenticated_unit_id

TENANT_OPERATIONAL_MODELS = (
    RosterSetting, AnnotationType, Watch, Staff, ShiftType, Requirement, Leave, Sickness,
    Assignment, ShiftRequest, RequestAudit, Notification, AnnotationAudit,
    AiRuleSet, ChangeLog, StaffWatchHistory, QualificationType,
    PersonQualification, PersonQualificationHistory,
    RosterPublication, RosterAcknowledgement, Scenario,
    OperationalPosition, PositionEndorsement, PositionRequirement, BreakPlan,
    AchievedDuty, FatigueReport, RosterRuleVersion,
    MfaCredential,
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
        return set(_normalise_codes(default))
    try:
        parsed = json.loads(raw)
    except Exception:
        return set(_normalise_codes(default))
    return set(_normalise_codes(parsed))


def get_working_codes() -> set[str]:
    return _load_codes_setting("working_codes", DEFAULT_WORKING_CODES)


def get_leave_codes() -> set[str]:
    return _load_codes_setting("leave_codes", DEFAULT_LEAVE_CODES)


def get_banned_roster_codes() -> set[str]:
    return _load_codes_setting("banned_codes", DEFAULT_BANNED_ROSTER_CODES)


def get_exclude_from_counters() -> set[str]:
    return _load_codes_setting("exclude_from_counters", DEFAULT_EXCLUDE_FROM_COUNTERS)


def get_non_working_codes() -> set[str]:
    return _load_codes_setting("non_working_codes", DEFAULT_NON_WORKING_CODES)


@lru_cache(maxsize=128)
def _annotation_snapshot(unit_id: int) -> dict[str, object]:
    rows = (AnnotationType.query
            .filter(AnnotationType.unit_id == unit_id)
            .order_by(AnnotationType.sort_order, AnnotationType.code)
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


def bootstrap_reference_data() -> None:
    Unit.__table__.create(bind=db.engine, checkfirst=True)
    AnnotationType.__table__.create(bind=db.engine, checkfirst=True)
    RosterSetting.__table__.create(bind=db.engine, checkfirst=True)
    if not db.session.get(Unit, 1):
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
            Assignment.annotation
        )
            .filter(Assignment.day >= start, Assignment.day < end)
            .all())

        a_map = {}
        for sid, d, code, source, ann in rows:
            a_map.setdefault(sid, {})[d] = (code, source, ann)

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
    return bool(getattr(u, "is_admin", False) or getattr(u, "role", "") == "admin")


def is_editor_user(u) -> bool:
    # admin counts as editor
    return getattr(u, "role", "") in ("editor", "admin")


def user_permissions(u) -> dict[str, bool]:
    try:
        raw = json.loads(getattr(u, "permissions_json", "") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {
        str(key): bool(value)
        for key, value in raw.items()
        if isinstance(key, str)
    } if isinstance(raw, dict) else {}


def has_unit_permission(u, permission: str) -> bool:
    return bool(user_permissions(u).get(permission, False))


def can_edit_roster(u) -> bool:
    return (
        is_admin_user(u)
        or is_editor_user(u)
        or (
            (
                bool(getattr(u, "is_wm", False))
                or bool(getattr(u, "is_dwm", False))
            )
            and has_unit_permission(u, "edit_roster")
        )
    )


def can_apply_annotations(u) -> bool:
    return (
        is_admin_user(u)
        or is_editor_user(u)
        or has_unit_permission(u, "apply_annotations")
    )


def can_override_roster_conflicts(u) -> bool:
    return is_admin_user(u) or has_unit_permission(
        u, "override_roster_conflicts"
    )


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
    start = date(year, month, 1)
    stop = date(year + (month // 12), (month % 12) + 1, 1)
    days = (stop - start).days
    return start, [start + timedelta(d) for d in range(days)]


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
    y, m = ym.split("-")
    return int(y), int(m)


def get_shift(code: str):
    # hot path → use cached lookup
    return _shift_by_code(
        int(_current_unit_id() or 1), (code or "").upper()
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


def pattern_for(staff: Staff):
    """
    CSV like 'M,M,A,A,N,N,OFF,OFF' plus multipliers: '6xM,14xOFF' or 'M*6,OFF*14'.
    """
    raw = [p.strip()
           for p in (staff.pattern_csv or "").split(",") if p.strip()]
    out = []
    for tok in raw:
        tok_u = tok.upper()
        m = re.match(r"^\s*(\d+)\s*[x\*]\s*([A-Z]+)\s*$", tok_u)
        m2 = re.match(r"^\s*([A-Z]+)\s*[x\*]\s*(\d+)\s*$", tok_u)
        if m:
            n, code = int(m.group(1)), m.group(2)
            out.extend([code] * n)
        elif m2:
            code, n = m2.group(1), int(m2.group(2))
            out.extend([code] * n)
        else:
            out.append(tok_u)
    return out


def day_leave_for(staff: Staff, d: date):
    for lv in staff.leaves:
        if lv.start <= d <= lv.end:
            return lv.leave_type
    return None


def code_from_pattern(staff: Staff, d: date):
    pat = pattern_for(staff)
    if not pat:
        return "OFF"
    anchor = staff.pattern_anchor or date(d.year, d.month, 1)
    idx = (d - anchor).days % len(pat)
    return pat[idx]


def _cycle_day_for(staff: Staff, d: date) -> int | None:
    """Return the 1-indexed pattern cycle day for `staff` on date `d`."""
    pat = pattern_for(staff)
    if not pat:
        return None
    anchor = staff.pattern_anchor or date(d.year, d.month, 1)
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

    # Keep explicit sickness & TOIL-use exactly as entered
    if existing and existing.code in {"SC", "SSC", "TOU8", "TOUI"}:
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

    if lv in {"PL", "SPL"}:
        a = overwrite_assignment(staff, d, lv, note="leave")
        a.annotation = ""
        return a

    # No leave: (re)write pattern but preserve annotations unless code changes
    a = set_assignment(staff, d, pat_code, source="auto", note="pattern")
    if prev_code is None or (prev_code != a.code and a.source != "manual"):
        a.annotation = ""
    return a


def shift_duration_minutes(shift: ShiftType):
    if not shift or not shift.start_time or not shift.end_time:
        return 0
    dt0 = datetime.combine(date(2000, 1, 1), shift.start_time)
    dt1 = datetime.combine(date(2000, 1, 1), shift.end_time)
    if dt1 <= dt0:
        dt1 += timedelta(days=1)
    return int((dt1 - dt0).total_seconds() // 60)


def ensure_month_requirement(year, month, default=(4, 4, 4, 2)):
    r = Requirement.query.filter_by(year=year, month=month).first()
    if not r:
        if len(default) == 3:
            dm, da, dn = default[0], default[1], default[2]
            dd = 0
        else:
            dm, dd, da, dn = default
        r = Requirement(year=year, month=month, req_m=dm,
                        req_d=dd, req_a=da, req_n=dn)
        db.session.add(r)
        db.session.commit()
    return r

# Idempotent month generation that preserves manual entries


def generate_month(year: int, month: int, *args, **kwargs):
    """Ensure rows exist/are correct for the month without touching manual edits."""
    _, days = month_range(year, month)
    for s in Staff.query.order_by(Staff.id):
        for d in days:
            refresh_day_from_pattern_and_leave(s, d)
    db.session.commit()


NIGHT_ELIGIBLE_CYCLE_DAYS: set[int] = {5, 6}


def generate_month_roster(year: int, month: int, who_user: "Staff"):
    # Guard: lock
    if is_month_locked(year, month):
        raise RuntimeError(
            f"Month {year:04d}-{month:02d} is locked (lock date {lock_date_for_month(year, month).isoformat()})."
        )

    rules = _load_ai_rules(year, month)
    night_code = rules["night_code"]
    day_code_mon_sat = rules["day_code_mon_sat"]
    day_code_sun = rules["day_code_sun"]
    no_nights_ids = rules["no_nights_ids"]

    # Month range
    start, days = month_range(year, month)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)

    # Staff in display/fairness order
    staff = (
        Staff.query
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .filter(Staff.is_operational == True)
        .order_by(Watch.order_index, Staff.name)
        .all()
    )

    req = Requirement.query.filter_by(year=year, month=month).first()
    if not req:
        return {"nights": 0, "days": 0}

    # Existing assignments → quick lookup + night counts
    night_count = defaultdict(int)
    by_staff_day = defaultdict(dict)
    existing_assignments = Assignment.query.filter(
        Assignment.day >= start, Assignment.day < month_end
    ).all()
    for a in existing_assignments:
        by_staff_day[a.staff_id][a.day] = a
        if a.code and a.code.upper().startswith("N"):
            night_count[a.staff_id] += 1

    # Candidate helper
    def eligible_staff(d: date, code: str):
        candidate_shift = get_shift(code)
        for s in staff:
            if code == night_code:
                if s.id in no_nights_ids:
                    continue

                cycle_day = _cycle_day_for(s, d)
                if cycle_day not in NIGHT_ELIGIBLE_CYCLE_DAYS:
                    continue

                pattern_code = code_from_pattern(s, d)
                if not _is_working_n_code(pattern_code):
                    continue
            if _has_leave_or_sick(s.id, d):
                continue
            a = by_staff_day[s.id].get(d) or Assignment(staff_id=s.id, day=d)
            if _cell_is_protected(a):
                continue
            if not _fatigue_ok(s, d, code):
                continue
            if candidate_shift and not _staff_has_shift_qualification(
                s, candidate_shift, d
            ):
                continue
            yield s

    changes_n = 0
    changes_d = 0

    for d in days:
        dow = d.weekday()  # 0=Mon .. 6=Sun
        day_code = day_code_mon_sat if dow < 6 else day_code_sun

        # --- Nights ---
        current_nights = Assignment.query.filter_by(
            day=d, code=night_code).count()
        needed_nights = max(0, req.req_n - current_nights)
        while needed_nights > 0:
            candidates = sorted(eligible_staff(d, night_code),
                                key=lambda s: night_count[s.id])
            if not candidates:
                break
            s = candidates[0]
            a = by_staff_day[s.id].get(d)
            if not a:
                a = Assignment(staff_id=s.id, day=d)
                db.session.add(a)
                by_staff_day[s.id][d] = a
            a.code = night_code
            a.source = "ai"      # mark as AI, not manual
            night_count[s.id] += 1
            changes_n += 1
            needed_nights -= 1

        # --- Days ---
        # count all working D* already on this day (any D-prefix working code)
        rows_today = Assignment.query.filter_by(day=d).all()
        haveD = sum(
            1 for a in rows_today if _is_working_day_code((a.code or "")))
        needD = getattr(req, "req_d", 0)
        short = max(0, needD - haveD)
        if short > 0:
            # candidates: empty/OFF cells only, fatigue OK
            candidates = []
            for s in staff:
                if _has_leave_or_sick(s.id, d):
                    continue
                a = by_staff_day[s.id].get(d)
                current_code = a.code if a else ""
                # only fill if truly empty-like ("" / "-" / "—") OR explicitly OFF
                if not (_is_empty_like(current_code) or _normalize_code(current_code) == "OFF"):
                    continue

                if a and _cell_is_protected(a):
                    continue
                if not _fatigue_ok(s, d, day_code):
                    continue
                candidates.append(s)

            # simple fairness: fewest assigned D today so far, then name
            day_count = defaultdict(int)
            candidates.sort(key=lambda s: (day_count[s.id], s.name.lower()))

            for s in candidates[:short]:
                a = by_staff_day[s.id].get(d)
                if not a:
                    a = Assignment(staff_id=s.id, day=d)
                    db.session.add(a)
                    by_staff_day[s.id][d] = a

                # safety double-check
                ex = (a.code or "").strip().upper() if a.code else ""
                if ex and ex not in ("", "OFF"):
                    continue

                a.code = day_code
                a.source = "ai"
                changes_d += 1
                day_count[s.id] += 1

    db.session.commit()
    return {"nights": changes_n, "days": changes_d}


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


def _is_night_0130_0529(start_dt: datetime, end_dt: datetime) -> bool:
    return _overlap_window(start_dt, end_dt, 1, 30, 5, 29) > 0


def _is_early_start(start_dt: datetime) -> Tuple[bool, bool]:
    hm = start_dt.time()
    is_early = (time(5, 30) <= hm <= time(6, 29))
    is_pre0600 = is_early and (hm < time(6, 0))
    return is_early, is_pre0600


def _is_morning_duty(start_dt: datetime) -> bool:
    hm = start_dt.time()
    return time(6, 30) <= hm <= time(7, 59)


def _segments_for_staff(staff: Staff, start_day: date, end_day: date):
    segs = []
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

        night = _is_night_0130_0529(sdt, edt)
        is_early, is_pre0600 = _is_early_start(sdt)
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


def _analyze_segments(segs):
    segs = sorted(segs, key=lambda x: x["start"])
    flags = {}
    if not segs:
        return flags

    thirty_days = timedelta(hours=720)
    six_days_hours = timedelta(hours=144)

    win30 = deque()
    duty_30 = 0
    reduced_intervals_30 = deque()
    rest_gaps_30 = deque()

    night_block_count = 0
    last_night_end = None

    consec_queue = deque(maxlen=6)

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
            req_hours = 48 if night_block_count == 1 else 54
            if gap < timedelta(hours=req_hours):
                flags.setdefault(the_day, []).append(
                    f"<{req_hours}h after {'single' if night_block_count == 1 else 'two consecutive'} night(s) (D31: {int(gap.total_seconds()//3600)}h)"
                )
            night_block_count = 0
            last_night_end = None

        if prev_end is not None:
            gap = start - prev_end
            while reduced_intervals_30 and (start - reduced_intervals_30[0]) > thirty_days:
                reduced_intervals_30.popleft()
            if gap < timedelta(hours=12):
                if gap >= timedelta(hours=11):
                    if len(reduced_intervals_30) == 0:
                        reduced_intervals_30.append(start)
                    else:
                        flags.setdefault(the_day, []).append(
                            f"<12h between duties (D22) and 11–12h allowance already used within last 30 days"
                        )
                else:
                    flags.setdefault(the_day, []).append(
                        f"<11h between duties (D22: {int(gap.total_seconds()//3600)}h)"
                    )

            rest_gaps_30.append((start, gap))
            while rest_gaps_30 and (end - rest_gaps_30[0][0]) > thirty_days:
                rest_gaps_30.popleft()

        # D24 — sum only qualifying rests (≥54h) in the last 30 days; need ≥180h total
        qual_hours = 0.0
        for _, g in rest_gaps_30:
            if g >= timedelta(hours=54):
                qual_hours += g.total_seconds() / 3600.0
        if qual_hours < 180.0:
            flags.setdefault(the_day, []).append(
                f"D24: qualifying rest {int(round(qual_hours))}h (<180h) in last 30d"
            )

        prior_consec_count = len(consec_queue)
        prior_consec_minutes = sum(m for (_, _, m) in consec_queue)
        if (prior_consec_count >= 6) or (prior_consec_minutes >= 50 * 60):
            if prev_end is not None:
                gap = start - prev_end
                if gap < timedelta(hours=60):
                    if gap < timedelta(hours=54):
                        flags.setdefault(the_day, []).append(
                            f"<60h after 6 consecutive duties or ≥50h across consecutive duties (D23: {int(gap.total_seconds()//3600)}h)"
                        )

        if mins > 10 * 60:
            flags.setdefault(the_day, []).append("Duty > 10h (D21)")

        while win30 and (end - win30[0][1]) > thirty_days:
            _, _, mo = win30.popleft()
            duty_30 -= mo
        win30.append((start, end, mins))
        duty_30 += mins
        if duty_30 > 200 * 60:
            flags.setdefault(the_day, []).append(
                ">200h duty in last 30 days (D21)")

        if night:
            if mins > int(9.5 * 60):
                flags.setdefault(the_day, []).append("Night duty > 9.5h (D30)")
            if end.time() > time(7, 30):
                flags.setdefault(the_day, []).append(
                    "Night duty ends after 07:30 (D30)")

            if last_duty_day and (the_day - last_duty_day).days == 1 and last_was_night:
                night_block_count += 1
            else:
                night_block_count = 1

            if night_block_count > 2:
                flags.setdefault(the_day, []).append(
                    "3rd consecutive night duty (D30)")

            last_night_end = end

        if early:
            early_window.append(start)
            while early_window and (start - early_window[0]) > six_days_hours:
                early_window.popleft()
            if len(early_window) > 2:
                flags.setdefault(the_day, []).append(
                    "More than 2 early starts in 144h (D39)")
            if early_pre0600 and last_was_early_pre0600 and last_duty_day and (the_day - last_duty_day).days == 1:
                flags.setdefault(the_day, []).append(
                    "Consecutive early starts both before 06:00 not permitted (D39)"
                )
            if mins > 8 * 60:
                flags.setdefault(the_day, []).append(
                    "Early start duty > 8h (D40)")

        if early or morning:
            points_today = 2 if early_pre0600 else 1
            if last_duty_day and (the_day - last_duty_day).days == 1 and (morning_streak_points > 0):
                morning_streak_points += points_today
            else:
                morning_streak_points = points_today
            if morning_streak_points > 5:
                flags.setdefault(the_day, []).append(
                    "More than 5 consecutive morning-duty periods (D43)")
        else:
            morning_streak_points = 0

        if morning and mins > int(8.5 * 60):
            flags.setdefault(the_day, []).append("Morning duty > 8.5h (D43)")

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
    all_flags = _analyze_segments(segs)
    target_set = set(day_list)
    return {d: f for d, f in all_flags.items() if d in target_set}


def would_trigger_fatigue(staff: Staff, day: date, code: str):
    sh = get_shift(code)
    if not _is_working(sh):
        return []
    start_lb = day - timedelta(days=30)
    end_day = day
    segs = _segments_for_staff(staff, start_lb, end_day)
    sdt, edt = _span(day, sh)
    if sdt:
        segs.append({
            "day": day, "start": sdt, "end": edt,
            "mins": int((edt - sdt).total_seconds() // 60),
            "night": _is_night_0130_0529(sdt, edt),
            "early": _is_early_start(sdt)[0],
            "early_pre0600": _is_early_start(sdt)[1],
            "morning": _is_morning_duty(sdt),
        })
    flags = _analyze_segments(segs)
    return flags.get(day, [])


def _year_month_iter(start_date: date, end_date: date):
    y, m = start_date.year, start_date.month
    last = date(end_date.year, end_date.month, 1)
    cur = date(y, m, 1)
    while cur <= last:
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1
        cur = date(y, m, 1)


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
    flags_base = _analyze_segments(segs_base)
    sdt, edt = _span(proposed_day, sh)
    if not sdt:
        return {}
    segs_prop = list(segs_base)
    segs_prop.append({
        "day": proposed_day,
        "start": sdt,
        "end": edt,
        "mins": int((edt - sdt).total_seconds() // 60),
        "night": _is_night_0130_0529(sdt, edt),
        "early": _is_early_start(sdt)[0],
        "early_pre0600": _is_early_start(sdt)[1],
        "morning": _is_morning_duty(sdt),
    })
    flags_prop = _analyze_segments(segs_prop)
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
    for person in people:
        flags = fatigue_flags_for_range(person, days)
        issues = []
        for finding_day, messages in sorted(flags.items()):
            assignment = Assignment.query.filter_by(
                staff_id=person.id, day=finding_day
            ).first()
            for message in messages:
                rule = message.split(":", 1)[0].split("(", 1)[0].strip()
                severity = "critical" if any(
                    token in message
                    for token in ("<11h", "3rd consecutive", ">200h", "> 10h")
                ) else "warning"
                rule_counts[rule] += 1
                issues.append({
                    "day": finding_day,
                    "message": message,
                    "rule": rule,
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
    if not is_admin_user(current_user):
        abort(403)
    year, month = _compliance_month(request.args.get("ym"))
    findings = _compliance_findings(year, month)
    py, pm = _month_add(year, month, -1)
    ny, nm = _month_add(year, month, 1)
    return render_template(
        "compliance_centre.html",
        ym=f"{year:04d}-{month:02d}",
        month_title=date(year, month, 1).strftime("%B %Y"),
        prev_ym=f"{py:04d}-{pm:02d}",
        next_ym=f"{ny:04d}-{nm:02d}",
        **findings,
    )


@app.route("/compliance-centre/export")
@login_required
def compliance_centre_export():
    if not is_admin_user(current_user):
        abort(403)
    year, month = _compliance_month(request.args.get("ym"))
    findings = _compliance_findings(year, month)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Airport", "Month", "ATCO", "Staff number", "Watch", "Date",
        "Severity", "Rule", "Finding",
    ])
    unit = db.session.get(Unit, _current_unit_id())
    for row in findings["rows"]:
        person = row["staff"]
        for issue in row["issues"]:
            writer.writerow([
                unit.code if unit else "",
                f"{year:04d}-{month:02d}",
                person.name,
                person.staff_no,
                person.watch.name if person.watch else "",
                issue["day"].isoformat(),
                issue["severity"],
                issue["rule"],
                issue["message"],
            ])
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                f"attachment; filename=compliance-evidence-{year:04d}-{month:02d}.csv"
        },
    )

# -------------------- Migrations / seeding --------------------


def migrate_tenant_foundation_compat():
    """Idempotently upgrade legacy SQLite desktops before normal startup."""
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if "unit" not in inspector.get_table_names():
        db.create_all()
        inspector = inspect(db.engine)
    if not db.session.get(Unit, 1):
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
    idx = y * 12 + (m - 1) + delta
    ny = idx // 12
    nm = idx % 12 + 1
    return ny, nm


def lock_date_for_month(y: int, m: int) -> date:
    ly, lm = _month_add(y, m, -2)
    return date(ly, lm, 20)


def is_month_locked(y: int, m: int, today: Optional[date] = None) -> bool:
    if today is None:
        today = date.today()
    return today >= lock_date_for_month(y, m)


# Source protection: we never overwrite these
LOCKED_SOURCES = {"manual", "leave", "sickness"}

# Fall back defaults if AiRuleSet doesn’t specify
AI_DEFAULTS = {
    "no_nights_ids": [],          # [staff_id, ...]
    "night_code": "N",            # your night code
    "day_code_mon_sat": "D10",    # fallback day code Mon–Sat
    "day_code_sun": "D12",        # fallback day code Sun
    "enable_nops": False,         # requests logic comes in Part 3
}


def _load_ai_rules(y: int, m: int) -> dict:
    ars = AiRuleSet.query.filter_by(year=y, month=m).first()
    if not ars:
        ars = AiRuleSet.query.order_by(AiRuleSet.id.desc()).first()
    if not ars or not getattr(ars, "rules_json", None):
        return dict(AI_DEFAULTS)
    try:
        rules = json.loads(ars.rules_json or "{}")
    except Exception:
        rules = {}
    merged = dict(AI_DEFAULTS)
    merged.update(rules or {})
    # normalise set types
    merged["no_nights_ids"] = set(merged.get("no_nights_ids", []))
    return merged


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

# Rules persistence used by admin UI


def _load_rules_for(y: int, m: int) -> dict:
    rs = AiRuleSet.query.filter_by(year=y, month=m).first()
    if rs and rs.rules_json:
        try:
            return json.loads(rs.rules_json)
        except Exception:
            return _default_ai_rules()
    return _default_ai_rules()


def _save_rules_for(y: int, m: int, rules: dict):
    rs = AiRuleSet.query.filter_by(year=y, month=m).first()
    payload = json.dumps(rules, separators=(",", ":"))
    if not rs:
        rs = AiRuleSet(year=y, month=m, rules_json=payload)
        db.session.add(rs)
    else:
        rs.rules_json = payload
    db.session.commit()
    log_change("AiRuleSet", rs.id, "rules_json", None,
               payload, context_day=date(y, m, 1))
    return rs


def _default_ai_rules():
    return {
        "respect_user_requests": True,
        "equalize_nights": True,
        "nights_in_pairs": True,
        "night_pool_watches": ["A", "B", "C", "D", "E"],
        "avoid_night_before_AL": True,
        "fill_short_with_D": True,
        "day_shift_codes": {"weekday": "D10", "sunday": "D12"},
        "no_nights_list": [],
        "nops_policy": {"integrate_required": True, "only_if_needed": True},
    }


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
            return redirect(url_for("password_change"))
        if not new1 or new1 != new2:
            flash("New passwords do not match.", "error")
            return redirect(url_for("password_change"))
        if len(new1) < 12:
            flash("Use a password of at least 12 characters.", "error")
            return redirect(url_for("password_change"))
        if new1 == cur:
            flash("Choose a password different from the current password.", "error")
            return redirect(url_for("password_change"))
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
            else url_for("staff_profile", sid=current_user.id)
        )
    return render_template("password.html")

# -------------------- Main / Roster --------------------


@app.route("/")
@login_required
def index():
    t = date.today()
    return redirect(url_for("roster_month", ym=f"{t.year}-{t.month:02d}"))


def _roster_snapshot(year: int, month: int) -> dict:
    start = date(year, month, 1)
    ny, nm = _month_add(year, month, 1)
    end = date(ny, nm, 1)
    assignments = (
        Assignment.query.filter(
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
            ):
                group = assignment.code[:1].upper()
                if group in ("M", "D", "A", "N"):
                    counts[day][group] += 1

    coverage_gaps = []
    for day in days:
        for group in ("M", "D", "A", "N"):
            needed = int(getattr(requirement, f"req_{group.lower()}", 0) or 0)
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
    hard_blocks = (
        len(qualification_gaps) + len(unassigned)
        + len(position_shortfalls) + len(critical_reports)
        + len(configuration_blocks)
    )
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
        "exceptions": fatigue["total"] + len(coverage_gaps),
        "ready": hard_blocks == 0,
    }


@app.route("/publications")
@login_required
def publication_index():
    today = date.today()
    return redirect(url_for(
        "publication_centre", ym=f"{today.year:04d}-{today.month:02d}"
    ))


@app.route("/publications/<ym>", methods=["GET", "POST"])
@login_required
def publication_centre(ym):
    year, month = _compliance_month(ym)
    if request.method == "POST":
        _validate_csrf()
        action = (request.form.get("action") or "").strip()
        if action == "publish":
            if not is_admin_user(current_user):
                abort(403)
            preflight = _publication_preflight(year, month)
            declaration = request.form.get("release_declaration") == "yes"
            exception_reason = (
                request.form.get("exception_reason") or ""
            ).strip()
            if preflight["hard_blocks"]:
                flash(
                    "Publication blocked: resolve all configuration, competence, "
                    "position-coverage and critical fatigue-report failures first.",
                    "error",
                )
                return redirect(url_for("publication_centre", ym=ym))
            if not declaration:
                flash(
                    "Confirm the accountable manager release declaration before publishing.",
                    "error",
                )
                return redirect(url_for("publication_centre", ym=ym))
            if preflight["exceptions"] and len(exception_reason) < 20:
                flash(
                    "Record an exception rationale of at least 20 characters "
                    "for fatigue findings or staffing shortfalls.",
                    "error",
                )
                return redirect(url_for("publication_centre", ym=ym))
            current = (
                RosterPublication.query.filter_by(
                    year=year, month=month, state="published"
                )
                .order_by(RosterPublication.version.desc())
                .first()
            )
            latest_version = (
                db.session.query(db.func.max(RosterPublication.version))
                .filter(
                    RosterPublication.unit_id == _current_unit_id(),
                    RosterPublication.year == year,
                    RosterPublication.month == month,
                )
                .scalar()
                or 0
            )
            if current:
                current.state = "superseded"
                current.superseded_at = utcnow()
            snapshot = _roster_snapshot(year, month)
            snapshot["release_assurance"] = {
                "declared_by_id": current_user.id,
                "declared_at": utcnow().isoformat(),
                "declaration": (
                    "Coverage, competence, fatigue findings and operational "
                    "contingencies reviewed by the accountable roster manager."
                ),
                "exception_reason": exception_reason,
                "fatigue_findings": preflight["fatigue_total"],
                "critical_fatigue_findings": preflight["fatigue_critical"],
                "coverage_shortfalls": len(preflight["coverage_gaps"]),
                "qualification_failures": len(preflight["qualification_gaps"]),
                "unassigned_cells": len(preflight["unassigned"]),
                "position_shortfalls": len(preflight["position_shortfalls"]),
                "open_critical_fatigue_reports": len(
                    preflight["critical_fatigue_reports"]
                ),
                "approved_rule_version": (
                    preflight["approved_rule"].version
                    if preflight["approved_rule"] else None
                ),
            }
            publication = RosterPublication(
                unit_id=_current_unit_id(),
                year=year, month=month,
                version=latest_version + 1,
                state="published",
                snapshot_json=json.dumps(snapshot),
                published_at=utcnow(),
            )
            db.session.add(publication)
            for person in Staff.query.filter_by(
                is_operational=True, membership_status="active"
            ).all():
                db.session.add(Notification(
                    unit_id=_current_unit_id(),
                    recipient_id=person.id,
                    kind="roster_published",
                    message=(
                        f"{date(year, month, 1).strftime('%B %Y')} roster "
                        f"version {publication.version} is ready to review."
                    ),
                ))
            db.session.commit()
            log_change(
                "RosterPublication", publication.id, "state", "draft",
                "published", note=(
                    f"Manager declaration recorded. Exceptions: "
                    f"{exception_reason or 'none'}"
                ), context_day=date(year, month, 1),
            )
            flash(f"Roster version {publication.version} published.", "ok")
            return redirect(url_for("publication_centre", ym=ym))
        if action == "acknowledge":
            publication_id = int(request.form.get("publication_id") or 0)
            publication = RosterPublication.query.filter_by(
                id=publication_id, year=year, month=month, state="published"
            ).first_or_404()
            existing = RosterAcknowledgement.query.filter_by(
                publication_id=publication.id, person_id=current_user.id
            ).first()
            if not existing:
                db.session.add(RosterAcknowledgement(
                    unit_id=_current_unit_id(),
                    publication_id=publication.id,
                    person_id=current_user.id,
                ))
                db.session.commit()
            flash("Roster acknowledgement recorded.", "ok")
            return redirect(url_for("publication_centre", ym=ym))
        if action == "rollback":
            if not is_admin_user(current_user):
                abort(403)
            try:
                publication_id = int(
                    request.form.get("publication_id") or 0
                )
            except ValueError:
                abort(400)
            target = RosterPublication.query.filter_by(
                id=publication_id, year=year, month=month,
            ).first_or_404()
            if target.state not in {"published", "superseded"}:
                abort(409, "Only a released roster version can be restored.")
            try:
                target_snapshot = json.loads(target.snapshot_json)
                target_rows = target_snapshot["assignments"]
                if not isinstance(target_rows, list):
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                abort(409, "The selected snapshot is not restorable.")
            valid_people = {
                row.id for row in Staff.query.filter_by(
                    unit_id=_current_unit_id()
                ).all()
            }
            restored = {}
            for item in target_rows:
                try:
                    staff_id = int(item["staff_id"])
                    target_day = date.fromisoformat(item["day"])
                    code = str(item["code"]).strip().upper()
                except (KeyError, TypeError, ValueError):
                    abort(409, "The selected snapshot contains invalid data.")
                if (
                    staff_id not in valid_people
                    or target_day.year != year
                    or target_day.month != month
                    or not code
                ):
                    abort(409, "The snapshot does not belong to this roster.")
                restored[(staff_id, target_day)] = {
                    "code": code,
                    "annotation": str(item.get("annotation") or "")[:20],
                }
            start = date(year, month, 1)
            next_year, next_month = _month_add(year, month, 1)
            end = date(next_year, next_month, 1)
            existing_rows = Assignment.query.filter(
                Assignment.unit_id == _current_unit_id(),
                Assignment.day >= start, Assignment.day < end,
            ).all()
            existing = {
                (row.staff_id, row.day): row for row in existing_rows
            }
            for key, row in existing.items():
                if key not in restored:
                    db.session.delete(row)
            for (staff_id, target_day), values in restored.items():
                row = existing.get((staff_id, target_day))
                if not row:
                    row = Assignment(
                        unit_id=_current_unit_id(),
                        staff_id=staff_id, day=target_day,
                    )
                    db.session.add(row)
                row.code = values["code"]
                row.annotation = values["annotation"]
                row.source = "rollback"
                row.note = f"Restored from roster version {target.version}"
            current = RosterPublication.query.filter_by(
                year=year, month=month, state="published"
            ).first()
            if current:
                current.state = "superseded"
                current.superseded_at = utcnow()
            latest_version = (
                db.session.query(db.func.max(RosterPublication.version))
                .filter(
                    RosterPublication.unit_id == _current_unit_id(),
                    RosterPublication.year == year,
                    RosterPublication.month == month,
                ).scalar() or 0
            )
            restored_snapshot = dict(target_snapshot)
            restored_snapshot["generated_at"] = utcnow().isoformat()
            restored_snapshot["rollback"] = {
                "source_version": target.version,
                "approved_by_id": current_user.id,
                "approved_at": utcnow().isoformat(),
            }
            release = RosterPublication(
                unit_id=_current_unit_id(), year=year, month=month,
                version=latest_version + 1, state="published",
                snapshot_json=json.dumps(restored_snapshot),
                published_at=utcnow(),
            )
            db.session.add(release)
            db.session.commit()
            log_change(
                "RosterPublication", release.id, "state",
                f"version {current.version if current else 'none'}",
                f"rollback of version {target.version}",
                note="Controlled roster rollback",
                context_day=start,
            )
            flash(
                f"Version {target.version} restored as new version "
                f"{release.version}.",
                "ok",
            )
            return redirect(url_for("publication_centre", ym=ym))
        abort(400)

    publications = (
        RosterPublication.query.filter_by(year=year, month=month)
        .order_by(RosterPublication.version.desc())
        .all()
    )
    active = next((row for row in publications if row.state == "published"), None)
    active_assignments = {}
    if active:
        try:
            active_assignments = {
                (int(item["staff_id"]), item["day"]): (
                    item.get("code"), item.get("annotation") or ""
                )
                for item in json.loads(active.snapshot_json).get(
                    "assignments", []
                )
            }
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            active_assignments = {}
    publication_diffs = {}
    for publication in publications:
        try:
            comparison = {
                (int(item["staff_id"]), item["day"]): (
                    item.get("code"), item.get("annotation") or ""
                )
                for item in json.loads(publication.snapshot_json).get(
                    "assignments", []
                )
            }
            keys = set(active_assignments) | set(comparison)
            publication_diffs[publication.id] = sum(
                active_assignments.get(key) != comparison.get(key)
                for key in keys
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            publication_diffs[publication.id] = None
    preflight = _publication_preflight(year, month)
    acknowledged = False
    acknowledgements = []
    expected_staff = Staff.query.filter_by(
        is_operational=True, membership_status="active"
    ).order_by(Staff.name).all()
    unacknowledged_staff = []
    release_assurance = {}
    if active:
        acknowledgements = RosterAcknowledgement.query.filter_by(
            publication_id=active.id
        ).all()
        acknowledged = any(
            row.person_id == current_user.id for row in acknowledgements
        )
        acknowledged_ids = {row.person_id for row in acknowledgements}
        unacknowledged_staff = [
            person for person in expected_staff if person.id not in acknowledged_ids
        ]
        try:
            release_assurance = json.loads(
                active.snapshot_json or "{}"
            ).get("release_assurance", {})
        except (TypeError, json.JSONDecodeError):
            release_assurance = {}
    py, pm = _month_add(year, month, -1)
    ny, nm = _month_add(year, month, 1)
    return render_template(
        "publication_centre.html",
        ym=ym, year=year, month=month,
        month_title=date(year, month, 1).strftime("%B %Y"),
        publications=publications, active=active,
        publication_diffs=publication_diffs,
        acknowledgements=acknowledgements, acknowledged=acknowledged,
        operational_count=len(expected_staff),
        unacknowledged_staff=unacknowledged_staff,
        release_assurance=release_assurance,
        preflight=preflight,
        prev_ym=f"{py:04d}-{pm:02d}", next_ym=f"{ny:04d}-{nm:02d}",
    )


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
    return {
        "is_admin":  bool(au) and is_admin_user(au),
        "is_editor": bool(au) and is_editor_user(au),
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


@app.route("/roster/<ym>")
@login_required
def roster_month(ym):
    year, month = parse_ym(ym)
    unit_id = _current_unit_id()

    # Build only if the month has no data yet
    if not month_has_data(year, month):
        ensure_month_requirement(year, month)
        generate_month(year, month)

    # Fast path: 2 queries total (staff + all assignments in month)
    days, staff, a_map_tuples, req = _load_month_roster_fast(
        unit_id, year, month
    )

    # --- ensure WM first, DWM second, then alphabetical within each watch ---
    def _rank_within_watch(person):
        # 0 = WM, 1 = DWM, 2 = everyone else
        return 0 if getattr(person, "is_wm", False) else (1 if getattr(person, "is_dwm", False) else 2)

    # Build code_map (what the template expects) and ann_map from the tuples
    # a_map_tuples[sid][day] = (code, source, annotation)
    a_map: dict[int, dict[date, str]] = {}
    ann_map: dict[int, dict[date, str]] = {}
    for sid, dmap in a_map_tuples.items():
        codes = {}
        anns = {}
        for d, (code, _src, ann) in dmap.items():
            codes[d] = code
            anns[d] = ann or ""
        a_map[sid] = codes
        ann_map[sid] = anns

    # Prev/next month strings
    py, pm = _month_add(year, month, -1)
    ny, nm = _month_add(year, month, +1)
    prev_ym = f"{py:04d}-{pm:02d}"
    next_ym = f"{ny:04d}-{nm:02d}"

    # Month bounds for queries like ShiftRequest
    start = date(year, month, 1)
    month_end = date(ny, nm, 1)

    # Shift dropdown groupings (cached)
    shifts_working, shifts_training, shifts_non = _shift_groups_snapshot(
        unit_id
    )
    training_codes = {sh.code for sh in shifts_training}

    # --- Effective watch for THIS month (first day of the month) ---
    def _watch_for(sid: int, on_date: date):
        fn = globals().get("watch_id_for_staff_on")
        if callable(fn):
            return fn(sid, on_date)
        s = tenant_get(Staff, sid)
        return s.watch_id if s else None

    display_watch_by_staff = {s.id: _watch_for(s.id, start) for s in staff}

    # Optional: ensure staff ordering matches watch order for the display month
    try:
        watch_order = {w.id: w.order_index for w in Watch.query.all()}
    except Exception:
        watch_order = {}

    staff.sort(
        key=lambda s: (
            watch_order.get(display_watch_by_staff.get(s.id), 9999),
            _rank_within_watch(s),
            s.name
        )
    )

    # Counters (operational only); exclude training and configured exclusions
    counters = {d: Counter() for d in days}
    for s in staff:
        if not getattr(s, "is_operational", True):
            continue
        row = a_map.get(s.id, {})
        for d in days:
            c = (row.get(d) or "").upper()
            if not c:
                continue
            # Never count leave/sickness/non-operational placeholders
            if c in get_exclude_from_counters():
                continue
            if c in training_codes:
                continue
            # Explicit exclusions
            if c in ("AL", "NOPS"):
                continue
            # Map EM to Morning counter
            if c == "EM":
                counters[d]["M"] += 1
                continue
            if c == "LA":
                counters[d]["A"] += 1
                continue
            grp = c[:1]
            if grp in ("M", "D", "A", "N"):
                counters[d][grp] += 1
    rag = {}
    for d in days:
        rag[d] = {}
        for code in ("M", "D", "A", "N"):
            have = counters[d][code]
            need = getattr(req, f"req_{code.lower()}") if req else 0
            # Green if we meet/beat requirement, Amber if short by 1, else Red
            rag[d][code] = (
                "green" if have >= need
                else ("amber" if have >= max(0, need - 1) else "red")
            )

    # Fatigue flags keyed by staff id -> {date -> [flags]}
    # Assumes you have a helper like fatigue_flags_for_range(person, days)
    try:
        fatigue = {s.id: fatigue_flags_for_range(s, days) for s in staff}
    except NameError:
        # If your helper is named differently, fall back to empty flags.
        # (Prevents NameError, but you should wire the real helper.)
        fatigue = {s.id: {} for s in staff}


# Pending requests for the month (indexed)
    reqs = ShiftRequest.query.filter(
        ShiftRequest.unit_id == _current_unit_id(),
        ShiftRequest.day >= start, ShiftRequest.day < month_end
    ).all()
    req_pending_map = {
        (r.staff_id, r.day): {"code": r.code, "status": (r.status or "pending").lower()}
        for r in reqs
        if (r.status or "pending").lower() in {"pending", "approved"}
    }

    # --- Unified editability flags ---
    can_edit = can_edit_roster(current_user)
    # If you don't lock months, keep False. If you do, let admin/editor override.
    readonly = False
    # Example for locks:
    # readonly = bool(getattr(req, "is_locked", False)) and not (is_admin_user(current_user) or is_editor_user(current_user))

    month_title = datetime(year, month, 1).strftime("%B %Y")
    today = date.today()

    def _expiry_class(expiry: date | None, ut_flag: bool = False) -> str:
        if ut_flag:
            return "exp-amber"
        if not expiry:
            return ""
        days_to_expiry = (expiry - today).days
        if days_to_expiry < 0:
            return "exp-red"
        if days_to_expiry <= 90:
            return "exp-amber"
        return "exp-green"

    expiry_classes = {}
    for person in staff:
        expiry_classes[person.id] = {
            "medical": _expiry_class(person.medical_expiry),
            "tower": _expiry_class(person.tower_ue_expiry, person.tower_ut),
            "radar": _expiry_class(person.radar_ue_expiry, person.radar_ut),
            "met": _expiry_class(person.met_ue_expiry, person.met_ut),
        }

    # Build row-separator helpers for the template: break between watches
    watch_break_after_ids = []
    prev_watch = None
    prev_id = None
    for s in staff:
        cur_watch = display_watch_by_staff.get(s.id)
        if prev_watch is not None and cur_watch != prev_watch and prev_id is not None:
            # Insert a separator after the previous staff row when the watch changes
            watch_break_after_ids.append(prev_id)
        prev_watch = cur_watch
        prev_id = s.id

    # Call Flask's render_template via the module to avoid any local name shadowing
        import flask as _flask
    return _flask.render_template(
        "roster_month.html",
        ym=ym, year=year, month=month,
        days=days,
        staff=staff,
        a_map=a_map,
        ann_map=ann_map,             # <<< required by template
        counters=counters,
        req=req,                     # <<< ensure 'req' exists for template
        requirement=req,             # <<< keep this if any blocks expect 'requirement'
        rag=rag,
        expiry_classes=expiry_classes,
        fatigue=fatigue,
        watch_break_after_ids=watch_break_after_ids,
        prev_ym=prev_ym, next_ym=next_ym,
        shifts_working=shifts_working,
        shifts_training=shifts_training,
        shifts_non=shifts_non,
        can_edit=can_edit,
        readonly=readonly,
        month_title=month_title,
        today=today,
        req_pending_map=req_pending_map,
        show_ot_finder=True,
        display_watch_by_staff=display_watch_by_staff,
        annotation_groups=get_annotation_groups(),
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


@app.route("/admin/ai/generate/<ym>", methods=["POST"], endpoint="admin_ai_generate")
@login_required
def admin_ai_generate(ym):
    if not is_admin_user(current_user):
        abort(403)
    _validate_csrf()
    y, m = parse_ym(ym)
    try:
        res = generate_month_roster(y, m, current_user) or {}
        n_changed = int(res.get("nights", 0))
        d_changed = int(res.get("days", 0))
        total = n_changed + d_changed
        if total == 0:
            # Give a useful hint if nothing changed
            req = Requirement.query.filter_by(year=y, month=m).first()
            hints = []
            if not req:
                hints.append("no Requirement row")
            else:
                if getattr(req, "req_n", 0) <= 0:
                    hints.append("req_n=0")
                if getattr(req, "req_d", 0) <= 0:
                    hints.append("req_d=0")
            msg = "AI ran but made 0 changes"
            if hints:
                msg += f" ({', '.join(hints)})"
            flash(msg + ".", "info")
        else:
            flash(
                f"AI updated {n_changed} night cells and {d_changed} day cells.", "ok")
    except RuntimeError as e:
        flash(str(e), "error")
    except Exception as e:
        db.session.rollback()
        flash(f"AI run failed: {e}", "error")
    return redirect(url_for("roster_month", ym=ym))


@app.route("/assign/<int:staff_id>/<ym>/<day>", methods=["POST"])
@login_required
@roster_edit_required
def assign_cell(staff_id, ym, day):
    _validate_csrf()
    # parse inputs
    try:
        d = date.fromisoformat(day)
        year, month = parse_ym(ym)
        if d.year != year or d.month != month:
            raise ValueError
    except (TypeError, ValueError):
        abort(400, "Invalid roster date.")
    unit_id = _current_unit_id()
    st = Staff.query.filter_by(id=staff_id, unit_id=unit_id).first_or_404()

    # fetch or create the assignment row for that staff/day
    a = Assignment.query.filter_by(
        unit_id=unit_id, staff_id=staff_id, day=d
    ).first()
    if a is None:
        a = Assignment(unit_id=unit_id, staff=st, day=d, code="OFF")
        db.session.add(a)

    # form fields (each cell posts either code OR annotation)
    code = (request.form.get("code") or "").strip().upper()
    annot = request.form.get("annotation")  # None => no change

    # if a shift code was posted, validate and set it
    if code != "":
        if code in get_banned_roster_codes():
            flash(
                "Leave, sickness and TOIL use must be logged via the form, not the roster grid.", "error")
            return redirect(url_for("roster_month", ym=ym))
        if not get_shift(code):
            flash(f"Unknown shift code '{code}'", "error")
            return redirect(url_for("roster_month", ym=ym))
        a.code = code
        a.source = "manual"

    # if an annotation field was posted, apply delta + update
    if annot is not None:
        if not can_apply_annotations(current_user):
            abort(403)
        old = a.annotation or ""
        newv = (annot or "").strip().upper()
        if old != newv:
            parsed = parse_annotation(newv) if newv else None
            ann_def = None
            if parsed:
                ann_def = AnnotationType.query.filter_by(
                    unit_id=unit_id, code=parsed["type"]
                ).first()
            if newv and (not parsed or not ann_def):
                flash(f"Unknown annotation '{newv}'.", "error")
                return redirect(url_for("roster_month", ym=ym))
            if ann_def and not ann_def.is_active:
                flash(
                    f"{ann_def.code} is inactive and cannot be newly applied.",
                    "error",
                )
                return redirect(url_for("roster_month", ym=ym))
            if ann_def and ann_def.admin_only and not is_admin_user(current_user):
                abort(403)
            annotation_note = (
                request.form.get("annotation_note") or ""
            ).strip()
            if ann_def and ann_def.note_required and not annotation_note:
                flash(f"{ann_def.code} requires a note.", "error")
                return redirect(url_for("roster_month", ym=ym))
            transaction_key = (request.form.get("transaction_key") or "").strip()[:64]
            if transaction_key and AnnotationAudit.query.filter_by(
                unit_id=unit_id, transaction_key=transaction_key
            ).first():
                return redirect(url_for("roster_month", ym=ym))
            _apply_toil_annotation_delta(
                staff=st, old_annot=old, new_annot=newv)
            a.annotation = newv
            if annotation_note:
                a.note = annotation_note[:140]
            if ann_def:
                ann_def.has_been_used = True
            db.session.flush()
            db.session.add(AnnotationAudit(
                unit_id=unit_id, annotation_type_id=ann_def.id if ann_def else None,
                assignment_id=a.id, actor_id=current_user.id,
                action="applied" if newv else "removed",
                old_value=old, new_value=newv,
                transaction_key=transaction_key or None,
            ))

    db.session.commit()
    return redirect(url_for("roster_month", ym=ym))


@app.route("/annotations/bulk", methods=["GET", "POST"])
@login_required
def bulk_annotations():
    """Preview and transactionally apply one annotation across a date range."""
    if not can_edit_roster(current_user) or not can_apply_annotations(current_user):
        abort(403)
    unit_id = _current_unit_id()
    preview = None
    if request.method == "POST":
        _validate_csrf()
        action = (request.form.get("action") or "preview").strip()
        if action == "preview":
            try:
                person_id = int(request.form.get("person_id") or "")
                start = date.fromisoformat(request.form.get("start") or "")
                end = date.fromisoformat(request.form.get("end") or "")
            except (TypeError, ValueError):
                abort(400, "Invalid person or date range.")
            if end < start or (end - start).days > 366:
                abort(400, "Select a range of no more than 367 days.")
            person = Staff.query.filter_by(
                id=person_id, unit_id=unit_id
            ).first_or_404()
            raw_annotation = (
                request.form.get("annotation") or ""
            ).strip().upper()
            parsed = parse_annotation(raw_annotation) if raw_annotation else None
            definition = None
            if parsed:
                definition = AnnotationType.query.filter_by(
                    unit_id=unit_id, code=parsed["type"], is_active=True
                ).first()
            if raw_annotation and (not parsed or not definition):
                abort(400, "Select an active annotation for this airport.")
            if definition and definition.admin_only and not is_admin_user(current_user):
                abort(403)
            note = (request.form.get("note") or "").strip()
            if definition and definition.note_required and not note:
                abort(400, f"{definition.code} requires a note.")
            days = [
                start + timedelta(days=offset)
                for offset in range((end - start).days + 1)
            ]
            existing = {
                row.day: row
                for row in Assignment.query.filter(
                    Assignment.unit_id == unit_id,
                    Assignment.staff_id == person.id,
                    Assignment.day >= start,
                    Assignment.day <= end,
                ).all()
            }
            changes = []
            total_toil_delta = 0
            for target_day in days:
                old_value = (
                    existing[target_day].annotation
                    if target_day in existing else ""
                ) or ""
                if old_value == raw_annotation:
                    continue
                old_toil = _toil_accrual_half_days_from_annotation(
                    parse_annotation(old_value)
                )
                new_toil = _toil_accrual_half_days_from_annotation(parsed)
                delta = new_toil - old_toil
                total_toil_delta += delta
                changes.append({
                    "day": target_day.isoformat(),
                    "old": old_value,
                    "new": raw_annotation,
                    "toil_delta": delta,
                })
            nonce = secrets.token_urlsafe(18)
            session["_bulk_annotation_preview"] = {
                "nonce": nonce,
                "unit_id": unit_id,
                "person_id": person.id,
                "annotation": raw_annotation,
                "note": note[:140],
                "changes": changes,
            }
            preview = {
                "nonce": nonce,
                "person": person,
                "changes": changes,
                "total_toil_delta": total_toil_delta,
            }
        elif action == "apply":
            saved = session.get("_bulk_annotation_preview") or {}
            nonce = (request.form.get("nonce") or "").strip()
            if (
                not nonce
                or not secrets.compare_digest(nonce, str(saved.get("nonce") or ""))
                or saved.get("unit_id") != unit_id
            ):
                abort(409, "The annotation preview has expired.")
            person = Staff.query.filter_by(
                id=saved.get("person_id"), unit_id=unit_id
            ).first_or_404()
            raw_annotation = saved.get("annotation") or ""
            parsed = parse_annotation(raw_annotation) if raw_annotation else None
            definition = None
            if parsed:
                definition = AnnotationType.query.filter_by(
                    unit_id=unit_id, code=parsed["type"], is_active=True
                ).first()
            if raw_annotation and not definition:
                abort(409, "The annotation is no longer active.")
            if definition and definition.admin_only and not is_admin_user(current_user):
                abort(403)
            for change in saved.get("changes") or []:
                target_day = date.fromisoformat(change["day"])
                assignment = Assignment.query.filter_by(
                    unit_id=unit_id, staff_id=person.id, day=target_day
                ).first()
                if not assignment:
                    assignment = Assignment(
                        unit_id=unit_id, staff_id=person.id,
                        day=target_day, code="OFF",
                    )
                    db.session.add(assignment)
                    db.session.flush()
                transaction_key = f"bulk:{nonce}:{assignment.id}"[:64]
                if AnnotationAudit.query.filter_by(
                    unit_id=unit_id, transaction_key=transaction_key
                ).first():
                    continue
                old_value = assignment.annotation or ""
                if old_value == raw_annotation:
                    continue
                _apply_toil_annotation_delta(
                    person, old_value, raw_annotation
                )
                assignment.annotation = raw_annotation
                if saved.get("note"):
                    assignment.note = saved["note"]
                if definition:
                    definition.has_been_used = True
                db.session.add(AnnotationAudit(
                    unit_id=unit_id,
                    annotation_type_id=(
                        definition.id if definition else None
                    ),
                    assignment_id=assignment.id,
                    actor_id=current_user.id,
                    action="bulk_applied" if raw_annotation else "bulk_removed",
                    old_value=old_value,
                    new_value=raw_annotation,
                    transaction_key=transaction_key,
                ))
            db.session.commit()
            session.pop("_bulk_annotation_preview", None)
            flash("Bulk annotation changes applied.", "ok")
            return redirect(url_for("bulk_annotations"))
        else:
            abort(400, "Invalid bulk annotation action.")
    people = Staff.query.filter_by(
        unit_id=unit_id
    ).order_by(Staff.name).all()
    return render_template(
        "annotations_bulk.html",
        people=people,
        annotation_groups=get_annotation_groups(),
        preview=preview,
    )


@app.route("/roster/<ym>/export")
@login_required
def roster_export_csv(ym):
    year, month = parse_ym(ym)
    start, days = month_range(year, month)

    staff = (Staff.query
             .outerjoin(Watch, Staff.watch_id == Watch.id)
             .order_by(Watch.order_index,
                       Staff.name).all())

    a_map = defaultdict(dict)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)
    for a in Assignment.query.filter(Assignment.day >= start, Assignment.day < month_end):
        a_map[a.staff_id][a.day] = a.code

    # compute daily counters + RAG for footer (prefix grouping) — EXCLUDE training shifts & excluded codes
    req = Requirement.query.filter_by(year=year, month=month).first()
    counters = {d: Counter() for d in days}
    for s in staff:
        if not s.is_operational:
            continue
        for d in days:
            c = a_map[s.id].get(d)
            if not c or c in get_exclude_from_counters():
                continue
            sh = get_shift(c) if c else None
            if not c or not sh or sh.is_training:
                continue
            grp = (c or "")[:1].upper()
            if grp in ("M", "D", "A", "N"):
                counters[d][grp] += 1

    # Replicate the RAG calculation used in the HTML view so the CSV footer
    # includes consistent status flags instead of raising a NameError.
    rag = {}
    for d in days:
        rag[d] = {}
        for code in ("M", "D", "A", "N"):
            have = counters[d][code]
            need = getattr(req, f"req_{code.lower()}") if req else 0
            rag[d][code] = (
                "green" if have >= need
                else ("amber" if have >= max(0, need - 1) else "red")
            )

    output = io.StringIO()
    w = csv.writer(output)
    header = ["Name", "Staff #", "Watch"] + [d.isoformat() for d in days]
    w.writerow(header)
    for s in staff:
        row = [s.name, s.staff_no, (s.watch.name.replace(
            "Watch ", "") if s.watch else "-")]
        for d in days:
            row.append(a_map[s.id].get(d, ""))
        w.writerow(row)

    w.writerow([])
    w.writerow(["Totals (M/D/A/N)", "", ""] + [
        f"M:{counters[d]['M']}/{getattr(req, 'req_m', 0)}-{rag[d]['M']} | "
        f"D:{counters[d]['D']}/{getattr(req, 'req_d', 0)}-{rag[d]['D']} | "
        f"A:{counters[d]['A']}/{getattr(req, 'req_a', 0)}-{rag[d]['A']} | "
        f"N:{counters[d]['N']}/{getattr(req, 'req_n', 0)}-{rag[d]['N']}"
        for d in days
    ])

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"roster_{year:04d}-{month:02d}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/roster/<ym>/print")
@login_required
def roster_print_view(ym):
    return redirect(url_for("roster_month", ym=ym))


@app.route("/logout", methods=["GET"], endpoint="logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "ok")
    return redirect(url_for("login"))


# -------------------- Admin --------------------


@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():

    if request.method == "POST":
        form = request.form.get("form", "")

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

            if not all([name, staff_no, username, watch_id]):
                flash("All fields required to create staff.", "error")
            elif Staff.query.filter((Staff.username == username) | (Staff.staff_no == staff_no)).first():
                flash("Username or Staff # already exists.", "error")
            else:
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
                flash("ATCO created.", "ok")
                return redirect(url_for("admin"))

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
            db.session.commit()
            refresh_shift_cache()
            _shift_groups_snapshot.cache_clear()
            flash("Shift deleted.", "ok")
            return redirect(url_for("admin"))

        # Save requirements grid (includes req_d)
        if form == "req":
            yms = request.form.getlist("ym")
            req_m = request.form.getlist("req_m")
            req_d = request.form.getlist("req_d")
            req_a = request.form.getlist("req_a")
            req_n = request.form.getlist("req_n")
            for i in range(len(yms)):
                y, m = [int(x) for x in yms[i].split("-")]
                r = Requirement.query.filter_by(year=y, month=m).first()
                if not r:
                    r = Requirement(year=y, month=m)
                    db.session.add(r)
                r.req_m = int(req_m[i] or 0)
                r.req_d = int(req_d[i] or 0)
                r.req_a = int(req_a[i] or 0)
                r.req_n = int(req_n[i] or 0)
            db.session.commit()
            flash("Requirements saved.", "ok")
            return redirect(url_for("admin"))

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
    months = [(y, m) for y in (2025, 2026) for m in range(1, 13)]
    requirements_by_month = {
        (r.year, r.month): r for r in Requirement.query.all()}
    leaves = Leave.query.order_by(Leave.start.desc()).all()
    return render_template("admin.html",
                           shifts=shifts, staff=staff, watches=watches,
                           months=months, requirements_by_month=requirements_by_month,
                           leaves=leaves,
                           qualification_types=qualification_types)


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
        "leave_codes": {
            "label": "Leave codes",
            "help": "Codes considered leave-like in automatic logic and reports.",
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
                allow_suffix = bool(request.form.get("allow_suffix"))
                suffixes = "".join(sorted({
                    c.upper() for c in (request.form.get("suffixes") or "")
                    if c.isalnum()
                }))
                try:
                    toil_half_days = int(request.form.get("toil_half_days") or 0)
                except ValueError:
                    toil_half_days = 0
                toil_half_days = max(-200, min(toil_half_days, 200))
                tags = ",".join(sorted({
                    t.strip().lower() for t in (request.form.get("tags") or "").split(",") if t.strip()
                }))
                try:
                    sort_order = int(request.form.get("sort_order") or 0)
                except ValueError:
                    sort_order = 0
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
                    allow_suffix=allow_suffix,
                    suffixes=suffixes,
                    toil_half_days=toil_half_days,
                    tags=tags,
                    note_required=bool(request.form.get("note_required")),
                    admin_only=bool(request.form.get("admin_only")),
                    is_active=is_active,
                    sort_order=sort_order,
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
                    "active": ann.is_active, "sort_order": ann.sort_order,
                }
                ann.code = new_code or ann.code
                ann.label = (request.form.get("label") or ann.label or new_code).strip() or new_code
                ann.category = (request.form.get("category") or ann.category or "Other").strip() or "Other"
                requested_colour = request.form.get("colour") or ann.colour
                if not re.fullmatch(r"#[0-9A-Fa-f]{6}", requested_colour or ""):
                    abort(400, "Invalid annotation colour.")
                ann.colour = requested_colour
                ann.description = (request.form.get("description") or ann.description or "")[:1000]
                ann.allow_suffix = bool(request.form.get("allow_suffix"))
                ann.suffixes = "".join(sorted({
                    c.upper() for c in (request.form.get("suffixes") or "")
                    if c.isalnum()
                }))
                try:
                    ann.toil_half_days = int(request.form.get("toil_half_days") or 0)
                except ValueError:
                    ann.toil_half_days = 0
                ann.toil_half_days = max(-200, min(ann.toil_half_days, 200))
                tags = ",".join(sorted({
                    t.strip().lower() for t in (request.form.get("tags") or "").split(",") if t.strip()
                }))
                ann.tags = tags
                ann.note_required = bool(request.form.get("note_required"))
                ann.admin_only = bool(request.form.get("admin_only"))
                try:
                    ann.sort_order = int(request.form.get("sort_order") or ann.sort_order or 0)
                except ValueError:
                    pass
                ann.is_active = bool(request.form.get("is_active"))
                db.session.add(AnnotationAudit(
                    unit_id=unit_id, annotation_type_id=ann.id,
                    actor_id=current_user.id, action="definition_updated",
                    old_value=json.dumps(old_value, sort_keys=True),
                    new_value=json.dumps({
                        "code": ann.code, "label": ann.label, "category": ann.category,
                        "active": ann.is_active, "sort_order": ann.sort_order,
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
                values = _parse_codes_input(request.form.get("values", ""))
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

    annotations = (AnnotationType.query.filter_by(unit_id=unit_id)
                   .order_by(AnnotationType.sort_order, AnnotationType.code)
                   .all())

    settings_view = []
    current_values = {
        "working_codes": sorted(get_working_codes()),
        "leave_codes": sorted(get_leave_codes()),
        "banned_codes": sorted(get_banned_roster_codes()),
        "exclude_from_counters": sorted(get_exclude_from_counters()),
        "non_working_codes": sorted(get_non_working_codes()),
    }
    for key, meta in settings_meta.items():
        settings_view.append({
            "key": key,
            "label": meta["label"],
            "help": meta["help"],
            "value": ", ".join(current_values.get(key, [])),
        })

    return render_template("admin_reference.html",
                           annotations=annotations,
                           settings=settings_view)

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
        s.username = request.form.get("username", s.username).strip()
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

        s.pattern_csv = request.form.get("pattern_csv", s.pattern_csv)
        s.pattern_anchor = _parse_date(request.form.get("pattern_anchor"))

        s.medical_expiry = _parse_date(request.form.get("medical_expiry"))
        s.tower_ue_expiry = _parse_date(request.form.get("tower_ue_expiry"))
        s.radar_ue_expiry = _parse_date(request.form.get("radar_ue_expiry"))
        s.met_ue_expiry = _parse_date(request.form.get("met_ue_expiry"))

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
            db.session.commit()
            flash("Staff updated.", "ok")
        except Exception as e:
            db.session.rollback()
            flash(f"Update failed: {e}", "error")

        return redirect(url_for("admin"))

    watches = Watch.query.order_by(Watch.order_index).all()
    return render_template(
        "staff_edit.html", s=s, watches=watches,
        permissions=user_permissions(s),
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

    db.session.add(StaffWatchHistory(
        staff_id=s.id, watch_id=new_watch_id, effective_date=eff_d))
    db.session.commit()

    log_change("Staff", s.id, "watch_id", s.watch_id,
               new_watch_id, note=f"effective {eff_d.isoformat()}")
    flash("Watch move recorded (effective date).", "ok")
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


@app.route("/admin/ai/rules", methods=["GET", "POST"])
@login_required
@admin_required
def admin_ai_rules():
    ...

    today = date.today()
    ym = request.args.get("ym") or f"{today.year:04d}-{today.month:02d}"
    y, m = parse_ym(ym)

    if request.method == "POST":
        mode = request.form.get("mode", "form")
        if mode == "json":
            txt = request.form.get("rules_json", "").strip()
            try:
                rules = _json.loads(txt) if txt else _default_ai_rules()
            except Exception as e:
                flash(f"Invalid JSON: {e}", "error")
                return redirect(url_for("admin_ai_rules", ym=ym))
        else:
            rules = _default_ai_rules()
            rules["respect_user_requests"] = bool(
                request.form.get("respect_user_requests"))
            rules["equalize_nights"] = bool(
                request.form.get("equalize_nights"))
            rules["nights_in_pairs"] = bool(
                request.form.get("nights_in_pairs"))
            rules["avoid_night_before_AL"] = bool(
                request.form.get("avoid_night_before_AL"))
            rules["fill_short_with_D"] = bool(
                request.form.get("fill_short_with_D"))
            pool = request.form.get(
                "night_pool_watches", "A,B,C,D,E").replace(" ", "").split(",")
            rules["night_pool_watches"] = [p for p in pool if p]
            rules["day_shift_codes"] = {
                "weekday": (request.form.get("day_weekday", "D10") or "D10").upper(),
                "sunday": (request.form.get("day_sunday", "D12") or "D12").upper(),
            }
            non = request.form.get("no_nights_list", "").strip()
            rules["no_nights_list"] = [int(x) for x in re.findall(r"\d+", non)]
            rules["nops_policy"] = {
                "integrate_required": bool(request.form.get("nops_integrate_required")),
                "only_if_needed": bool(request.form.get("nops_only_if_needed")),
            }

        _save_rules_for(y, m, rules)
        flash("AI rules saved.", "ok")
        return redirect(url_for("admin_ai_rules", ym=ym))

    rules = _load_rules_for(y, m)
    staff_map = [(p.name, p.id)
                 for p in Staff.query.order_by(Staff.name.asc()).all()]
    return render_template("ai_rules.html",
                           ym=f"{y:04d}-{m:02d}",
                           rules_json=_json.dumps(rules, indent=2),
                           rules=rules, staff_map=staff_map)


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


@app.route("/leave", methods=["GET", "POST"])
@login_required
def leave():
    # Page visibility: editors & admins only
    if not (is_admin_user(current_user) or getattr(current_user, "role", "") in ("editor", "admin")):
        abort(403)

    staff = Staff.query.order_by(Staff.name).all()

    # ---------- month selection ----------
    today = date.today()
    ym_param = request.args.get("ym") or f"{today.year:04d}-{today.month:02d}"
    year, month = parse_ym(ym_param)
    start_of_month, days = month_range(year, month)
    end_of_month = days[-1]
    month_title = datetime(year, month, 1).strftime("%B %Y")
    prev_ym, next_ym = _clamp_prev_next(year, month)

    if request.method == "POST":
        _validate_csrf()
        # (still restrict POST actions too)
        if not (is_admin_user(current_user) or getattr(current_user, "role", "") in ("editor", "admin")):
            flash("Editors or Admins only.", "error")
            return redirect(url_for("leave", ym=ym_param))

        form = request.form.get("form", "")

        if form == "leave_add":
            staff_id = int(request.form["staff_id"])
            lv_type = request.form["leave_type"].upper().strip()
            start_d = date.fromisoformat(request.form["start"])
            end_d = date.fromisoformat(request.form["end"])

            # NEW: allow TOU8 / TOUI in this form (write to roster, deduct TOIL)
            if lv_type in {"TOU8", "TOUI"}:
                s = tenant_get(Staff, staff_id)
                if not s:
                    abort(404)
                used_per_day_half = 2 if lv_type == "TOU8" else 1
                cur = start_d
                while cur <= end_d:
                    a = Assignment.query.filter_by(
                        unit_id=_current_unit_id(),
                        staff_id=staff_id, day=cur).first()
                    if not a:
                        a = Assignment(staff=s, day=cur)
                    a.code, a.source, a.note, a.annotation = lv_type, "manual", "toil use (via leave form)", ""
                    db.session.add(a)
                    # deduct TOIL balance (half-days)
                    s.toil_half_days = int(
                        (s.toil_half_days or 0) - used_per_day_half)
                    cur += timedelta(days=1)
                db.session.commit()
                flash(
                    f"TOIL use recorded: {lv_type} from {start_d.isoformat()} to {end_d.isoformat()}.", "ok")
                return redirect(url_for("leave", ym=ym_param))

            # Original behaviour: AL/PL/SPL create Leave rows
            if lv_type not in {"AL", "PL", "SPL"}:
                flash("Only AL / PL / SPL / TOU8 / TOUI supported here.", "error")
                return redirect(url_for("leave", ym=ym_param))

            lv = Leave(staff_id=staff_id, leave_type=lv_type,
                       start=start_d, end=end_d)
            db.session.add(lv)
            db.session.commit()
            s = tenant_get(Staff, staff_id)
            if not s:
                abort(404)
            cur = start_d
            while cur <= end_d:
                refresh_day_from_pattern_and_leave(s, cur)
                cur += timedelta(days=1)
            db.session.commit()
            flash("Leave recorded", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "leave_edit":
            lid = int(request.form["leave_id"])
            lv = Leave.query.filter_by(
                id=lid, unit_id=_current_unit_id()
            ).first_or_404()
            old_range = (lv.start, lv.end)
            lv.staff_id = int(request.form["staff_id"])
            lv.leave_type = request.form["leave_type"].upper()
            if lv.leave_type not in {"AL", "PL", "SPL"}:
                flash("Only AL / PL / SPL supported.", "error")
                return redirect(url_for("leave", ym=ym_param))
            lv.start = date.fromisoformat(request.form["start"])
            lv.end = date.fromisoformat(request.form["end"])
            db.session.commit()
            s = tenant_get(Staff, lv.staff_id)
            for rng in [old_range, (lv.start, lv.end)]:
                cur = rng[0]
                while cur <= rng[1]:
                    refresh_day_from_pattern_and_leave(s, cur)
                    cur += timedelta(days=1)
            db.session.commit()
            flash("Leave updated", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "leave_delete":
            lid = int(request.form["leave_id"])
            lv = Leave.query.filter_by(
                id=lid, unit_id=_current_unit_id()
            ).first_or_404()
            s = tenant_get(Staff, lv.staff_id)
            start_d, end_d = lv.start, lv.end
            db.session.delete(lv)
            db.session.commit()
            cur = start_d
            while cur <= end_d:
                refresh_day_from_pattern_and_leave(s, cur)
                cur += timedelta(days=1)
            db.session.commit()
            flash("Leave deleted.", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "sick_add":
            staff_id = int(request.form["staff_id"])
            code = request.form["sick_code"].upper()
            if code not in {"SC", "SSC"}:
                flash("Invalid sickness code.", "error")
                return redirect(url_for("leave", ym=ym_param))
            start_d = date.fromisoformat(request.form["start"])
            end_d = date.fromisoformat(request.form["end"])
            s = tenant_get(Staff, staff_id)
            if not s:
                abort(404)
            cur = start_d
            while cur <= end_d:
                a = Assignment.query.filter_by(
                    unit_id=_current_unit_id(),
                    staff_id=staff_id, day=cur).first()
                if not a:
                    a = Assignment(staff=s, day=cur)
                a.code, a.source, a.note, a.annotation = code, "manual", "sickness", ""
                db.session.add(a)
                cur += timedelta(days=1)
            db.session.commit()
            flash(f"Sickness {code} recorded.", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "sick_edit":
            staff_id = int(request.form["staff_id"])
            start_d = date.fromisoformat(request.form["start"])
            end_d = date.fromisoformat(request.form["end"])
            new_code = request.form["sick_code"].upper()
            if new_code not in {"SC", "SSC"}:
                flash("Invalid sickness code.", "error")
                return redirect(url_for("leave", ym=ym_param))
            cur = start_d
            while cur <= end_d:
                a = Assignment.query.filter_by(
                    unit_id=_current_unit_id(),
                    staff_id=staff_id, day=cur).first()
                if a and a.code in {"SC", "SSC"}:
                    a.code = new_code
                    a.annotation = ""
                    a.source = "manual"
                    a.note = "sickness"
                    db.session.add(a)
                cur += timedelta(days=1)
            db.session.commit()
            flash("Sickness updated.", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "sick_delete":
            staff_id = int(request.form["staff_id"])
            start_d = date.fromisoformat(request.form["start"])
            end_d = date.fromisoformat(request.form["end"])
            s = tenant_get(Staff, staff_id)
            if not s:
                abort(404)
            cur = start_d
            while cur <= end_d:
                a = Assignment.query.filter_by(
                    unit_id=_current_unit_id(),
                    staff_id=staff_id, day=cur).first()
                if a and a.code in {"SC", "SSC"}:
                    db.session.delete(a)
                cur += timedelta(days=1)
            db.session.commit()
            cur = start_d
            while cur <= end_d:
                refresh_day_from_pattern_and_leave(s, cur)
                cur += timedelta(days=1)
            db.session.commit()
            flash("Sickness deleted.", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "toil_use":
            staff_id = int(request.form["staff_id"])
            code = request.form["toil_code"].upper()
            if code not in {"TOU8", "TOUI"}:
                flash("Invalid TOIL code.", "error")
                return redirect(url_for("leave", ym=ym_param))
            day = date.fromisoformat(request.form["day"])
            s = tenant_get(Staff, staff_id)
            if not s:
                abort(404)
            a = Assignment.query.filter_by(
                unit_id=_current_unit_id(),
                staff_id=staff_id, day=day,
            ).first()
            if not a:
                a = Assignment(staff=s, day=day)
            a.code, a.source, a.note, a.annotation = code, "manual", "toil use", ""
            db.session.add(a)
            used_half = 2 if code == "TOU8" else 1
            s.toil_half_days = int((s.toil_half_days or 0) - used_half)
            db.session.commit()
            flash(f"TOIL used: {code} on {day.isoformat()}.", "ok")
            return redirect(url_for("leave", ym=ym_param))

    # ---------- GET: month-filtered data ----------
    leaves = (Leave.query
              .filter(Leave.end >= start_of_month, Leave.start <= end_of_month)
              .order_by(Leave.start.asc())
              .all())
    sickness = (Assignment.query
                .filter(Assignment.code.in_(("SC", "SSC")),
                        Assignment.day >= start_of_month,
                        Assignment.day <= end_of_month)
                .order_by(Assignment.day.asc())
                .all())

    return render_template("leave.html",
                           staff=staff,
                           leaves=leaves,
                           sickness=sickness,
                           ym=f"{year:04d}-{month:02d}",
                           month_title=month_title,
                           prev_ym=prev_ym, next_ym=next_ym)

# -------------------- Staff profile --------------------


@app.route("/staff/<int:sid>")
@login_required
def staff_profile(sid):
    s = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
    if s.id != current_user.id and not is_editor_user(current_user):
        abort(403)
    today = date.today()

    # ensure_month_requirement(today.year, today.month)
    # generate_month(today.year, today.month)

    yr_ago = today - timedelta(days=365)

    al_days = sum((lv.end - lv.start).days + 1 for lv in s.leaves
                  if lv.leave_type == "AL" and lv.end >= yr_ago and lv.start <= today)

    # Sickness = SC/SSC only, counted via Assignments
    q = (Assignment.query
         .filter(Assignment.staff_id == s.id,
                 Assignment.day >= yr_ago,
                 Assignment.day <= today))
    sick_days = sum(1 for a in q.all() if a.code in ("SC", "SSC"))

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
    recent_requests = (
        ShiftRequest.query.filter_by(staff_id=s.id)
        .order_by(ShiftRequest.updated_at.desc())
        .limit(5)
        .all()
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
        recent_requests=recent_requests, notifications=notifications,
        unread_notifications=sum(1 for item in notifications if not item.read_at),
    )


@app.post("/notifications/read")
@login_required
def notifications_read():
    _validate_csrf()
    Notification.query.filter_by(
        unit_id=_current_unit_id(),
        recipient_id=current_user.id, read_at=None
    ).update({"read_at": utcnow()}, synchronize_session=False)
    db.session.commit()
    return redirect(url_for("staff_profile", sid=current_user.id))

# -------------------- Metrics + CSV (date range; FYTD default) --------------------
# (… unchanged metrics functions from your file …)


def _compute_metrics_range(start_day: date, end_day: date):
    assignments = (Assignment.query
                   .filter(Assignment.day >= start_day, Assignment.day <= end_day)
                   .all())

    annotation_snapshot = _annotation_snapshot(
        int(_current_unit_id() or 1)
    )["items"]
    tag_map = {item["code"]: {t for t in item["tags"]} for item in annotation_snapshot}
    label_map = {item["code"]: item["label"] for item in annotation_snapshot}

    ot_columns = [
        {"code": item["code"], "label": label_map[item["code"]]}
        for item in annotation_snapshot
        if item["is_active"] and "ot" in tag_map.get(item["code"], set())
    ]
    ot_order = [col["code"] for col in ot_columns]
    ot_known = set(ot_order)

    staff_by_id = {s.id: s for s in Staff.query.all()}
    metrics_map: dict[int, dict[str, object]] = {}

    for a in assignments:
        s = staff_by_id.get(a.staff_id)
        if not s:
            continue
        if s.id not in metrics_map:
            metrics_map[s.id] = {
                "staff": s,
                "ext_long": 0,
                "ext_short": 0,
                "ext_total": 0,
                "swaps": 0,
                "ot": {code: 0 for code in ot_order},
                "ot_total": 0,
                "aava_total": 0,
            }
        parsed = parse_annotation(a.annotation)
        if not parsed:
            continue
        t = parsed["type"]
        tags = tag_map.get(t, set())
        if "ext_long" in tags:
            metrics_map[s.id]["ext_long"] += 1
            metrics_map[s.id]["ext_total"] += 1
        elif "ext_short" in tags:
            metrics_map[s.id]["ext_short"] += 1
            metrics_map[s.id]["ext_total"] += 1
        elif "swap" in tags:
            metrics_map[s.id]["swaps"] += 1

        if "ot" in tags:
            if t not in ot_known:
                ot_known.add(t)
                ot_order.append(t)
                ot_columns.append({"code": t, "label": label_map.get(t, t)})
            metrics_map[s.id]["ot"].setdefault(t, 0)
            metrics_map[s.id]["ot"][t] += 1
            metrics_map[s.id]["ot_total"] += 1
            if "aava" in tags:
                metrics_map[s.id]["aava_total"] += 1

    staff_order = (Staff.query
                   .outerjoin(Watch, Staff.watch_id == Watch.id)
                   .order_by(Watch.order_index, Staff.name).all())

    staff_metrics = []
    for s in staff_order:
        base = {
            "staff": s,
            "ext_long": 0,
            "ext_short": 0,
            "ext_total": 0,
            "swaps": 0,
            "ot": {code: 0 for code in ot_order},
            "ot_total": 0,
            "aava_total": 0,
        }
        row = metrics_map.get(s.id, base)
        # ensure OT keys match ot_order
        row["ot"] = {code: row["ot"].get(code, 0) for code in ot_order}
        staff_metrics.append(row)

    totals = {
        "ext_long": sum(r["ext_long"] for r in staff_metrics),
        "ext_short": sum(r["ext_short"] for r in staff_metrics),
        "ext_total": sum(r["ext_total"] for r in staff_metrics),
        "swaps": sum(r["swaps"] for r in staff_metrics),
        "ot": {code: sum(r["ot"].get(code, 0) for r in staff_metrics) for code in ot_order},
        "ot_total": sum(r["ot_total"] for r in staff_metrics),
        "aava_total": sum(r["aava_total"] for r in staff_metrics),
    }
    return staff_metrics, totals, ot_columns


def _fy_start_for(d: date) -> date:
    return date(d.year if d.month >= 4 else d.year - 1, 4, 1)


@app.route("/metrics")
@login_required
def metrics():
    if not (is_admin_user(current_user) or getattr(current_user, "role", "") in ("editor", "admin")):
        abort(403)
    # ... existing body unchanged ...
    today = date.today()
    default_start = _fy_start_for(today)
    start_str = request.args.get("start", default_start.isoformat())
    end_str = request.args.get("end", today.isoformat())
    start_day = date.fromisoformat(start_str)
    end_day = date.fromisoformat(end_str)
    staff_metrics, totals, ot_columns = _compute_metrics_range(start_day, end_day)
    return render_template("metrics.html",
                           start=start_day, end=end_day,
                           staff_metrics=staff_metrics, totals=totals,
                           ot_columns=ot_columns)


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


@app.route("/metrics/export")
@login_required
def metrics_export():
    if not is_admin_user(current_user):
        abort(403)
    today = date.today()
    default_start = _fy_start_for(today)
    start_day = date.fromisoformat(
        request.args.get("start", default_start.isoformat()))
    end_day = date.fromisoformat(request.args.get("end", today.isoformat()))
    staff_metrics, totals, ot_columns = _compute_metrics_range(start_day, end_day)

    output = io.StringIO()
    w = csv.writer(output)
    header = ["ATCO", "Staff #", "Watch",
              "Ext L", "Ext S", "Ext Total", "Swaps"]
    header.extend([col["code"] for col in ot_columns])
    header.extend(["AAVA Total", "OT Total"])
    w.writerow(header)
    for row in staff_metrics:
        s = row["staff"]
        watch = s.watch.name.replace("Watch ", "") if s.watch else "-"
        ot_values = [row["ot"].get(col["code"], 0) for col in ot_columns]
        w.writerow([s.name, s.staff_no, watch,
                    row["ext_long"], row["ext_short"], row["ext_total"], row["swaps"]] +
                   ot_values + [row["aava_total"], row["ot_total"]])
    w.writerow([])
    total_row = ["All ATCOs", "", "",
                 totals["ext_long"], totals["ext_short"], totals["ext_total"], totals["swaps"]]
    total_row.extend([totals["ot"].get(col["code"], 0) for col in ot_columns])
    total_row.extend([totals["aava_total"], totals["ot_total"]])
    w.writerow(total_row)

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"overtime-swap-ext-count_{start_day.isoformat()}_to_{end_day.isoformat()}.csv"
    return Response(csv_bytes,
                    mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


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
    sh = get_shift(shift_code)
    if not (chosen_date and sh and sh.is_working):
        return [], "Please select a valid date and working shift."

    lookahead_days = 14
    ensure_assignments_for_range(chosen_date - timedelta(days=30),
                                 chosen_date + timedelta(days=lookahead_days))

    staff_members = (
        Staff.query
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .filter(
            Staff.is_operational.is_(True),
            Staff.membership_status == "active",
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
    for s in staff_members:
        if s.exclude_from_ot:
            continue

        if not _staff_has_shift_qualification(s, sh, chosen_date):
            continue

        a_today = Assignment.query.filter_by(
            staff_id=s.id, day=chosen_date).first()
        code_today = a_today.code if a_today else "OFF"
        sh_today = get_shift(code_today)
        if sh_today and sh_today.is_working:
            continue

        if code_today in ("SC", "SSC"):
            continue

        if not _has_in_date_ue(s, chosen_date):
            continue

        if _worked_like_consecutive_days(s, chosen_date - timedelta(days=1), lookback_days=6) >= 6:
            continue

        future_issues = would_create_new_fatigue_issues(
            s, chosen_date, shift_code, lookback_days=30, lookahead_days=lookahead_days
        )

        d24_warnings = []
        blocking_issues = {}
        for _d, _lst in future_issues.items():
            keep = []
            for _f in _lst:
                if _f.startswith("D24 rest deficit"):
                    d24_warnings.append(f"{_d.isoformat()}: {_f}")
                else:
                    keep.append(_f)
            if keep:
                blocking_issues[_d] = keep

        if any(blocking_issues.values()):
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
    return results, None


@app.route("/overtime", methods=["GET", "POST"])
@login_required
def overtime():
    if not (is_admin_user(current_user) or getattr(current_user, "role", "") in ("editor", "admin")):
        abort(403)

    shifts = ShiftType.query.filter_by(
        is_working=True).order_by(ShiftType.code).all()
    results = []
    chosen_date = None
    chosen_shift = None
    selected_staff_ids: set[str] = set()
    sms_body = ""
    searched = request.method == "POST"

    if request.method == "POST":
        _validate_csrf()
        action = request.form.get("action", "find")
        chosen_date = _parse_date(request.form.get("date"))
        chosen_shift = (request.form.get("shift_code") or "").upper().strip()
        selected_staff_ids = {sid for sid in request.form.getlist("staff_ids")}
        sms_body = (request.form.get("message") or "").strip()

        results, error_msg = _compute_overtime_candidates(chosen_date, chosen_shift)

        if action == "send_sms":
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
                           chosen_date=chosen_date, chosen_shift=chosen_shift,
                           sms_body=sms_body, sms_ready=sms_ready,
                           selected_staff_ids=selected_staff_ids,
                           searched=searched)

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
    s = Staff.query.filter_by(
        id=sid, unit_id=_current_unit_id()
    ).first_or_404()
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
        sh = get_shift(a.code)
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


def _leave_summary_for_month(year: int, month: int):
    start, days = month_range(year, month)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)

    a_map = defaultdict(dict)
    for a in Assignment.query.filter(Assignment.day >= start, Assignment.day < month_end):
        a_map[a.staff_id][a.day] = a.code

    staff = (Staff.query
             .outerjoin(Watch, Staff.watch_id == Watch.id)
             .order_by(Watch.order_index, Staff.name).all())

    codes_sorted = ["AL"]  # only AL
    rows = []
    totals = Counter()

    for s in staff:
        counts = {c: 0 for c in codes_sorted}
        for d in days:
            code = a_map[s.id].get(d)
            if code == "AL":
                counts["AL"] += 1
        total = sum(counts.values())
        for c, v in counts.items():
            totals[c] += v
        rows.append({"staff": s, "counts": counts, "total": total})

    grand_total = sum(totals.values())
    return rows, codes_sorted, totals, grand_total, days


@app.route("/reports/leave/<ym>")
@login_required
def report_leave(ym):
    if not is_admin_user(current_user):
        abort(403)
    year, month = parse_ym(ym)
    ensure_month_requirement(year, month)
    generate_month(year, month)
    rows, codes, totals, grand_total, days = _leave_summary_for_month(
        year, month)
    month_title = datetime(year, month, 1).strftime("%B %Y")
    return render_template("report_leave.html",
                           ym=ym, year=year, month=month, month_title=month_title,
                           rows=rows, codes=codes,
                           totals=totals, grand_total=grand_total)


@app.route("/reports/leave.csv")
@login_required
def report_leave_csv():
    if not is_admin_user(current_user):
        abort(403)
    ym = request.args.get("ym")
    if not ym:
        abort(400)
    year, month = parse_ym(ym)
    ensure_month_requirement(year, month)
    generate_month(year, month)
    rows, codes, totals, grand_total, days = _leave_summary_for_month(
        year, month)

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["Name", "Staff #", "Watch", *codes, "Total"])
    for r in rows:
        s = r["staff"]
        watch = s.watch.name.replace("Watch ", "") if s.watch else "-"
        w.writerow([s.name, s.staff_no, watch, *[r["counts"].get(c, 0)
                   for c in codes], r["total"]])
    w.writerow([])
    w.writerow(["Totals", "", "", *[totals.get(c, 0)
               for c in codes], grand_total])

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"leave_{year:04d}-{month:02d}.csv"
    return Response(csv_bytes,
                    mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


# ===== Leave-Year report (per-person config; AL only; includes TOIL days) =====
# (unchanged from your post)

def _current_leave_year_window(s: Staff, today: date | None = None):
    today = today or date.today()
    start_month = s.leave_year_start_month or 4
    start_year = today.year if today.month >= start_month else today.year - 1
    start = date(start_year, start_month, 1)
    end_year = start_year + 1 if start_month > 1 else start_year + 1
    end_month = start_month - 1 if start_month > 1 else 12
    _, end_days = month_range(end_year, end_month)
    end = end_days[-1]
    return start, end


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


@app.route("/reports/leave-year")
@login_required
def report_leave_year():
    if not is_admin_user(current_user):
        abort(403)
    today = date.today()
    people = (Staff.query
              .outerjoin(Watch, Staff.watch_id == Watch.id)
              .order_by(Watch.order_index, Staff.name).all())
    rows = []
    for s in people:
        start, end = _current_leave_year_window(s, today)
        q = (Assignment.query
             .filter(Assignment.staff_id == s.id,
                     Assignment.day >= start,
                     Assignment.day <= end))
        al_taken = sum(1 for a in q.all() if a.code == "AL")
        entitlement = (s.leave_entitlement_days or 0)
        ph = (s.leave_public_holidays or 0)
        carry = (s.leave_carryover_days or 0)
        remaining = entitlement + ph + carry - al_taken
        acc_half, use_half = _toil_accrued_used_in_range_half_days(
            s.id, start, end)
        rows.append({
            "staff": s,
            "watch": s.watch.name.replace("Watch ", "") if s.watch else "-",
            "leave_year_start": start,
            "leave_year_end": end,
            "entitlement": entitlement,
            "public_holidays": ph,
            "carryover": carry,
            "al_taken": al_taken,
            "remaining": remaining,
            "toil_accrued_days": acc_half / 2.0,
            "toil_used_days": use_half / 2.0,
            "toil_balance_days": (s.toil_half_days or 0) / 2.0,
        })
    return render_template("report_leave_year.html", rows=rows, today=today)


# ===== Sickness Report (unchanged) =====


def _group_consecutive_days(days_set):
    if not days_set:
        return 0
    days = sorted(days_set)
    groups = 0
    prev = None
    for d in days:
        if prev is None or (d - prev).days > 1:
            groups += 1
        prev = d
    return groups


@app.route("/reports/sickness")
@login_required
def report_sickness():
    if not is_admin_user(current_user):
        abort(403)
    today = date.today()
    start = today - timedelta(days=365)
    people = (Staff.query
              .outerjoin(Watch, Staff.watch_id == Watch.id)
              .order_by(Watch.order_index, Staff.name).all())
    rows = []
    for s in people:
        q = (Assignment.query
             .filter(Assignment.staff_id == s.id,
                     Assignment.day >= start,
                     Assignment.day <= today))
        sick_days = sorted([a.day for a in q.all() if a.code in ("SC", "SSC")])
        total = len(sick_days)
        groups = _group_consecutive_days(set(sick_days))
        rows.append({
            "staff": s, "watch": s.watch.name.replace("Watch ", "") if s.watch else "-",
            "total": total, "groups": groups
        })
    return render_template("report_sickness.html", start=start, end=today, rows=rows)


# -------------------- Request Sheets (shift requests) --------------------


def _unit_request_rules(unit_id: int | None = None) -> tuple[int, int]:
    unit = db.session.get(Unit, unit_id or _current_unit_id())
    months = max(1, min(int(getattr(unit, "request_months_ahead", 3) or 3), 24))
    lock_day = max(1, min(int(getattr(unit, "request_lock_day", 20) or 20), 28))
    return months, lock_day


def _lock_date_for_target_month(y: int, m: int, unit_id: int | None = None):
    _, lock_day = _unit_request_rules(unit_id)
    prev_m = m - 1
    prev_y = y
    if prev_m <= 0:
        prev_m = 12
        prev_y -= 1
    return date(prev_y, prev_m, lock_day)


def _is_month_locked(y: int, m: int, today: date | None = None, unit_id: int | None = None):
    today = today or date.today()
    return today >= _lock_date_for_target_month(y, m, unit_id)


def _add_months(first: date, count: int) -> date:
    idx = first.year * 12 + first.month - 1 + count
    return date(idx // 12, idx % 12 + 1, 1)


def _request_date_bounds(today: date, unit_id: int) -> tuple[date, date]:
    months, _ = _unit_request_rules(unit_id)
    start = _add_months(date(today.year, today.month, 1), 1)
    next_after_window = _add_months(start, months)
    return start, next_after_window - timedelta(days=1)


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
    db.session.add(Notification(
        unit_id=req.unit_id,
        recipient_id=req.staff_id,
        kind=f"shift_request_{req.status}",
        message=f"Your {req.code} request for {req.day.isoformat()} is now {req.status}.",
    ))


def _safe_request_admin_month(raw_value: str | None, fallback: date) -> str:
    """Return a canonical admin month without allowing malformed redirects."""
    candidate = (raw_value or "").strip()
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", candidate):
        return f"{fallback.year:04d}-{fallback.month:02d}"
    return candidate


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


@app.route("/requests", methods=["GET", "POST"])
@login_required
def requests_page():
    today = date.today()
    unit_id = _current_unit_id()
    if not unit_id:
        abort(403)
    months_ahead, _ = _unit_request_rules(unit_id)
    first_allowed, last_allowed = _request_date_bounds(today, unit_id)

    # ---- user/editor: show configured future months they can request into ----
    months = []
    base_y, base_m = today.year, today.month
    for k in range(1, months_ahead + 1):
        t_m = base_m + k
        t_y = base_y + (t_m - 1) // 12
        t_m = ((t_m - 1) % 12) + 1
        months.append((t_y, t_m))

    # ---- POST (create/delete own requests) ----
    if request.method == "POST":
        _validate_csrf()
        form = request.form.get("form", "")
        if form == "add":
            try:
                day = date.fromisoformat(request.form.get("day", ""))
            except (TypeError, ValueError):
                flash("Enter a valid request date.", "error")
                return redirect(url_for("requests_page"))
            code = (request.form.get("code") or "").upper().strip()
            comment = (request.form.get("comment") or "").strip()
            if len(comment) > 500:
                flash("Requester comments are limited to 500 characters.", "error")
                return redirect(url_for("requests_page"))
            shift = ShiftType.query.filter_by(
                unit_id=unit_id, code=code, is_active=True,
                is_requestable=True, is_working=True,
            ).first()
            if not shift:
                flash("That shift is inactive or cannot be requested.", "error")
                return redirect(url_for("requests_page"))
            if day < first_allowed or day > last_allowed:
                flash(f"Requests must be between {first_allowed} and {last_allowed}.", "error")
                return redirect(url_for("requests_page"))
            if _is_month_locked(day.year, day.month, today, unit_id):
                flash("Requests for that month are locked.", "error")
                return redirect(url_for("requests_page"))
            ex = ShiftRequest.query.filter_by(
                unit_id=unit_id, staff_id=current_user.id, day=day).first()
            if not ex:
                ex = ShiftRequest(
                    unit_id=unit_id, staff_id=current_user.id, day=day,
                    code=code, requester_comment=comment,
                )
                db.session.add(ex)
                db.session.flush()
                _request_audit(ex, current_user.id, "created", {}, {
                    "code": code, "comment": comment, "status": "pending",
                })
            else:
                if ex.status != "pending":
                    flash("Only pending requests can be edited.", "error")
                    return redirect(url_for("requests_page"))
                old = {"code": ex.code, "comment": ex.requester_comment, "status": ex.status}
                ex.code = code
                ex.requester_comment = comment
                ex.updated_at = utcnow()
                ex.submitted_at = utcnow()
                ex.status = "pending"
                ex.admin_response = ""
                ex.responded_by_id = None
                ex.responded_at = None
                _request_audit(ex, current_user.id, "updated", old, {
                    "code": code, "comment": comment, "status": "pending",
                })
            db.session.commit()
            flash("Request saved.", "ok")
            return redirect(url_for("requests_page"))

        if form == "del":
            try:
                rid = int(request.form.get("rid", ""))
            except (TypeError, ValueError):
                abort(400)
            req = ShiftRequest.query.filter_by(id=rid, unit_id=unit_id).first_or_404()
            if req.staff_id != current_user.id:
                abort(403)
            if req.status != "pending":
                abort(409, "Only pending requests can be cancelled.")
            if _is_month_locked(req.day.year, req.day.month, today, unit_id):
                flash("Requests for that month are locked.", "error")
                return redirect(url_for("requests_page"))
            old = req.status
            req.status = "cancelled"
            req.cancelled_at = utcnow()
            req.updated_at = utcnow()
            _request_audit(req, current_user.id, "cancelled", old, req.status, "Cancelled by requester")
            db.session.commit()
            flash("Request cancelled; its history has been preserved.", "ok")
            return redirect(url_for("requests_page"))
        abort(400)

    # ---- My requests (everyone) ----
    my_reqs = ShiftRequest.query.filter_by(unit_id=unit_id, staff_id=current_user.id).all()
    req_map = defaultdict(dict)
    for r in my_reqs:
        req_map[(r.day.year, r.day.month)][r.day] = r

    all_shifts = ShiftType.query.filter_by(
        unit_id=unit_id, is_active=True, is_requestable=True
    ).order_by(ShiftType.code).all()
    codes = [s.code for s in all_shifts]

    # ---- Admin: month-selectable “All requests” panel ----
    admin_view = is_admin_user(current_user)
    admin_grouped = {}
    admin_ym = None
    admin_month_title = None
    admin_prev_ym = None
    admin_next_ym = None
    admin_total = 0

    if admin_view:
        # default to current month unless ?ym=YYYY-MM provided
        admin_ym = _safe_request_admin_month(request.args.get("ym"), today)
        ay, am = parse_ym(admin_ym)
        start_of_month, month_days = month_range(ay, am)
        end_of_month = month_days[-1]

        admin_month_title = datetime(ay, am, 1).strftime("%B %Y")
        admin_prev_ym, admin_next_ym = _clamp_prev_next(ay, am)

        # fetch only the chosen month; order by day then staff name
        admin_requests = (ShiftRequest.query
                          .join(Staff, ShiftRequest.staff_id == Staff.id)
                          .filter(ShiftRequest.unit_id == unit_id,
                                  ShiftRequest.day >= start_of_month,
                                  ShiftRequest.day <= end_of_month)
                          .order_by(ShiftRequest.day.asc(), Staff.name.asc())
                          .all())

        # group by day for a tidy display
        grouped = defaultdict(list)
        for r in admin_requests:
            grouped[r.day].append(r)
        admin_grouped = dict(grouped)
        admin_total = len(admin_requests)

    return render_template("requests.html",
                           months=months,
                           is_locked=_is_month_locked,
                           req_map=req_map,
                           codes=codes,
                           # admin block
                           admin_view=admin_view,
                           admin_grouped=admin_grouped,
                           admin_total=admin_total,
                           admin_ym=admin_ym,
                           admin_month_title=admin_month_title,
                           admin_prev_ym=admin_prev_ym,
                           admin_next_ym=admin_next_ym,
                           request_lock_day=_unit_request_rules(unit_id)[1],
                           first_allowed=first_allowed,
                           last_allowed=last_allowed)


# >>> Admin can respond to a specific request


@app.route("/admin/requests/<int:rid>/respond", methods=["POST"])
@login_required
def admin_request_respond(rid):
    if not is_admin_user(current_user):
        abort(403)
    _validate_csrf()
    unit_id = _current_unit_id()
    r = ShiftRequest.query.filter_by(id=rid, unit_id=unit_id).first_or_404()
    action = (request.form.get("action") or "status").strip()
    if action not in {"status", "approve_only", "approve_apply"}:
        abort(400, "Invalid request action.")
    return_month = _safe_request_admin_month(request.form.get("ym"), date.today())
    response = (request.form.get("admin_response") or "").strip()
    if len(response) > 500:
        abort(400, "Response is limited to 500 characters.")
    requested_status = (request.form.get("status") or "").strip().lower()
    if action == "approve_only":
        requested_status = "approved"
    elif action == "approve_apply":
        requested_status = "fulfilled"
    if requested_status not in REQUEST_STATUSES:
        abort(400, "Invalid request status.")
    if (r.status or "pending") not in REQUEST_STATUSES:
        abort(409, "The request has an invalid current status.")
    if not r.staff or r.staff.unit_id != unit_id:
        abort(409, "The requester does not belong to this airport.")

    old = {"status": r.status, "response": r.admin_response}
    if action == "approve_apply":
        if r.status not in {"pending", "approved"}:
            abort(409, "Only pending or approved requests can be applied.")
        shift = ShiftType.query.filter_by(
            unit_id=unit_id, code=r.code, is_active=True,
            is_requestable=True, is_working=True,
        ).first()
        if not shift:
            abort(409, "The requested shift is no longer valid.")
        if _is_month_locked(r.day.year, r.day.month, unit_id=unit_id):
            abort(409, "The roster month is locked.")
        published = RosterPublication.query.filter_by(
            unit_id=unit_id, year=r.day.year, month=r.day.month,
            state="published",
        ).first()
        if published:
            abort(
                409,
                "The roster is published. Create a controlled superseding "
                "version before applying this request.",
            )
        conflicts = list(would_create_new_fatigue_issues(r.staff, r.day, r.code).values())
        if not _staff_has_shift_qualification(r.staff, shift, r.day):
            conflicts.append(["Required qualification is missing or expired."])
        override = request.form.get("confirm_override") == "yes"
        if override and not can_override_roster_conflicts(current_user):
            abort(403, "You do not have permission to override conflicts.")
        if conflicts and override and len(response) < 10:
            abort(400, "A reason of at least 10 characters is required.")
        if conflicts and not override:
            warning_text = "; ".join(
                str(item)
                for group in conflicts
                for item in (group if isinstance(group, (list, tuple, set)) else [group])
            )
            flash(
                "Applying this request has conflicts: "
                f"{warning_text[:700]}. Review and confirm the permitted override.",
                "error",
            )
            return redirect(url_for("requests_page", ym=return_month))
        assignment = Assignment.query.filter_by(
            unit_id=unit_id, staff_id=r.staff_id, day=r.day
        ).first()
        if not assignment:
            assignment = Assignment(unit_id=unit_id, staff_id=r.staff_id, day=r.day)
            db.session.add(assignment)
        assignment.code = r.code
        assignment.source = "request"
        assignment.note = f"Applied from shift request #{r.id}"
        db.session.flush()
        r.resulting_assignment_id = assignment.id
        r.fulfilled_at = utcnow()
        requested_status = "fulfilled"
    else:
        if requested_status == "fulfilled":
            abort(400, "Fulfilment is only available through Approve and apply.")
        allowed = REQUEST_TRANSITIONS.get(r.status or "pending", frozenset())
        if requested_status not in allowed:
            abort(
                409,
                f"Transition from {r.status or 'pending'} to "
                f"{requested_status} is not permitted.",
            )
        if r.status == "approved" and requested_status in {
            "rejected", "cancelled"
        } and len(response) < 10:
            abort(400, "Changing an approved request requires an audited reason.")

    r.admin_response = response
    r.status = requested_status
    r.responded_by_id = getattr(current_user, "id", None)
    r.responded_at = utcnow()
    r.updated_at = utcnow()
    if requested_status == "cancelled" and r.cancelled_at is None:
        r.cancelled_at = utcnow()
    _request_audit(r, current_user.id, action, old, {
        "status": r.status, "response": response,
        "assignment_id": r.resulting_assignment_id,
    }, response)
    if old["status"] != r.status:
        _notify_requester(r)
    db.session.commit()
    flash("Response saved.", "ok")
    return redirect(url_for("requests_page", ym=return_month))


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
            unit_id = int(request.form.get("unit_id") or 0)
            unit = db.session.get(Unit, unit_id)
            routing = db.session.get(DatabaseRoutingMetadata, unit_id)
            if not unit or unit.status == "platform_control" or not routing:
                abort(404)
            routing.attempt_count = int(routing.attempt_count or 0) + 1
            routing.last_attempt_at = utcnow()
            routing.last_error_code = ""
            db.session.commit()
            try:
                if not re.fullmatch(
                    r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL",
                    routing.secret_name or "",
                ):
                    raise RuntimeError("invalid_secret_reference")
                operational_url = os.environ.get(routing.secret_name)
                if not operational_url:
                    raise RuntimeError("database_secret_unavailable")
                control_url = (
                    os.environ.get("CONTROL_DATABASE_URL")
                    or app.config["SQLALCHEMY_DATABASE_URI"]
                )
                if operational_url == control_url:
                    raise RuntimeError("database_not_isolated")
                routing.provisioning_state = "database_configured"
                db.session.commit()
                from scripts.migrate_all_databases import upgrade_database
                routing.migration_version = upgrade_database(
                    operational_url, "operational"
                )
                routing.provisioning_state = "migrations_complete"
                from sqlalchemy import create_engine, inspect
                health_engine = create_engine(
                    operational_url, pool_pre_ping=True
                )
                try:
                    with health_engine.connect() as connection:
                        connection.execute(text("SELECT 1"))
                    required = {
                        "staff", "assignment", "shift_request",
                        "qualification_type",
                    }
                    if not required.issubset(
                        set(inspect(health_engine).get_table_names())
                    ):
                        raise RuntimeError("schema_health_check_failed")
                finally:
                    health_engine.dispose()
                invitation = SecureInvitation.query.filter_by(
                    unit_id=unit.id, role="UnitAdmin",
                    accepted_at=None, disabled_at=None,
                ).first()
                raw_token = None
                if not invitation:
                    raw_token = secrets.token_urlsafe(32)
                    invitation = SecureInvitation(
                        unit_id=unit.id,
                        token_digest=hashlib.sha256(
                            raw_token.encode()
                        ).hexdigest(),
                        role="UnitAdmin",
                        expires_at=utcnow() + timedelta(days=7),
                    )
                    db.session.add(invitation)
                routing.provisioning_state = "invitation_issued"
                routing.health = "healthy"
                routing.ready_at = utcnow()
                db.session.add(SuperAdminAudit(
                    actor_identity_id=platform_actor.id,
                    unit_id=unit.id,
                    action="airport_provisioned",
                    safe_summary=(
                        f"Operational schema ready for {unit.code}; "
                        "bootstrap invitation issued."
                    ),
                ))
                db.session.commit()
                if raw_token:
                    flash(
                        "Provisioning completed. Transfer this one-time "
                        f"bootstrap link securely: {url_for('accept_invitation', token=raw_token, _external=True)}",
                        "ok",
                    )
                else:
                    flash(
                        "Provisioning is already complete; the existing "
                        "invitation remains valid.", "ok",
                    )
            except Exception as exc:
                db.session.rollback()
                routing = db.session.get(DatabaseRoutingMetadata, unit_id)
                allowed = {
                    "invalid_secret_reference",
                    "database_secret_unavailable",
                    "database_not_isolated",
                    "schema_health_check_failed",
                }
                code = (
                    str(exc) if str(exc) in allowed
                    else "database_provisioning_failed"
                )
                routing.provisioning_state = "failed"
                routing.last_error_code = code
                routing.health = "failed"
                db.session.add(SuperAdminAudit(
                    actor_identity_id=platform_actor.id,
                    unit_id=unit_id,
                    action="airport_provisioning_failed",
                    safe_summary=f"Provisioning failed with code {code}.",
                ))
                db.session.commit()
                flash(
                    f"Provisioning did not complete ({code}). Correct the "
                    "configuration and retry.", "error",
                )
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
        routing = db.session.get(DatabaseRoutingMetadata, unit.id)
        activity = db.session.query(
            db.func.coalesce(db.func.sum(AggregateUsageEvent.count), 0)
        ).filter(AggregateUsageEvent.unit_id == unit.id).scalar()
        bootstrap = SecureInvitation.query.filter_by(
            unit_id=unit.id, role="UnitAdmin"
        ).order_by(SecureInvitation.id.desc()).first()
        if not bootstrap:
            bootstrap_status = "not issued"
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
        })
    return render_template(
        "platform_admin.html", rows=rows,
        feature_keys=sorted(PLATFORM_FEATURE_FLAGS),
    )


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
                from account_limits import lock_unit_capacity
                lock_unit_capacity(db, Unit, UnitMembership, unit_id)
                raw_token = secrets.token_urlsafe(32)
                invitation = SecureInvitation(
                    unit_id=unit_id,
                    token_digest=hashlib.sha256(
                        raw_token.encode()
                    ).hexdigest(),
                    role=role,
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
                "Invitation created. Copy this link now; it is shown only "
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
    return render_template(
        "unit_accounts.html", unit=unit, memberships=memberships,
        active_count=active_count,
        pending_invitations=pending_invitations,
    )


class SignupWorkflowError(RuntimeError):
    pass


def _normalized_login(value: str) -> str:
    return value.strip().casefold()


def _run_invitation_signup(
    invitation, unit, name, username, password, fail_after=None,
):
    """Resume an invitation saga without claiming cross-DB atomicity."""
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
        raise SignupWorkflowError(
            "This invitation already has an incomplete setup attempt."
        )
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
            marker = f"{unit.code}-SIGNUP-{workflow.id}"
            staff = Staff.query.filter_by(staff_no=marker).first()
            if not staff:
                if Staff.query.filter(
                    db.func.lower(Staff.username) == normalized
                ).first():
                    raise SignupWorkflowError(
                        "That login identifier is unavailable."
                    )
                role_map = {
                    "UnitAdmin": "admin", "RosterEditor": "editor",
                    "WatchManager": "user", "StaffUser": "user",
                    "ReadOnlyAuditor": "auditor",
                }
                staff = Staff(
                    unit_id=unit.id, username=normalized, name=name[:80],
                    staff_no=marker, role=role_map[invitation.role],
                    is_wm=invitation.role == "WatchManager",
                    is_operational=False, membership_status="pending",
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
    if request.method == "POST":
        _validate_csrf()
        if DEPLOYMENT_ENV == "production" and not routing:
            abort(503, "Operational database routing is unavailable.")
        g.tenant_context_token = bind_authenticated_unit(
            invitation.unit_id,
            routing.secret_name if routing else None,
        )
        name = (request.form.get("name") or "").strip()
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        if not name or not re.fullmatch(r"[a-z0-9._-]{3,120}", username):
            flash("Enter a name and a valid username.", "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit
            ), 400
        if len(password) < 12:
            flash("Use a password of at least 12 characters.", "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit
            ), 400
        try:
            _run_invitation_signup(
                invitation, unit, name, username, password
            )
        except (SignupWorkflowError, ValueError) as exc:
            flash(str(exc), "error")
            return render_template(
                "invitation_accept.html", invitation=invitation, unit=unit
            ), 409
        flash("Account created. Sign in and configure MFA.", "ok")
        return redirect(url_for("login"))
    return render_template(
        "invitation_accept.html", invitation=invitation, unit=unit
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
        ("Compliance review available", True, "compliance_centre"),
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


@app.route("/operations/<ym>", methods=["GET", "POST"])
@login_required
def operations_assurance(ym):
    if not is_admin_user(current_user):
        abort(403)
    year, month = _compliance_month(ym)
    if request.method == "POST":
        _validate_csrf()
        action = (request.form.get("action") or "").strip()
        try:
            if action == "create_position":
                code = (request.form.get("code") or "").strip().upper()
                label = (request.form.get("label") or "").strip()
                if not re.fullmatch(r"[A-Z0-9_-]{2,30}", code) or not label:
                    raise ValueError("Position code and label are required.")
                db.session.add(OperationalPosition(
                    unit_id=_current_unit_id(), code=code, label=label,
                    description=(request.form.get("description") or "").strip()[:1000],
                    is_safety_critical=request.form.get("is_safety_critical") == "on",
                ))
            elif action == "grant_endorsement":
                person_id = int(request.form.get("person_id") or 0)
                position_id = int(request.form.get("position_id") or 0)
                person = Staff.query.filter_by(id=person_id, is_operational=True).first_or_404()
                position = OperationalPosition.query.filter_by(id=position_id).first_or_404()
                row = PositionEndorsement.query.filter_by(
                    person_id=person.id, position_id=position.id
                ).first()
                if not row:
                    row = PositionEndorsement(
                        unit_id=_current_unit_id(), person_id=person.id,
                        position_id=position.id,
                    )
                    db.session.add(row)
                row.valid_from = date.fromisoformat(request.form["valid_from"])
                valid_until = (request.form.get("valid_until") or "").strip()
                row.valid_until = date.fromisoformat(valid_until) if valid_until else None
                row.status = "valid"
                row.restrictions = (request.form.get("restrictions") or "").strip()[:1000]
            elif action == "set_position_requirement":
                position_id = int(request.form.get("position_id") or 0)
                OperationalPosition.query.filter_by(id=position_id, is_active=True).first_or_404()
                duty_day = date.fromisoformat(request.form["day"])
                shift_code = (request.form.get("shift_code") or "").strip().upper()
                ShiftType.query.filter_by(code=shift_code, is_active=True).first_or_404()
                required = max(0, int(request.form.get("required_count") or 0))
                contingency = max(0, int(request.form.get("contingency_count") or 0))
                row = PositionRequirement.query.filter_by(
                    day=duty_day, shift_code=shift_code, position_id=position_id
                ).first()
                if not row:
                    row = PositionRequirement(
                        unit_id=_current_unit_id(), day=duty_day,
                        shift_code=shift_code, position_id=position_id,
                    )
                    db.session.add(row)
                row.required_count = required
                row.contingency_count = contingency
            elif action == "add_break":
                duty_day = date.fromisoformat(request.form["day"])
                start_time = time.fromisoformat(request.form["start_time"])
                end_time = time.fromisoformat(request.form["end_time"])
                if end_time <= start_time:
                    raise ValueError("Break end must be after its start.")
                person_id = int(request.form.get("person_id") or 0)
                Staff.query.filter_by(id=person_id, is_operational=True).first_or_404()
                position_id = int(request.form.get("position_id") or 0) or None
                if position_id:
                    OperationalPosition.query.filter_by(id=position_id).first_or_404()
                db.session.add(BreakPlan(
                    unit_id=_current_unit_id(), day=duty_day,
                    person_id=person_id, position_id=position_id,
                    start_time=start_time, end_time=end_time,
                    kind=(request.form.get("kind") or "break")[:20],
                    recorded_by_id=current_user.id,
                ))
            elif action == "record_actual":
                duty_day = date.fromisoformat(request.form["day"])
                person_id = int(request.form.get("person_id") or 0)
                Staff.query.filter_by(id=person_id, is_operational=True).first_or_404()
                actual_start = datetime.fromisoformat(request.form["actual_start"])
                actual_end = datetime.fromisoformat(request.form["actual_end"])
                if actual_end <= actual_start:
                    raise ValueError("Actual duty end must be after its start.")
                assignment = Assignment.query.filter_by(
                    staff_id=person_id, day=duty_day
                ).first()
                row = AchievedDuty.query.filter_by(
                    person_id=person_id, day=duty_day
                ).first()
                if not row:
                    row = AchievedDuty(
                        unit_id=_current_unit_id(), person_id=person_id,
                        day=duty_day, recorded_by_id=current_user.id,
                    )
                    db.session.add(row)
                row.planned_assignment_id = assignment.id if assignment else None
                row.actual_start = actual_start
                row.actual_end = actual_end
                row.duty_type = (request.form.get("duty_type") or "operational")[:30]
                row.variance_reason = (
                    request.form.get("variance_reason") or ""
                ).strip()[:500]
            elif action == "review_fatigue":
                report = FatigueReport.query.filter_by(
                    id=int(request.form.get("report_id") or 0)
                ).first_or_404()
                response = (request.form.get("manager_response") or "").strip()
                if len(response) < 10:
                    raise ValueError("Record the assessment and action taken.")
                report.manager_response = response[:1000]
                report.status = request.form.get("status") if request.form.get(
                    "status"
                ) in {"reviewed", "closed"} else "reviewed"
                report.reviewed_by_id = current_user.id
                report.reviewed_at = utcnow()
                report.closed_at = utcnow() if report.status == "closed" else None
            elif action == "create_rule_version":
                latest = db.session.query(
                    db.func.max(RosterRuleVersion.version)
                ).filter(
                    RosterRuleVersion.unit_id == _current_unit_id()
                ).scalar() or 0
                rules = request.form.get("rules_json") or "{}"
                parsed = json.loads(rules)
                if not isinstance(parsed, dict):
                    raise ValueError("Rules must be a JSON object.")
                db.session.add(RosterRuleVersion(
                    unit_id=_current_unit_id(), version=latest + 1,
                    name=(request.form.get("name") or f"Rule set {latest + 1}")[:120],
                    rules_json=json.dumps(parsed),
                    change_reference=(request.form.get("change_reference") or "")[:120],
                    consultation_summary=(
                        request.form.get("consultation_summary") or ""
                    )[:2000],
                ))
            elif action == "approve_rule_version":
                rule = RosterRuleVersion.query.filter_by(
                    id=int(request.form.get("rule_id") or 0), state="draft"
                ).first_or_404()
                if not rule.change_reference or len(rule.consultation_summary) < 20:
                    raise ValueError(
                        "Approval requires a change reference and consultation summary."
                    )
                RosterRuleVersion.query.filter_by(
                    unit_id=_current_unit_id(), state="approved"
                ).update(
                    {"state": "superseded"},
                    synchronize_session=False,
                )
                rule.state = "approved"
                rule.effective_from = date.fromisoformat(
                    request.form["effective_from"]
                )
                rule.approved_by_id = current_user.id
                rule.approved_at = utcnow()
            else:
                abort(400)
            db.session.commit()
            log_change(
                "OperationalAssurance", 0, action, None, "completed",
                context_day=date(year, month, 1),
            )
            flash("Operational assurance record saved.", "ok")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            db.session.rollback()
            flash(str(exc), "error")
        return redirect(url_for("operations_assurance", ym=ym))

    positions = OperationalPosition.query.filter_by(is_active=True).order_by(
        OperationalPosition.code
    ).all()
    staff = Staff.query.filter_by(is_operational=True).order_by(Staff.name).all()
    endorsements = PositionEndorsement.query.order_by(
        PositionEndorsement.valid_until
    ).all()
    breaks = BreakPlan.query.filter(
        BreakPlan.day >= date(year, month, 1),
        BreakPlan.day < date(*_month_add(year, month, 1), 1),
    ).order_by(BreakPlan.day, BreakPlan.start_time).all()
    actuals = AchievedDuty.query.filter(
        AchievedDuty.day >= date(year, month, 1),
        AchievedDuty.day < date(*_month_add(year, month, 1), 1),
    ).order_by(AchievedDuty.day.desc()).all()
    reports = FatigueReport.query.filter(
        FatigueReport.status.in_(("open", "reviewed"))
    ).order_by(FatigueReport.reported_at.desc()).all()
    rules = RosterRuleVersion.query.order_by(
        RosterRuleVersion.version.desc()
    ).all()
    assurance = _position_assurance(year, month)
    return render_template(
        "operations_assurance.html", ym=ym, year=year, month=month,
        positions=positions, staff=staff, endorsements=endorsements,
        breaks=breaks, actuals=actuals, reports=reports, rules=rules,
        assurance=assurance,
        staff_by_id={row.id: row for row in staff},
        positions_by_id={row.id: row for row in positions},
        shifts=ShiftType.query.filter_by(is_active=True).order_by(ShiftType.code).all(),
    )


@app.route("/fatigue/report", methods=["GET", "POST"])
@login_required
def fatigue_self_report():
    if getattr(current_user, "role", "") == "superadmin":
        abort(403)
    if request.method == "POST":
        _validate_csrf()
        try:
            duty_day = date.fromisoformat(request.form["duty_day"])
            severity = request.form.get("severity") or ""
            summary = (request.form.get("summary") or "").strip()
            if severity not in {"low", "medium", "high", "unfit"}:
                raise ValueError("Select a fatigue severity.")
            if len(summary) < 10:
                raise ValueError("Provide a short description of the fatigue concern.")
            db.session.add(FatigueReport(
                unit_id=_current_unit_id(), person_id=current_user.id,
                duty_day=duty_day, severity=severity, summary=summary[:500],
            ))
            db.session.commit()
            flash(
                "Fatigue report submitted. If you may be unfit for duty, "
                "follow the unit's immediate reporting procedure as well.",
                "ok",
            )
            return redirect(url_for("fatigue_self_report"))
        except (ValueError, KeyError) as exc:
            db.session.rollback()
            flash(str(exc), "error")
    reports = FatigueReport.query.filter_by(
        person_id=current_user.id
    ).order_by(FatigueReport.reported_at.desc()).all()
    return render_template("fatigue_report.html", reports=reports, today=date.today())


@app.route("/planning/coverage/<ym>")
@login_required
def coverage_heatmap(ym):
    if not can_edit_roster(current_user):
        abort(403)
    year, month = parse_ym(ym)
    start, days = month_range(year, month)
    end = days[-1]
    counts = defaultdict(Counter)
    competence_exclusions = defaultdict(Counter)
    assignments = Assignment.query.filter(
        Assignment.unit_id == _current_unit_id(),
        Assignment.day >= start, Assignment.day <= end,
    ).all()
    for assignment in assignments:
        shift = ShiftType.query.filter_by(
            unit_id=_current_unit_id(), code=assignment.code
        ).first()
        prefix = (assignment.code or "")[:1]
        if (
            shift
            and shift.required_qualification
            and not _staff_has_shift_qualification(
                assignment.staff, shift, assignment.day
            )
        ):
            competence_exclusions[assignment.day][prefix] += 1
            continue
        counts[assignment.day][prefix] += 1
    return render_template(
        "coverage_heatmap.html", days=days, counts=counts, ym=ym,
        competence_exclusions=competence_exclusions,
    )


@app.route("/planning/scenarios", methods=["GET", "POST"])
@login_required
def scenarios_page():
    if not can_edit_roster(current_user):
        abort(403)
    unit_id = _current_unit_id()
    if request.method == "POST":
        _validate_csrf()
        changes = request.form.get("changes_json") or "[]"
        try:
            parsed = json.loads(changes)
            if not isinstance(parsed, list):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            abort(400, "Scenario changes must be a JSON list")
        evaluated = []
        for change in parsed:
            if not isinstance(change, dict):
                abort(400, "Each scenario change must be an object.")
            item = dict(change)
            reasons = []
            try:
                person_id = int(item.get("staff_id"))
                duty_date = date.fromisoformat(str(item.get("day") or ""))
            except (TypeError, ValueError):
                abort(400, "Scenario staff and dates must be valid.")
            person = Staff.query.filter_by(
                id=person_id, unit_id=unit_id, is_operational=True
            ).first()
            shift = ShiftType.query.filter_by(
                unit_id=unit_id,
                code=str(item.get("code") or "").upper(),
                is_active=True,
            ).first()
            if not person:
                reasons.append("Person is unavailable in this airport.")
            if not shift:
                reasons.append("Shift is unavailable in this airport.")
            if person and shift and not _staff_has_shift_qualification(
                person, shift, duty_date
            ):
                reasons.append("Required qualification is not valid.")
            item["eligibility"] = {
                "eligible": not reasons, "reasons": reasons,
            }
            evaluated.append(item)
        scenario = Scenario(
            unit_id=unit_id,
            name=(request.form.get("name") or "Untitled scenario")[:120],
            changes_json=json.dumps(evaluated),
            created_by_id=current_user.id,
        )
        db.session.add(scenario)
        db.session.commit()
        flash("Scenario saved without changing the live roster.", "ok")
        return redirect(url_for("scenarios_page"))
    rows = Scenario.query.filter_by(unit_id=unit_id).order_by(Scenario.id.desc()).all()
    return render_template("scenarios.html", scenarios=rows)

# -------------------- Manual TOIL entry page (no bulk seed in UI) --------------------


@app.route("/admin/toil/new")
@login_required
@admin_required
def admin_toil_new():
    atcos = Staff.query.filter_by(
        is_operational=True).order_by(Staff.name.asc()).all()
    if request.method == "POST":
        sid = int(request.form["staff_id"])
        amount = float(request.form.get("amount", "0") or 0)
        unit = request.form.get("unit", "days").lower()
        note = (request.form.get("note") or "").strip()
        s = Staff.query.filter_by(
            id=sid, unit_id=_current_unit_id()
        ).first_or_404()
        # Convert to half-days
        if unit.startswith("day"):
            half = int(round(amount * 2))
        else:  # hours
            half = int(round((amount / 8.0) * 2))
        s.toil_half_days = int((s.toil_half_days or 0) + half)
        db.session.commit()
        flash("TOIL balance updated.", "ok")
        return redirect(url_for("admin_toil_new"))
    return render_template("admin_toil_new.html", atcos=atcos)

# -------------------- Reports hub --------------------


@app.route("/reports", methods=["GET", "POST"])
@login_required
def reports_index():
    # Admin: show the hub
    if is_admin_user(current_user):
        today = date.today()
        month_title = datetime(today.year, today.month, 1).strftime("%B %Y")
        links = {
            "leave_year": url_for("report_leave_year"),
            "sickness": url_for("report_sickness"),
            "roster": url_for("roster_month", ym=f"{today.year}-{today.month:02d}"),
            "metrics": url_for("metrics"),
        }
        months = []  # hide month selector
        return render_template(
            "reports_index.html",
            ym=f"{today.year}-{today.month:02d}",
            year=today.year,
            month=today.month,
            month_title=month_title,
            months=months,
            links=links,
            page_title="OT/Swap/Ext Totals",
        )

    # Editor: only OT/SWAP totals
    if getattr(current_user, "role", "") in ("editor", "admin"):
        return redirect(url_for("metrics"))

    # Everyone else: no access
    abort(403)


_login_attempts: dict[str, deque[datetime]] = defaultdict(deque)
LOGIN_RATE_WINDOW = timedelta(minutes=15)
LOGIN_RATE_LIMIT = 10


def _login_rate_key(username: str) -> str:
    remote = request.remote_addr or "unknown"
    digest = hashlib.sha256(username.lower().encode()).hexdigest()[:16]
    return f"{remote}:{digest}"


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


@app.route("/login", methods=["GET", "POST"], endpoint="login")
def signin_form():   # function name can be anything; endpoint is 'login'
    if request.method == "POST":
        username = _normalized_login(
            request.form.get("username") or ""
        )
        password = (request.form.get("password") or "").strip()
        rate_key = _login_rate_key(username)
        attempts = _login_attempts[rate_key]
        cutoff = utcnow() - LOGIN_RATE_WINDOW
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        if len(attempts) >= LOGIN_RATE_LIMIT:
            _security_event("login_rate_limited", principal=rate_key[-16:])
            abort(429, "Too many login attempts. Try again later.")
        identity = PlatformIdentity.query.filter_by(username=username).first()
        user = None
        platform_login = False
        credentials_valid = False
        if identity:
            credentials_valid = check_password_hash(
                identity.password_hash, password
            )
            if credentials_valid:
                membership = UnitMembership.query.filter_by(
                    identity_id=identity.id, status="active"
                ).first()
                if membership and membership.person_id:
                    routing = db.session.get(
                        DatabaseRoutingMetadata, membership.unit_id
                    )
                    if DEPLOYMENT_ENV == "production" and not routing:
                        _security_event(
                            "operational_route_missing",
                            unit_id=membership.unit_id,
                        )
                        abort(503, "Operational database routing is unavailable.")
                    g.tenant_context_token = bind_authenticated_unit(
                        membership.unit_id,
                        routing.secret_name if routing else None,
                    )
                    user = db.session.get(Staff, membership.person_id)
                else:
                    user = identity
                    platform_login = identity.public_id.startswith(
                        "platform-"
                    )
        else:
            user = Staff.query.filter_by(username=username).first()
            credentials_valid = bool(
                user and user.check_password(password)
            )
        if user and credentials_valid:
            if user.membership_status != "active":
                flash("This account is not active.", "error")
                return render_template("login.html"), 403
            login_unit = db.session.get(Unit, user.unit_id)
            if (
                user.role != "superadmin"
                and (not login_unit or login_unit.status != "active")
            ):
                _security_event(
                    "suspended_unit_login_blocked",
                    principal=rate_key[-16:],
                    unit_id=user.unit_id,
                )
                flash("This airport account is not active.", "error")
                return render_template("login.html"), 403
            session.clear()
            if platform_login:
                credential = PlatformMfaCredential.query.filter_by(
                    identity_id=identity.id, enabled=True,
                    reset_required=False,
                ).first()
                session["_platform_mfa_identity_id"] = identity.id
                session["_platform_mfa_user_id"] = user.id
                session["_platform_mfa_rate_key"] = rate_key
                session["_platform_mfa_next"] = (
                    request.args.get("next")
                    if _is_safe_local_redirect(request.args.get("next")) else ""
                )
                _central_security_event(
                    "platform_password_verified", "challenge",
                    identity.id, rate_key[-16:],
                )
                db.session.commit()
                return redirect(url_for(
                    "platform_mfa_challenge"
                    if credential else "platform_mfa_setup"
                ))
            credential = MfaCredential.query.filter_by(
                person_id=user.id, enabled=True
            ).first()
            if credential:
                session["_mfa_user_id"] = user.id
                session["_mfa_unit_id"] = user.unit_id
                session["_mfa_rate_key"] = rate_key
                session["_mfa_next"] = (
                    request.args.get("next")
                    if _is_safe_local_redirect(request.args.get("next")) else ""
                )
                return redirect(url_for("mfa_challenge"))
            login_user(user)
            session.permanent = True
            session["_session_started_at"] = utcnow().isoformat()
            _login_attempts.pop(rate_key, None)
            _security_event(
                "login_succeeded",
                principal=rate_key[-16:],
                unit_id=user.unit_id,
            )
            _record_successful_login(user)
            flash("Logged in successfully", "ok")
            # support ?next=... to return where user was going
            nxt = request.args.get("next")
            return redirect(nxt if _is_safe_local_redirect(nxt) else url_for("index"))
        attempts.append(utcnow())
        if identity:
            _central_security_event(
                "platform_login_failed", "denied", identity.id,
                rate_key[-16:],
            )
            db.session.commit()
        _security_event("login_failed", principal=rate_key[-16:])
        flash("Invalid username or password.", "error")
    return render_template("login.html")


def _decrypt_mfa_secret(credential) -> str:
    try:
        return _field_cipher().decrypt(
            credential.encrypted_secret.encode()
        ).decode()
    except (InvalidToken, ValueError) as exc:
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


_platform_mfa_attempts: dict[int, deque[datetime]] = defaultdict(deque)


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
    rate_key = session.get("_platform_mfa_rate_key", "")
    session.clear()
    login_user(user)
    session.permanent = True
    session["_session_started_at"] = utcnow().isoformat()
    session["_platform_mfa_verified"] = True
    identity.last_active_at = utcnow()
    _central_security_event(
        "platform_recovery_code_used" if recovery_used
        else "platform_mfa_verified",
        "success", identity.id,
        hashlib.sha256(identity.username.lower().encode()).hexdigest()[:16],
    )
    db.session.commit()
    if rate_key:
        _login_attempts.pop(rate_key, None)
    _platform_mfa_attempts.pop(identity.id, None)
    return redirect(next_url or url_for("platform_admin"))


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
    if request.method == "POST":
        _validate_csrf()
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
        credential.encrypted_secret = _field_cipher().encrypt(
            pending.encode()
        ).decode()
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
        provisioning_uri=provisioning_uri, platform_enrolment=True,
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
    attempts = _platform_mfa_attempts[identity.id]
    cutoff = utcnow() - LOGIN_RATE_WINDOW
    while attempts and attempts[0] < cutoff:
        attempts.popleft()
    if len(attempts) >= LOGIN_RATE_LIMIT:
        _central_security_event(
            "platform_mfa_rate_limited", "denied", identity.id
        )
        db.session.commit()
        abort(429, "Too many verification attempts. Try again later.")
    if request.method == "POST":
        _validate_csrf()
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
        attempts.append(utcnow())
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
            next_url = session.pop("_mfa_next", "")
            session.pop("_mfa_user_id", None)
            session.pop("_mfa_unit_id", None)
            login_user(user)
            session.permanent = True
            session["_session_started_at"] = utcnow().isoformat()
            rate_key = session.pop("_mfa_rate_key", "")
            if rate_key:
                _login_attempts.pop(rate_key, None)
            _security_event(
                "mfa_login_succeeded",
                principal=hashlib.sha256(
                    user.username.lower().encode()
                ).hexdigest()[:16],
                unit_id=user.unit_id,
            )
            _record_successful_login(user)
            db.session.commit()
            return redirect(next_url or url_for("index"))
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
    if request.method == "POST":
        _validate_csrf()
        code = re.sub(r"\s", "", request.form.get("code") or "")
        if not pyotp.TOTP(pending).verify(code, valid_window=1):
            flash("The verification code is not valid.", "error")
            return redirect(url_for("mfa_setup"))
        recovery_codes = [
            secrets.token_hex(5).upper() for _ in range(10)
        ]
        if not credential:
            credential = MfaCredential(
                unit_id=_current_unit_id(), person_id=current_user.id,
                encrypted_secret="",
            )
            db.session.add(credential)
        credential.encrypted_secret = _field_cipher().encrypt(
            pending.encode()
        ).decode()
        credential.enabled = True
        credential.enrolled_at = utcnow()
        credential.recovery_codes_digest = json.dumps([
            hashlib.sha256(code.encode()).hexdigest()
            for code in recovery_codes
        ])
        db.session.commit()
        session.pop("_pending_mfa_secret", None)
        return render_template(
            "mfa_setup.html", enabled=True, recovery_codes=recovery_codes
        )
    return render_template(
        "mfa_setup.html", enabled=False, secret=pending,
        provisioning_uri=provisioning_uri,
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
def reconcile_signups(apply_changes):
    """Report or safely reconcile interrupted cross-database signups."""
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
                    if staff and staff.membership_status != "active":
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


# -------------------- DB init (single, safe block) --------------------

with app.app_context():
    if (
        DEPLOYMENT_ENV != "production"
        and os.environ.get("ATCROSTER_SKIP_RUNTIME_SCHEMA") != "1"
    ):
        db.create_all()
        is_sqlite = db.engine.dialect.name == "sqlite"
        if is_sqlite:
            # Legacy desktop compatibility only. Production uses Alembic.
            migrate_tenant_foundation_compat()
            migrate_add_perf_indexes()
            migrate_add_met_and_assessor()
            migrate_add_toil_half_days_and_convert()
            migrate_add_ut_flags()
            migrate_add_assignment_annotation()
            migrate_add_unique_assignment_key()
            migrate_add_requirement_req_d()
            migrate_add_is_training()
            migrate_add_wm_dwm_exclude()
            migrate_add_phone_number()
            migrate_add_role_and_calendar_token()

            cols = [row[1] for row in db.session.execute(
                text("PRAGMA table_info(shift_request)"))]

            def _add_col(name, ddl):
                if name not in cols:
                    try:
                        db.session.execute(
                            text(f"ALTER TABLE shift_request ADD COLUMN {ddl}"))
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
            _add_col("admin_response", "admin_response TEXT DEFAULT ''")
            _add_col("responded_by_id", "responded_by_id INTEGER")
            _add_col("responded_at", "responded_at TEXT")
            _add_col("status", "status VARCHAR(20) DEFAULT 'pending'")

        seed_once()
        refresh_shift_cache()

# Expose helpers & models needed by Jinja templates that refer to them directly
app.jinja_env.globals['month_range'] = month_range
app.jinja_env.globals['ShiftType'] = ShiftType

# -------------------- Run --------------------


# -------------------- WSGI entry point --------------------
# PythonAnywhere’s WSGI file imports "application"
application = app

# -------------------- Local dev server --------------------
if __name__ == "__main__":
    # bind explicitly & avoid debug reloader port conflicts
    app.run(host="127.0.0.1", port=5001, debug=False)
