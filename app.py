import flask as _flask
from functools import wraps
from calendar import monthrange
from collections import defaultdict, Counter, deque
from typing import Optional, Tuple
import base64
from urllib import parse as urllib_parse, request as urllib_request, error as urllib_error
from flask import Flask, render_template, request, redirect, url_for, flash, Response, abort, session, g
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

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text, or_

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

# Make external URLs prefer https behind PA’s proxy
app.config["PREFERRED_URL_SCHEME"] = "https"
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Jinja helper
app.jinja_env.globals['now'] = lambda: datetime.now()


def utcnow():
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)

# Database & login
db = SQLAlchemy(app, session_options={"expire_on_commit": False})
login_manager = LoginManager(app)
login_manager.login_view = "login"


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


def _invalidate_month_cache_for_day(d: date, unit_id: int | None = None):
    if _cache and d:
        try:
            unit_ids = []
            if unit_id is not None:
                unit_ids.append(unit_id)
            else:
                unit_ids.extend([u.id for u in Unit.query.all()])
                unit_ids.append(None)
            for uid in unit_ids:
                _cache.delete_memoized(_load_month_roster_fast, d.year, d.month, uid)
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


# -------------------- Constants --------------------
MIN_MONTH = date(2025, 4, 1)   # Start app from April 2025

# Annotations (includes TOA8/TOAI accrual; keep legacy TOAU accepted)
ANNOT_RE = re.compile(
    r"^(EXTS|EXTL|SWAP|SOAL|A2|A4|A6|A8|TOA8|TOAI|TOAU)([MDAN])?$", re.I)

# Working codes (treat SC/SSC as working training shifts)
WORKING_CODES = {"M", "D", "A", "N", "SC", "SSC", "SBY"}

# Codes considered leave-like in generic logic (but reports only count AL, see below)
LEAVE_CODES = {"AL", "PL", "SPL"}

# Codes that must NOT be set via the roster grid (per user requirement)
BANNED_ROSTER_CODES = {"SIC", "SC", "SSC",
                       "AL", "SP", "SPL", "PL", "TOU8", "TOUI"}

# Codes that must be excluded from day totals (counters)
EXCLUDE_FROM_COUNTERS = {"OSS"}

# -------------------- Models --------------------


class Unit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)


class Watch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32), nullable=False)
    order_index = db.Column(db.Integer, nullable=False, default=0)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=True)
    unit = db.relationship("Unit", backref="watches")

    __table_args__ = (
        db.UniqueConstraint("name", "unit_id", name="uniq_watch_unit_name"),
    )


class Staff(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    def set_password(self, password: str) -> None:
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        from werkzeug.security import check_password_hash
        # be robust if password_hash is None/empty
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)

    # Auth
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    # Roles: 'superadmin' | 'admin' | 'editor' | 'user'
    role = db.Column(db.String(10), nullable=False, default="user")

    phone_number = db.Column(db.String(30), default="")

    @property
    def is_admin_role(self) -> bool:
        return (self.role or "user") in {"admin", "superadmin"}

    @property
    def is_editor_role(self) -> bool:
        return (self.role or "user") in {"editor", "admin", "superadmin"}

    @property
    def is_super_admin_role(self) -> bool:
        return (self.role or "user") == "superadmin"

    # Back-compat (kept but unused in logic)
    is_admin = db.Column(db.Boolean, default=False)

    # Public ICS token for calendar subscription
    calendar_token = db.Column(db.String(64), unique=True, nullable=True)

    # Identity / roster fields
    name = db.Column(db.String(80), nullable=False)
    staff_no = db.Column(db.String(20), unique=True, nullable=False)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=True)
    unit = db.relationship("Unit", backref="staff_members")

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
    code = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(40), nullable=False, default="")
    start_time = db.Column(db.Time, nullable=True)
    end_time = db.Column(db.Time, nullable=True)
    is_working = db.Column(db.Boolean, default=True)
    # training flag (counts to fatigue but excluded from daily M/D/A/N counters)
    is_training = db.Column(db.Boolean, default=False)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=True)
    unit = db.relationship("Unit", backref="shift_types")

    __table_args__ = (
        db.UniqueConstraint("code", "unit_id", name="uniq_shift_code_unit"),
    )


class Requirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    req_m = db.Column(db.Integer, default=0)
    req_d = db.Column(db.Integer, default=0)
    req_a = db.Column(db.Integer, default=0)
    req_n = db.Column(db.Integer, default=0)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=True)
    unit = db.relationship("Unit", backref="requirements")
    __table_args__ = (db.UniqueConstraint(
        "year", "month", "unit_id", name="uniq_year_month_unit"),)


class Leave(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
    staff = db.relationship("Staff", backref="leaves")
    leave_type = db.Column(db.String(10), nullable=False)  # AL/PL/SPL only
    start = db.Column(db.Date, nullable=False)
    end = db.Column(db.Date, nullable=False)


class Sickness(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
    start = db.Column(db.Date, nullable=False)
    end = db.Column(db.Date, nullable=False)
    code = db.Column(db.String(10), nullable=False, default="SC")
    staff = db.relationship("Staff", backref="sickness_periods")


class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), index=True)
    staff = db.relationship("Staff", backref="assignments")
    day = db.Column(db.Date, index=True)
    code = db.Column(db.String(10), nullable=False)
    source = db.Column(db.String(10), default="auto")
    note = db.Column(db.String(140), default="")
    # EXTS/EXTL/SWAP/A2/A4/A6/A8/SOAL/TOA8/TOAI (+ optional suffix like A6M)
    annotation = db.Column(db.String(20), default="")
    __table_args__ = (db.UniqueConstraint(
        "staff_id", "day", name="uniq_staff_day"),)


class ShiftRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey(
        "staff.id"), index=True, nullable=False)
    staff = db.relationship("Staff", backref="shift_requests")
    day = db.Column(db.Date, index=True, nullable=False)
    code = db.Column(db.String(10), nullable=False)
    submitted_at = db.Column(db.DateTime, default=utcnow)
    __table_args__ = (db.UniqueConstraint("staff_id", "day",
                      name="uniq_shift_request_staff_day"),)
    # >>> NEW admin response fields
    admin_response = db.Column(db.Text, default="")
    responded_by_id = db.Column(db.Integer)  # FK optional (kept simple)
    responded_at = db.Column(db.DateTime)
    # pending/approved/rejected/closed
    status = db.Column(db.String(20), default="pending")


class AiRuleSet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    rules_json = db.Column(db.Text, nullable=False, default="{}")
    __table_args__ = (db.UniqueConstraint(
        "year", "month", name="uniq_ai_ruleset_month"),)


class ChangeLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
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
    staff_id = db.Column(db.Integer, db.ForeignKey(
        "staff.id"), nullable=False, index=True)
    watch_id = db.Column(db.Integer, db.ForeignKey("watch.id"), nullable=False)
    effective_date = db.Column(db.Date, nullable=False, index=True)
    staff = db.relationship("Staff", backref="watch_history")
    watch = db.relationship("Watch")

# Cached shift lookup (define after models so ShiftType exists when called)


@lru_cache(maxsize=512)
def _shift_by_code(code: str, unit_id: int | None = None):
    rows = (ShiftType.query
            .filter(ShiftType.code == code)
            .order_by(ShiftType.unit_id.desc())
            .all())
    if not rows:
        return None
    if unit_id is not None:
        for row in rows:
            if row.unit_id == unit_id:
                return row
        for row in rows:
            if row.unit_id is None:
                return row
        # No unit-specific or global definition – treat as unknown in this unit.
        return None
    for row in rows:
        if row.unit_id is None:
            return row
    return rows[0]


def refresh_shift_cache():
    _shift_by_code.cache_clear()

# -------------------- Login --------------------


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Staff, int(user_id))

# --------- Fast month loader & cache (uses functions defined later but safe) ----------


def _load_month_roster_core(y: int, m: int, unit_id: int | None = None):
    """
    Returns (days, staff_list, a_map, req) and NEVER returns None.
    On failure: returns ([], [], {}, ensure_month_requirement(y,m)).
    """
    try:
        unit_filter = unit_id
        start = date(y, m, 1)
        days_in_m = monthrange(y, m)[1]
        days = [start + timedelta(days=i) for i in range(days_in_m)]
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = date(ny, nm, 1)

        # Staff ordering
        try:
            staff_query = (Staff.query
                           .outerjoin(Watch, Staff.watch_id == Watch.id))
            if unit_filter is not None:
                staff_query = staff_query.filter(Staff.unit_id == unit_filter)
            staff = (staff_query
                     .order_by(Watch.order_index, Staff.name)
                     .all())
        except Exception:
            q = Staff.query
            if unit_filter is not None:
                q = q.filter(Staff.unit_id == unit_filter)
            staff = q.order_by(Staff.id).all()

        # Assignments for the month (narrow columns)
        rows = (db.session.query(
            Assignment.staff_id,
            Assignment.day,
            Assignment.code,
            Assignment.source,
            Assignment.annotation
        )
            .join(Staff, Staff.id == Assignment.staff_id)
            .filter(Assignment.day >= start, Assignment.day < end))
        if unit_filter is not None:
            rows = rows.filter(Staff.unit_id == unit_filter)
        rows = rows.all()

        a_map = {}
        for sid, d, code, source, ann in rows:
            a_map.setdefault(sid, {})[d] = (code, source, ann)

        req = Requirement.query.filter_by(year=y, month=m, unit_id=unit_filter).first()
        if not req:
            req = ensure_month_requirement(y, m, unit_id=unit_filter)

        return days, staff, a_map, req

    except Exception as e:
        try:
            app.logger.exception(
                "Failed _load_month_roster_core(%s,%s): %s", y, m, e)
        except Exception:
            pass
        # Ensure we still return a valid 4-tuple
        return ([], [], {}, ensure_month_requirement(y, m, unit_id=unit_filter))


# IMPORTANT: overwrite any previously memoized wrapper
_load_month_roster_fast = _memoize(seconds=300)(_load_month_roster_core)


# -------------------- Helpers --------------------
# === Unified permissions (admins, editors, WM, DWM) ===


def is_admin_user(u) -> bool:
    role = getattr(u, "role", "")
    return role in ("admin", "superadmin") or bool(getattr(u, "is_admin", False))


def is_editor_user(u) -> bool:
    # admin counts as editor
    return getattr(u, "role", "") in ("editor", "admin", "superadmin")


def is_super_admin_user(u) -> bool:
    return getattr(u, "role", "") == "superadmin"


def can_edit_roster(u) -> bool:
    return (
        is_admin_user(u)
        or is_editor_user(u)
        or bool(getattr(u, "is_wm", False))
        or bool(getattr(u, "is_dwm", False))
    )


def roster_edit_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not can_edit_roster(current_user):
            return ("Forbidden", 403)
        return f(*args, **kwargs)
    return wrapper


def active_unit_id() -> int | None:
    return getattr(g, "active_unit_id", None)


def active_unit() -> Unit | None:
    uid = active_unit_id()
    if uid is None:
        return None
    for u in getattr(g, "available_units", []) or []:
        if u and u.id == uid:
            return u
    return Unit.query.get(uid)


@app.before_request
def determine_active_unit():
    g.active_unit_id = None
    g.available_units = []
    if not current_user.is_authenticated:
        session.pop("active_unit_id", None)
        return

    if is_super_admin_user(current_user):
        units = Unit.query.order_by(Unit.name).all()
        g.available_units = units
        requested = request.args.get("unit_id")
        if requested is not None:
            if requested == "all":
                session["active_unit_id"] = None
            else:
                try:
                    session["active_unit_id"] = int(requested)
                except (TypeError, ValueError):
                    pass
        active_id = session.get("active_unit_id")
        if isinstance(active_id, str) and active_id.isdigit():
            active_id = int(active_id)
        if active_id is not None and not any(u.id == active_id for u in units):
            active_id = None
        if active_id is None and units:
            active_id = units[0].id
            session["active_unit_id"] = active_id
        g.active_unit_id = active_id
    else:
        uid = current_user.unit_id
        if uid is not None:
            g.active_unit_id = uid
            if current_user.unit:
                g.available_units = [current_user.unit]
            else:
                unit = Unit.query.filter_by(id=uid).first()
                g.available_units = [unit] if unit else []
        else:
            g.active_unit_id = None
            g.available_units = []


@app.context_processor
def inject_perms():
    au = current_user if getattr(
        current_user, "is_authenticated", False) else None
    return {
        "is_admin": bool(au) and is_admin_user(au),
        "is_editor": bool(au) and is_editor_user(au),
        "is_super_admin": bool(au) and is_super_admin_user(au),
        "active_unit": active_unit(),
        "available_units": getattr(g, "available_units", []),
    }


def month_has_data(year: int, month: int, unit_id: int | None = None) -> bool:
    """Fast check: do we already have any assignments for this month?"""
    start = date(year, month, 1)
    ny, nm = _month_add(year, month, 1)
    end = date(ny, nm, 1)  # exclusive
    q = (db.session.query(Assignment.id)
         .join(Staff, Staff.id == Assignment.staff_id)
         .filter(Assignment.day >= start, Assignment.day < end))
    if unit_id is not None:
        q = q.filter(Staff.unit_id == unit_id)
    return q.limit(1).first() is not None


def month_range(year: int, month: int):
    start = date(year, month, 1)
    stop = date(year + (month // 12), (month % 12) + 1, 1)
    days = (stop - start).days
    return start, [start + timedelta(d) for d in range(days)]


@lru_cache(maxsize=4096)
def watch_id_for_staff_on(staff_id: int, on_date: date) -> int | None:
    """Return the watch_id that applies to this staff on a given date
    using StaffWatchHistory; fall back to Staff.watch_id if no history."""
    hist = (StaffWatchHistory.query
            .filter(StaffWatchHistory.staff_id == staff_id,
                    StaffWatchHistory.effective_date <= on_date)
            .order_by(StaffWatchHistory.effective_date.desc())
            .first())
    if hist:
        return hist.watch_id
    s = db.session.get(Staff, staff_id)
    return s.watch_id if s else None


def parse_ym(ym: str):
    y, m = ym.split("-")
    return int(y), int(m)


def get_shift(code: str):
    # hot path → use cached lookup
    normalized = (code or "").upper()
    uid = active_unit_id()
    return _shift_by_code(normalized, uid)


@lru_cache(maxsize=64)
def _shift_groups_snapshot(unit_id: int | None):
    query = ShiftType.query.order_by(ShiftType.code)
    if unit_id is not None:
        query = query.filter(or_(ShiftType.unit_id == unit_id, ShiftType.unit_id.is_(None)))
    all_shifts = query.all()
    allowed = [sh for sh in all_shifts if sh.code not in BANNED_ROSTER_CODES]
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


def ensure_month_requirement(year, month, default=(4, 4, 4, 2), unit_id: int | None = None):
    r = Requirement.query.filter_by(year=year, month=month, unit_id=unit_id).first()
    if not r:
        if len(default) == 3:
            dm, da, dn = default[0], default[1], default[2]
            dd = 0
        else:
            dm, dd, da, dn = default
        r = Requirement(year=year, month=month, unit_id=unit_id,
                        req_m=dm, req_d=dd, req_a=da, req_n=dn)
        db.session.add(r)
        db.session.commit()
    return r

# Idempotent month generation that preserves manual entries


def generate_month(year: int, month: int, unit_id: int | None = None, *args, **kwargs):
    """Ensure rows exist/are correct for the month without touching manual edits."""
    _, days = month_range(year, month)
    if unit_id is None and "unit_id" in kwargs:
        unit_id = kwargs.get("unit_id")
    staff_query = Staff.query.order_by(Staff.id)
    if unit_id is not None:
        staff_query = staff_query.filter(Staff.unit_id == unit_id)
    for s in staff_query:
        for d in days:
            refresh_day_from_pattern_and_leave(s, d)
    db.session.commit()


def generate_month_roster(year: int, month: int, who_user: "User", unit_id: int | None = None):
    # Guard: lock
    if is_month_locked(year, month):
        raise RuntimeError(
            f"Month {year:04d}-{month:02d} is locked (lock date {lock_date_for_month(year, month).isoformat()})."
        )

    if unit_id is None:
        unit_id = getattr(who_user, "unit_id", None)

    rules = _load_ai_rules(year, month)
    night_code = rules["night_code"]
    day_code_mon_sat = rules["day_code_mon_sat"]
    day_code_sun = rules["day_code_sun"]
    no_nights_ids = rules["no_nights_ids"]

    # Month range
    start, days = month_range(year, month)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)

    # Staff in display/fairness order
    staff_query = (
        Staff.query
        .outerjoin(Watch, Staff.watch_id == Watch.id)
        .filter(Staff.is_operational == True)
    )
    if unit_id is not None:
        staff_query = staff_query.filter(Staff.unit_id == unit_id)
    staff = staff_query.order_by(Watch.order_index, Staff.name).all()

    req = Requirement.query.filter_by(year=year, month=month, unit_id=unit_id).first()
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
        for s in staff:
            if code == night_code and s.id in no_nights_ids:
                continue
            if _has_leave_or_sick(s.id, d):
                continue
            a = by_staff_day[s.id].get(d) or Assignment(staff_id=s.id, day=d)
            if _cell_is_protected(a):
                continue
            if not _fatigue_ok(s, d, code):
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
        sh = _shift_by_code(c)
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


def generate_range(start_day: date, end_day: date, unit_id: int | None = None):
    """
    Ensure requirements and (re)build each month from start_day's month through
    end_day's month (inclusive). Safe to re-run; respects manual/protected codes.
    """
    unit_filter = unit_id if unit_id is not None else active_unit_id()
    for y, m in _year_month_iter(start_day, end_day):
        ensure_month_requirement(y, m, unit_id=unit_filter)
        generate_month(y, m, unit_id=unit_filter)


def ensure_assignments_for_range(start_day: date, end_day: date, unit_id: int | None = None):
    unit_filter = unit_id if unit_id is not None else active_unit_id()
    for y, m in _year_month_iter(start_day, end_day):
        ensure_month_requirement(y, m, unit_id=unit_filter)
        generate_month(y, m, unit_id=unit_filter)


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

# -------------------- Migrations / seeding --------------------


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
        if not u.role or u.role not in ("admin", "editor", "user"):
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


def migrate_add_unit_fields():
    from sqlalchemy import inspect, text
    insp = inspect(db.engine)
    try:
        tables = insp.get_table_names()
    except Exception:
        tables = []

    if "unit" not in tables:
        try:
            Unit.__table__.create(bind=db.engine, checkfirst=True)
        except Exception:
            pass

    def _add_column_if_missing(table: str, column: str, ddl: str):
        try:
            cols = {c["name"] for c in insp.get_columns(table)}
        except Exception:
            return
        if column not in cols:
            try:
                db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
                db.session.commit()
            except Exception:
                db.session.rollback()

    _add_column_if_missing("staff", "unit_id", "unit_id INTEGER")
    _add_column_if_missing("watch", "unit_id", "unit_id INTEGER")
    _add_column_if_missing("shift_type", "unit_id", "unit_id INTEGER")
    _add_column_if_missing("requirement", "unit_id", "unit_id INTEGER")

    default_unit = Unit.query.order_by(Unit.id).first()
    if default_unit:
        uid = default_unit.id
        try:
            db.session.execute(text("UPDATE staff SET unit_id = :uid WHERE unit_id IS NULL"), {"uid": uid})
            db.session.execute(text("UPDATE watch SET unit_id = :uid WHERE unit_id IS NULL"), {"uid": uid})
            db.session.execute(text("UPDATE requirement SET unit_id = :uid WHERE unit_id IS NULL"), {"uid": uid})
            db.session.commit()
        except Exception:
            db.session.rollback()


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


def ensure_unit(name: str) -> Unit:
    u = Unit.query.filter_by(name=name).first()
    if not u:
        u = Unit(name=name)
        db.session.add(u)
        db.session.commit()
    return u


def ensure_shift(code, name, start=None, end=None, is_working=False, is_training=False, unit: Unit | None = None):
    unit_id = unit.id if unit else None
    sh = ShiftType.query.filter_by(code=code, unit_id=unit_id).first()
    if not sh:
        sh = ShiftType(code=code, name=name, start_time=start, end_time=end,
                       is_working=is_working, is_training=is_training,
                       unit_id=unit_id)
        db.session.add(sh)
        db.session.commit()
    return sh


def ensure_watch(name: str, order_index: int, unit: Unit | None = None):
    w = Watch.query.filter_by(name=name, unit=unit).first()
    if not w:
        w = Watch(name=name, order_index=order_index, unit=unit)
        db.session.add(w)
        db.session.commit()
    return w


def seed_once():
    default_unit = ensure_unit("Leeds")
    ensure_unit("Prestwick")

    if Watch.query.count() > 0:
        # make sure TOU* & OSS exist if DB already seeded
        ensure_shift("TOUI", "TOIL (UI)", is_working=False)
        ensure_shift("TOU8", "TOIL (U8)", is_working=False)
        ensure_shift("OSS",  "Operational Support", is_working=False)
        ensure_super_admin_account(default_unit)
        return

    watches = []
    for idx, letter in enumerate(["A", "B", "C", "D", "E"], start=1):
        watches.append(Watch(name=f"Watch {letter}", order_index=idx, unit=default_unit))
    watches.append(Watch(name="Watch NOPS", order_index=6, unit=default_unit))
    db.session.add_all(watches)

    db.session.add_all([
        ShiftType(code="M",   name="Morning",     start_time=time(
            6, 0),  end_time=time(14, 0), is_working=True),
        ShiftType(code="D",   name="Day",         start_time=time(
            8, 0),  end_time=time(16, 0), is_working=True),
        ShiftType(code="A",   name="Afternoon",   start_time=time(
            14, 0), end_time=time(22, 0), is_working=True),
        ShiftType(code="N",   name="Night",       start_time=time(
            22, 0), end_time=time(6, 0),  is_working=True),
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
                unit=default_unit,
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

    ensure_super_admin_account(default_unit)


def ensure_super_admin_account(default_unit: Unit | None = None):
    unit = default_unit or Unit.query.order_by(Unit.id).first()
    admin_user = Staff.query.filter_by(username="801210").first()
    if not admin_user:
        watch = None
        if unit:
            watch = (Watch.query
                     .filter(or_(Watch.unit_id == unit.id, Watch.unit_id.is_(None)))
                     .order_by(Watch.order_index)
                     .first())
        admin_user = Staff(
            username="801210",
            name="Super Admin",
            staff_no="801210",
            role="superadmin",
            unit=unit,
            watch=watch,
            is_operational=False,
            leave_year_start_month=4,
            leave_entitlement_days=0,
            leave_public_holidays=0,
            leave_carryover_days=0,
        )
        admin_user.set_password("password")
        if not admin_user.calendar_token:
            admin_user.calendar_token = secrets.token_hex(16)
        db.session.add(admin_user)
    else:
        if admin_user.role != "superadmin":
            admin_user.role = "superadmin"
        if unit and admin_user.unit_id != unit.id:
            admin_user.unit_id = unit.id
        admin_user.set_password("password")
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
    m = ANNOT_RE.match(s.strip().upper())
    if not m:
        return None
    typ, suf = m.group(1).upper(), (m.group(2).upper() if m.group(2) else None)
    return {"type": typ, "suffix": suf}


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

    staff_obj = getattr(a, "staff", None)
    if staff_obj is None and a.staff_id:
        staff_obj = db.session.get(Staff, a.staff_id)

    # Invalidate month cache for this day
    _invalidate_month_cache_for_day(a.day, getattr(staff_obj, "unit_id", None))

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


# central place to mark things that should never count as "working"
_NON_WORKING_CODES = {
    "OFF", "AL", "PL", "SPL", "TOU8", "TOUI",
    "OSS", "OFFICE", "WFH", "CTB", "MTG"
}


def _is_non_working(code: str) -> bool:
    return _normalize_code(code) in _NON_WORKING_CODES


def _is_working_code_prefix(code: str, prefix: str) -> bool:
    """
    True if:
      - code is not in the non-working list
      - ShiftType says it's working (when known)
      - AND the normalized code startswith the given prefix
    Falls back to just the prefix check if ShiftType is unknown.
    """
    cu = _normalize_code(code)
    if not cu or cu in _NON_WORKING_CODES:
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
        cur = request.form.get("current_password", "")
        new1 = request.form.get("new_password", "")
        new2 = request.form.get("confirm_password", "")
        if not current_user.check_password(cur):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("password_change"))
        if not new1 or new1 != new2:
            flash("New passwords do not match.", "error")
            return redirect(url_for("password_change"))
        u = db.session.get(Staff, current_user.id)
        u.set_password(new1)
        db.session.commit()
        flash("Password updated.", "ok")
        return redirect(url_for("staff_profile", sid=current_user.id))
    return render_template("password.html")

# -------------------- Main / Roster --------------------


@app.route("/")
@login_required
def index():
    t = date.today()
    return redirect(url_for("roster_month", ym=f"{t.year}-{t.month:02d}"))


@app.route("/unit/switch", methods=["POST"])
@login_required
def switch_unit():
    if not is_super_admin_user(current_user):
        abort(403)
    requested = request.form.get("unit_id")
    next_url = request.form.get("next") or request.referrer or url_for("index")
    if requested in {"all", "global", ""}:
        session["active_unit_id"] = None
    else:
        try:
            session["active_unit_id"] = int(requested)
        except (TypeError, ValueError):
            pass
    return redirect(next_url)


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
    return {
        "is_admin":  bool(au) and is_admin_user(au),
        "is_editor": bool(au) and is_editor_user(au),
    }


@app.route("/roster/<ym>")
@login_required
def roster_month(ym):
    year, month = parse_ym(ym)
    unit_id = active_unit_id()

    # Build only if the month has no data yet
    if not month_has_data(year, month, unit_id=unit_id):
        ensure_month_requirement(year, month, unit_id=unit_id)
        generate_month(year, month, unit_id=unit_id)

    # Fast path: 2 queries total (staff + all assignments in month)
    days, staff, a_map_tuples, req = _load_month_roster_fast(year, month, unit_id)

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
    shifts_working, shifts_training, shifts_non = _shift_groups_snapshot(unit_id)
    training_codes = {sh.code for sh in shifts_training}

    # --- Effective watch for THIS month (first day of the month) ---
    def _watch_for(sid: int, on_date: date):
        fn = globals().get("watch_id_for_staff_on")
        if callable(fn):
            return fn(sid, on_date)
        s = db.session.get(Staff, sid)
        return s.watch_id if s else None

    display_watch_by_staff = {s.id: _watch_for(s.id, start) for s in staff}

    # Optional: ensure staff ordering matches watch order for the display month
    try:
        watch_query = Watch.query
        if unit_id is not None:
            watch_query = watch_query.filter(or_(Watch.unit_id == unit_id, Watch.unit_id.is_(None)))
        watch_order = {w.id: w.order_index for w in watch_query.all()}
    except Exception:
        watch_order = {}

    staff.sort(
        key=lambda s: (
            watch_order.get(display_watch_by_staff.get(s.id), 9999),
            _rank_within_watch(s),
            s.name
        )
    )

    # Counters (operational only); exclude training and EXCLUDE_FROM_COUNTERS
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
            if 'EXCLUDE_FROM_COUNTERS' in globals() and c in EXCLUDE_FROM_COUNTERS:
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
        ShiftRequest.day >= start, ShiftRequest.day < month_end
    ).all()
    req_pending_map = {
        (r.staff_id, r.day): r.code
        for r in reqs
        if (r.status or "pending").lower() != "closed"
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
    # parse inputs
    d = date.fromisoformat(day)
    st = db.session.get(Staff, staff_id) or Staff.query.get_or_404(staff_id)

    # fetch or create the assignment row for that staff/day
    a = Assignment.query.filter_by(staff_id=staff_id, day=d).first()
    if a is None:
        a = Assignment(staff=st, day=d, code="OFF")
        db.session.add(a)

    # form fields (each cell posts either code OR annotation)
    code = (request.form.get("code") or "").strip().upper()
    annot = request.form.get("annotation")  # None => no change

    # if a shift code was posted, validate and set it
    if code != "":
        if 'BANNED_ROSTER_CODES' in globals() and code in BANNED_ROSTER_CODES:
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
        old = a.annotation or ""
        newv = (annot or "").strip().upper()
        if old != newv:
            _apply_toil_annotation_delta(
                staff=st, old_annot=old, new_annot=newv)
            a.annotation = newv

    # clear any pending request now that the roster cell is written
    req = ShiftRequest.query.filter_by(staff_id=staff_id, day=d).first()
    if req:
        # Keep the request for auditing/history. Auto-close only if it had
        # never been actioned so that explicit admin decisions (approved /
        # rejected) remain visible to the requester.
        status = (req.status or "pending").lower()
        if status == "pending":
            req.status = "closed"
        if not req.responded_at:
            req.responded_at = utcnow()
        if not req.responded_by_id:
            req.responded_by_id = getattr(current_user, "id", None)

    db.session.commit()
    return redirect(url_for("roster_month", ym=ym))


@app.route("/roster/<ym>/export")
@login_required
def roster_export_csv(ym):
    year, month = parse_ym(ym)
    start, days = month_range(year, month)

    unit_id = active_unit_id()

    staff_query = (Staff.query
                   .outerjoin(Watch, Staff.watch_id == Watch.id))
    if unit_id is not None:
        staff_query = staff_query.filter(Staff.unit_id == unit_id)
    staff = (staff_query
             .order_by(Watch.order_index,
                       Staff.name).all())

    a_map = defaultdict(dict)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)
    assignments = (Assignment.query
                   .join(Staff, Staff.id == Assignment.staff_id)
                   .filter(Assignment.day >= start, Assignment.day < month_end))
    if unit_id is not None:
        assignments = assignments.filter(Staff.unit_id == unit_id)
    for a in assignments:
        a_map[a.staff_id][a.day] = a.code

    # compute daily counters + RAG for footer (prefix grouping) — EXCLUDE training shifts & excluded codes
    req = Requirement.query.filter_by(year=year, month=month, unit_id=unit_id).first()
    counters = {d: Counter() for d in days}
    for s in staff:
        if not s.is_operational:
            continue
        for d in days:
            c = a_map[s.id].get(d)
            if not c or c in EXCLUDE_FROM_COUNTERS:
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
    unit_id = active_unit_id()
    unit_obj = active_unit()

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

            # Leave/TOIL config
            leave_year_start_month = int(
                request.form.get("leave_year_start_month", 4) or 4)
            leave_entitlement_days = int(
                request.form.get("leave_entitlement_days", 0) or 0)
            leave_public_holidays = int(
                request.form.get("leave_public_holidays", 0) or 0)
            leave_carryover_days = int(
                request.form.get("leave_carryover_days", 0) or 0)

            if is_super_admin_user(current_user):
                unit_field = request.form.get("unit_id")
                try:
                    target_unit_id = int(unit_field)
                except (TypeError, ValueError):
                    target_unit_id = None
            else:
                target_unit_id = unit_id

            if not target_unit_id:
                flash("A unit is required for new staff.", "error")
            elif not all([name, staff_no, username, watch_id]):
                flash("All fields required to create staff.", "error")
            elif Staff.query.filter((Staff.username == username) | (Staff.staff_no == staff_no)).first():
                flash("Username or Staff # already exists.", "error")
            else:
                watch = Watch.query.get(int(watch_id)) if watch_id else None
                if watch and target_unit_id and watch.unit_id not in (None, target_unit_id):
                    flash("Selected watch belongs to another unit.", "error")
                    return redirect(url_for("admin"))
                s = Staff(
                    name=name,
                    staff_no=staff_no,
                    username=username,
                    watch_id=int(watch_id),
                    role=role,
                    is_wm=is_wm,
                    is_dwm=is_dwm,
                    exclude_from_ot=exclude_from_ot,
                    leave_year_start_month=leave_year_start_month,
                    leave_entitlement_days=leave_entitlement_days,
                    leave_public_holidays=leave_public_holidays,
                    leave_carryover_days=leave_carryover_days,
                    unit_id=target_unit_id,
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
            if is_super_admin_user(current_user):
                shift_unit_val = request.form.get("shift_unit_id", "")
                if shift_unit_val == "global":
                    shift_unit_id = None
                else:
                    try:
                        shift_unit_id = int(shift_unit_val)
                    except (TypeError, ValueError):
                        shift_unit_id = unit_id
            else:
                shift_unit_id = unit_id
            if not code:
                flash("Shift code is required.", "error")
            elif ShiftType.query.filter_by(code=code, unit_id=shift_unit_id).first():
                flash("Shift code already exists.", "error")
            else:
                sh = ShiftType(code=code, name=name or code, start_time=start, end_time=end,
                               is_working=is_working, is_training=is_training,
                               unit_id=shift_unit_id)
                db.session.add(sh)
                db.session.commit()
                refresh_shift_cache()
                _shift_groups_snapshot.cache_clear()
                flash("Shift added.", "ok")
                return redirect(url_for("admin"))

        if form == "shift_edit":
            sid = int(request.form.get("shift_id"))
            sh = ShiftType.query.get_or_404(sid)
            sh.name = request.form.get("name", "").strip() or sh.name
            sh.start_time = _parse_hhmm(request.form.get("start"))
            sh.end_time = _parse_hhmm(request.form.get("end"))
            sh.is_working = bool(request.form.get("is_working"))
            sh.is_training = bool(request.form.get("is_training"))
            db.session.commit()
            refresh_shift_cache()
            _shift_groups_snapshot.cache_clear()

            flash("Shift updated.", "ok")
            return redirect(url_for("admin"))

        if form == "shift_delete":
            sid = int(request.form.get("shift_id"))
            sh = ShiftType.query.get_or_404(sid)
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
                r = Requirement.query.filter_by(year=y, month=m, unit_id=unit_id).first()
                if not r:
                    r = Requirement(year=y, month=m, unit_id=unit_id)
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
    watches_query = Watch.query.order_by(Watch.order_index)
    if unit_id is not None:
        watches_query = watches_query.filter(or_(Watch.unit_id == unit_id, Watch.unit_id.is_(None)))
    watches = watches_query.all()

    shifts_query = ShiftType.query.order_by(ShiftType.code)
    if unit_id is not None:
        shifts_query = shifts_query.filter(or_(ShiftType.unit_id == unit_id, ShiftType.unit_id.is_(None)))
    shifts = shifts_query.all()

    staff_query = (Staff.query
                   .outerjoin(Watch, Staff.watch_id == Watch.id))
    if unit_id is not None:
        staff_query = staff_query.filter(Staff.unit_id == unit_id)
    staff = staff_query.order_by(Watch.order_index, Staff.name).all()
    months = [(y, m) for y in (2025, 2026) for m in range(1, 13)]
    requirements_by_month = {
        (r.year, r.month): r for r in Requirement.query.filter_by(unit_id=unit_id).all()}
    leave_query = (Leave.query
                   .join(Staff, Staff.id == Leave.staff_id)
                   .order_by(Leave.start.desc()))
    if unit_id is not None:
        leave_query = leave_query.filter(Staff.unit_id == unit_id)
    leaves = leave_query.all()
    return render_template("admin.html",
                           shifts=shifts, staff=staff, watches=watches,
                           months=months, requirements_by_month=requirements_by_month,
                           leaves=leaves, current_unit=unit_obj)

# Keep your dedicated staff edit route (ATCO edit)


@app.route("/admin/staff/<int:sid>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_staff_edit(sid):
    s = Staff.query.get_or_404(sid)
    if request.method == "POST":
        s.name = request.form.get("name", s.name).strip()
        s.staff_no = request.form.get("staff_no", s.staff_no).strip()
        s.username = request.form.get("username", s.username).strip()
        s.phone_number = _normalise_phone_number(
            request.form.get("phone_number", s.phone_number))
        s.watch_id = int(request.form.get("watch_id", s.watch_id or 0)) or None

        if is_super_admin_user(current_user):
            unit_field = request.form.get("unit_id")
            try:
                s.unit_id = int(unit_field)
            except (TypeError, ValueError):
                pass

        s.is_operational = bool(request.form.get("operational"))
        s.is_trainee = bool(request.form.get("trainee"))
        s.has_ojti = bool(request.form.get("ojti"))

        # NEW flags
        s.is_wm = bool(request.form.get("is_wm"))
        s.is_dwm = bool(request.form.get("is_dwm"))
        s.exclude_from_ot = bool(request.form.get("exclude_from_ot"))

        # update role
        new_role = request.form.get("role", s.role)
        if is_super_admin_user(current_user) or new_role != "superadmin":
            s.role = new_role

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

    watch_query = Watch.query.order_by(Watch.order_index)
    if s.unit_id is not None:
        watch_query = watch_query.filter(or_(Watch.unit_id == s.unit_id, Watch.unit_id.is_(None)))
    watches = watch_query.all()
    return render_template("staff_edit.html", s=s, watches=watches)


@app.route("/admin/staff/<int:sid>/watch-move", methods=["POST"])
@login_required
def admin_watch_move(sid):
    if not is_admin_user(current_user):
        abort(403)
    s = Staff.query.get_or_404(sid)
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

    hist = StaffWatchHistory.query.get_or_404(hid)
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

    hist = StaffWatchHistory.query.get_or_404(hid)
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
        flash("Editors or Admins only.", "error")
        return redirect(url_for("index"))

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
                s = db.session.get(Staff, staff_id)
                used_per_day_half = 2 if lv_type == "TOU8" else 1
                cur = start_d
                while cur <= end_d:
                    a = Assignment.query.filter_by(
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
            s = db.session.get(Staff, staff_id)
            cur = start_d
            while cur <= end_d:
                refresh_day_from_pattern_and_leave(s, cur)
                cur += timedelta(days=1)
            db.session.commit()
            flash("Leave recorded", "ok")
            return redirect(url_for("leave", ym=ym_param))

        if form == "leave_edit":
            lid = int(request.form["leave_id"])
            lv = Leave.query.get_or_404(lid)
            old_range = (lv.start, lv.end)
            lv.staff_id = int(request.form["staff_id"])
            lv.leave_type = request.form["leave_type"].upper()
            if lv.leave_type not in {"AL", "PL", "SPL"}:
                flash("Only AL / PL / SPL supported.", "error")
                return redirect(url_for("leave", ym=ym_param))
            lv.start = date.fromisoformat(request.form["start"])
            lv.end = date.fromisoformat(request.form["end"])
            db.session.commit()
            s = db.session.get(Staff, lv.staff_id)
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
            lv = Leave.query.get_or_404(lid)
            s = db.session.get(Staff, lv.staff_id)
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
            s = db.session.get(Staff, staff_id)
            cur = start_d
            while cur <= end_d:
                a = Assignment.query.filter_by(
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
            s = db.session.get(Staff, staff_id)
            cur = start_d
            while cur <= end_d:
                a = Assignment.query.filter_by(
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
            s = db.session.get(Staff, staff_id)
            a = Assignment.query.filter_by(staff_id=staff_id, day=day).first()
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
    s = Staff.query.get_or_404(sid)
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

    return render_template("staff_profile.html", staff=s,
                           al_days=al_days, sick_days=sick_days,
                           hours_this_month=hours_this_month,
                           cal_link=cal_link, apple_link=apple_link, google_link=google_link)

# -------------------- Metrics + CSV (date range; FYTD default) --------------------
# (… unchanged metrics functions from your file …)


def _compute_metrics_range(start_day: date, end_day: date):
    assignments = (Assignment.query
                   .filter(Assignment.day >= start_day, Assignment.day <= end_day)
                   .all())

    staff_by_id = {s.id: s for s in Staff.query.all()}
    metrics_map = {}
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
                "ot": {"A2": 0, "A4": 0, "A6": 0, "A8": 0, "SOAL": 0},
                "ot_total": 0,
                "aava_total": 0
            }
        parsed = parse_annotation(a.annotation)
        if not parsed:
            continue
        t = parsed["type"]
        if t == "EXTL":
            metrics_map[s.id]["ext_long"] += 1
            metrics_map[s.id]["ext_total"] += 1
        elif t == "EXTS":
            metrics_map[s.id]["ext_short"] += 1
            metrics_map[s.id]["ext_total"] += 1
        elif t == "SWAP":
            metrics_map[s.id]["swaps"] += 1
        elif t in ("A2", "A4", "A6", "A8", "SOAL"):
            metrics_map[s.id]["ot"][t] += 1
            metrics_map[s.id]["ot_total"] += 1
            if t in ("A2", "A4", "A6", "A8"):
                metrics_map[s.id]["aava_total"] += 1

    staff_order = (Staff.query
                   .outerjoin(Watch, Staff.watch_id == Watch.id)
                   .order_by(Watch.order_index, Staff.name).all())

    staff_metrics = []
    for s in staff_order:
        row = metrics_map.get(s.id, {
            "staff": s, "ext_long": 0, "ext_short": 0, "ext_total": 0, "swaps": 0,
            "ot": {"A2": 0, "A4": 0, "A6": 0, "A8": 0, "SOAL": 0},
            "ot_total": 0, "aava_total": 0
        })
        staff_metrics.append(row)

    totals = {
        "ext_long": sum(r["ext_long"] for r in staff_metrics),
        "ext_short": sum(r["ext_short"] for r in staff_metrics),
        "ext_total": sum(r["ext_total"] for r in staff_metrics),
        "swaps": sum(r["swaps"] for r in staff_metrics),
        "ot": {
            "A2": sum(r["ot"]["A2"] for r in staff_metrics),
            "A4": sum(r["ot"]["A4"] for r in staff_metrics),
            "A6": sum(r["ot"]["A6"] for r in staff_metrics),
            "A8": sum(r["ot"]["A8"] for r in staff_metrics),
            "SOAL": sum(r["ot"]["SOAL"] for r in staff_metrics),
        },
        "ot_total": sum(r["ot_total"] for r in staff_metrics),
        "aava_total": sum(r["aava_total"] for r in staff_metrics),
    }
    return staff_metrics, totals


def _fy_start_for(d: date) -> date:
    return date(d.year if d.month >= 4 else d.year - 1, 4, 1)


@app.route("/metrics")
@login_required
def metrics():
    if not (is_admin_user(current_user) or getattr(current_user, "role", "") in ("editor", "admin")):
        flash("Editors or Admins only.", "error")
        return redirect(url_for("index"))
    # ... existing body unchanged ...
    today = date.today()
    default_start = _fy_start_for(today)
    start_str = request.args.get("start", default_start.isoformat())
    end_str = request.args.get("end", today.isoformat())
    start_day = date.fromisoformat(start_str)
    end_day = date.fromisoformat(end_str)
    staff_metrics, totals = _compute_metrics_range(start_day, end_day)
    return render_template("metrics.html",
                           start=start_day, end=end_day,
                           staff_metrics=staff_metrics, totals=totals)


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
        t = p["type"]
        if t in ("A2", "A4", "A6", "A8"):
            aava += 1
        elif t == "SOAL":
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
        if code in WORKING_CODES:
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
        flash("Admins only!", "error")
        return redirect(url_for("index"))
    today = date.today()
    default_start = _fy_start_for(today)
    start_day = date.fromisoformat(
        request.args.get("start", default_start.isoformat()))
    end_day = date.fromisoformat(request.args.get("end", today.isoformat()))
    staff_metrics, totals = _compute_metrics_range(start_day, end_day)

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["ATCO", "Staff #", "Watch",
                "Ext L", "Ext S", "Ext Total", "Swaps",
                "A2", "A4", "A6", "A8", "SOAL",
                "AAVA Total", "OT Total"])
    for row in staff_metrics:
        s = row["staff"]
        watch = s.watch.name.replace("Watch ", "") if s.watch else "-"
        w.writerow([s.name, s.staff_no, watch,
                    row["ext_long"], row["ext_short"], row["ext_total"], row["swaps"],
                    row["ot"]["A2"], row["ot"]["A4"], row["ot"]["A6"], row["ot"]["A8"], row["ot"]["SOAL"],
                    row["aava_total"], row["ot_total"]])
    w.writerow([])
    w.writerow(["All ATCOs", "", "",
                totals["ext_long"], totals["ext_short"], totals["ext_total"], totals["swaps"],
                totals["ot"]["A2"], totals["ot"]["A4"], totals["ot"]["A6"], totals["ot"]["A8"], totals["ot"]["SOAL"],
                totals["aava_total"], totals["ot_total"]])

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
        if p and (p["type"] in ("A2", "A4", "A6", "A8", "SOAL")):
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

    staff_members = (Staff.query
                     .outerjoin(Watch, Staff.watch_id == Watch.id)
                     .order_by(Watch.order_index, Staff.name).all())

    results = []
    for s in staff_members:
        if s.exclude_from_ot:
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
            flags.append("On AL that day — SOAL required")
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
        flash("Editors or Admins only.", "error")
        return redirect(url_for("index"))

    shifts = ShiftType.query.filter_by(
        is_working=True).order_by(ShiftType.code).all()
    results = []
    chosen_date = None
    chosen_shift = None
    selected_staff_ids: set[str] = set()
    sms_body = ""

    if request.method == "POST":
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
                           selected_staff_ids=selected_staff_ids)

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
    s = Staff.query.get_or_404(sid)
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


def _leave_summary_for_month(year: int, month: int, unit_id: int | None = None):
    start, days = month_range(year, month)
    month_end = (start.replace(day=28) + timedelta(days=10)).replace(day=1)

    a_map = defaultdict(dict)
    assignment_query = (Assignment.query
                        .join(Staff, Staff.id == Assignment.staff_id)
                        .filter(Assignment.day >= start, Assignment.day < month_end))
    if unit_id is not None:
        assignment_query = assignment_query.filter(Staff.unit_id == unit_id)
    for a in assignment_query.all():
        a_map[a.staff_id][a.day] = a.code

    staff_query = (Staff.query
                   .outerjoin(Watch, Staff.watch_id == Watch.id))
    if unit_id is not None:
        staff_query = staff_query.filter(Staff.unit_id == unit_id)
    staff = staff_query.order_by(Watch.order_index, Staff.name).all()

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
        flash("Admins only!", "error")
        return redirect(url_for("index"))
    year, month = parse_ym(ym)
    unit_id = active_unit_id()
    ensure_month_requirement(year, month, unit_id=unit_id)
    generate_month(year, month, unit_id=unit_id)
    rows, codes, totals, grand_total, days = _leave_summary_for_month(
        year, month, unit_id=unit_id)
    month_title = datetime(year, month, 1).strftime("%B %Y")
    return render_template("report_leave.html",
                           ym=ym, year=year, month=month, month_title=month_title,
                           rows=rows, codes=codes,
                           totals=totals, grand_total=grand_total)


@app.route("/reports/leave.csv")
@login_required
def report_leave_csv():
    if not is_admin_user(current_user):
        flash("Admins only!", "error")
        return redirect(url_for("index"))
    ym = request.args.get("ym")
    if not ym:
        abort(400)
    year, month = parse_ym(ym)
    unit_id = active_unit_id()
    ensure_month_requirement(year, month, unit_id=unit_id)
    generate_month(year, month, unit_id=unit_id)
    rows, codes, totals, grand_total, days = _leave_summary_for_month(
        year, month, unit_id=unit_id)

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
    typ = parsed["type"]
    if typ in ("TOA8", "TOAU"):
        return 2
    if typ == "TOAI":
        return 1
    return 0


def _apply_toil_annotation_delta(staff: Staff, old_annot: str, new_annot: str):
    old_half = _toil_accrual_half_days_from_annotation(
        parse_annotation(old_annot))
    new_half = _toil_accrual_half_days_from_annotation(
        parse_annotation(new_annot))
    delta = new_half - old_half
    if delta:
        s = db.session.get(Staff, staff.id)
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
        flash("Admins only!", "error")
        return redirect(url_for("index"))
    today = date.today()
    unit_id = active_unit_id()
    people_query = (Staff.query
                    .outerjoin(Watch, Staff.watch_id == Watch.id))
    if unit_id is not None:
        people_query = people_query.filter(Staff.unit_id == unit_id)
    people = people_query.order_by(Watch.order_index, Staff.name).all()
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
        flash("Admins only!", "error")
        return redirect(url_for("index"))
    today = date.today()
    start = today - timedelta(days=365)
    unit_id = active_unit_id()
    people_query = (Staff.query
                    .outerjoin(Watch, Staff.watch_id == Watch.id))
    if unit_id is not None:
        people_query = people_query.filter(Staff.unit_id == unit_id)
    people = people_query.order_by(Watch.order_index, Staff.name).all()
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


def _lock_date_for_target_month(y: int, m: int):
    prev2_m = m - 2
    prev2_y = y
    if prev2_m <= 0:
        prev2_m += 12
        prev2_y -= 1
    return date(prev2_y, prev2_m, 20)


def _is_month_locked(y: int, m: int, today: date | None = None):
    today = today or date.today()
    return today >= _lock_date_for_target_month(y, m)


@app.route("/requests", methods=["GET", "POST"])
@login_required
def requests_page():
    today = date.today()
    unit_id = active_unit_id()

    # ---- user/editor: show next 3 months they can request into ----
    months = []
    base_y, base_m = today.year, today.month
    for k in range(1, 4):
        t_m = base_m + k
        t_y = base_y + (t_m - 1) // 12
        t_m = ((t_m - 1) % 12) + 1
        months.append((t_y, t_m))

    # ---- POST (create/delete own requests) ----
    if request.method == "POST":
        form = request.form.get("form", "")
        if form == "add":
            day = date.fromisoformat(request.form["day"])
            code = request.form["code"].upper().strip()
            if not get_shift(code):
                flash("Invalid code.", "error")
                return redirect(url_for("requests_page"))
            if _is_month_locked(day.year, day.month, today):
                flash("Requests for that month are locked.", "error")
                return redirect(url_for("requests_page"))
            ex = ShiftRequest.query.filter_by(
                staff_id=current_user.id, day=day).first()
            if not ex:
                ex = ShiftRequest(staff_id=current_user.id, day=day, code=code)
                db.session.add(ex)
            else:
                ex.code = code
                # reset status on change by the requester
                ex.status = "pending"
                ex.admin_response = ""
                ex.responded_by_id = None
                ex.responded_at = None
            db.session.commit()
            flash("Request saved.", "ok")
            return redirect(url_for("requests_page"))

        if form == "del":
            rid = int(request.form["rid"])
            req = ShiftRequest.query.get_or_404(rid)
            if req.staff_id != current_user.id and not is_admin_user(current_user):
                flash("Not allowed.", "error")
                return redirect(url_for("requests_page"))
            if _is_month_locked(req.day.year, req.day.month, today):
                flash("Requests for that month are locked.", "error")
                return redirect(url_for("requests_page"))
            db.session.delete(req)
            db.session.commit()
            flash("Request deleted.", "ok")
            return redirect(url_for("requests_page"))

    # ---- My requests (everyone) ----
    my_reqs = ShiftRequest.query.filter_by(staff_id=current_user.id).all()
    req_map = defaultdict(dict)
    for r in my_reqs:
        req_map[(r.day.year, r.day.month)][r.day] = r

    all_shifts_query = ShiftType.query.order_by(ShiftType.code)
    if unit_id is not None:
        all_shifts_query = all_shifts_query.filter(or_(ShiftType.unit_id == unit_id, ShiftType.unit_id.is_(None)))
    all_shifts = all_shifts_query.all()
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
        admin_ym = request.args.get(
            "ym") or f"{today.year:04d}-{today.month:02d}"
        ay, am = parse_ym(admin_ym)
        start_of_month, month_days = month_range(ay, am)
        end_of_month = month_days[-1]

        admin_month_title = datetime(ay, am, 1).strftime("%B %Y")
        admin_prev_ym, admin_next_ym = _clamp_prev_next(ay, am)

        # fetch only the chosen month; order by day then staff name
        admin_requests = (ShiftRequest.query
                          .join(Staff, ShiftRequest.staff_id == Staff.id)
                          .filter(ShiftRequest.day >= start_of_month,
                                  ShiftRequest.day <= end_of_month))
        if unit_id is not None and not is_super_admin_user(current_user):
            admin_requests = admin_requests.filter(Staff.unit_id == unit_id)
        admin_requests = (admin_requests
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
                           admin_next_ym=admin_next_ym)


# >>> Admin can respond to a specific request


@app.route("/admin/requests/<int:rid>/respond", methods=["POST"])
@login_required
def admin_request_respond(rid):
    if not is_admin_user(current_user):
        abort(403)
    r = ShiftRequest.query.get_or_404(rid)
    if not is_super_admin_user(current_user):
        unit_id = active_unit_id()
        if unit_id is not None and r.staff and r.staff.unit_id != unit_id:
            abort(403)
    r.admin_response = (request.form.get("admin_response") or "").strip()
    r.status = request.form.get("status", r.status or "pending")
    r.responded_by_id = getattr(current_user, "id", None)
    r.responded_at = utcnow()
    db.session.commit()
    flash("Response saved.", "ok")
    return redirect(url_for("requests_page"))

# -------------------- Manual TOIL entry page (no bulk seed in UI) --------------------


@app.route("/admin/toil/new")
@login_required
@admin_required
def admin_toil_new():
    unit_id = active_unit_id()
    atco_query = Staff.query.filter_by(is_operational=True)
    if unit_id is not None and not is_super_admin_user(current_user):
        atco_query = atco_query.filter(Staff.unit_id == unit_id)
    atcos = atco_query.order_by(Staff.name.asc()).all()
    if request.method == "POST":
        sid = int(request.form["staff_id"])
        amount = float(request.form.get("amount", "0") or 0)
        unit = request.form.get("unit", "days").lower()
        note = (request.form.get("note") or "").strip()
        s = Staff.query.get_or_404(sid)
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
    flash("Admins only.", "error")
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"], endpoint="login")
def signin_form():   # function name can be anything; endpoint is 'login'
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()
        user = Staff.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully", "ok")
            # support ?next=... to return where user was going
            nxt = request.args.get("next")
            return redirect(nxt or url_for("index"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


# -------------------- DB init (single, safe block) --------------------

with app.app_context():
    db.create_all()
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
    migrate_add_unit_fields()
    migrate_add_role_and_calendar_token()

    # >>> Ensure new ShiftRequest columns exist (SQLite safe)
    from sqlalchemy import text
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

