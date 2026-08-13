"""Operational workforce, roster, and audit SQLAlchemy models."""

from flask_login import UserMixin
from sqlalchemy.ext.hybrid import hybrid_property
from werkzeug.security import check_password_hash

# The application constructs the sole SQLAlchemy extension before importing
# this model module. This compatibility edge will disappear when extension
# ownership moves to atcroster.extensions.
from atcroster.application import db, utcnow

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
    protected_roster_months_ahead = db.Column(
        db.Integer, nullable=False, default=2, server_default="2"
    )
    preserve_redundant_overrides = db.Column(
        db.Boolean, nullable=False, default=True, server_default="1"
    )
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
        from atcroster.application import UnitMembership

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
    role = db.Column(db.String(32), nullable=False, default="user")
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
    employment_start_date = db.Column(db.Date)
    unit_join_date = db.Column(db.Date)
    roster_start_date = db.Column(db.Date)
    employment_type = db.Column(
        db.String(20), nullable=False, default="FULL_TIME"
    )
    contracted_minutes_per_week = db.Column(db.Integer)
    workforce_notes = db.Column(db.Text, nullable=False, default="")
    final_unit_date = db.Column(db.Date)
    final_operational_duty_date = db.Column(db.Date)
    employment_end_date = db.Column(db.Date)
    leaving_reason_category = db.Column(db.String(40), nullable=False, default="")
    leaving_notes = db.Column(db.Text, nullable=False, default="")

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
        db.UniqueConstraint("unit_id", "id", name="uq_staff_unit_id"),
        db.CheckConstraint(
            "employment_type IN ('FULL_TIME','PART_TIME')",
            name="ck_staff_employment_type",
        ),
        db.CheckConstraint(
            "contracted_minutes_per_week IS NULL OR "
            "contracted_minutes_per_week >= 0",
            name="ck_staff_contracted_minutes_nonnegative",
        ),
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
    __table_args__ = (
        db.UniqueConstraint("unit_id", "code", name="uq_shift_unit_code"),
        db.UniqueConstraint("unit_id", "id", name="uq_shift_unit_id"),
    )


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
    provider = db.Column(db.String(30), nullable=False, default="messagemedia")
    delivery_status = db.Column(db.String(30), nullable=False, default="submitted")


class SmsSenderRegistration(db.Model):
    """Dashboard-assisted verification record for a Watch Manager's own mobile."""
    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    staff_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
    number = db.Column(db.String(20), nullable=False)
    provider = db.Column(db.String(30), nullable=False, default="messagemedia")
    status = db.Column(db.String(30), nullable=False, default="pending_dashboard_verification")
    provider_identifier = db.Column(db.String(120), nullable=False, default="")
    verification_requested_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    verified_at = db.Column(db.DateTime, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    __table_args__ = (db.UniqueConstraint("unit_id", "staff_id", "number", "provider", name="uq_sms_sender_registration"),)


class HandoverField(db.Model):
    """An airport-defined prompt shown on each operational handover."""
    __tablename__ = "handover_field"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    section_name = db.Column(db.String(80), nullable=False, default="Operational overview")
    label = db.Column(db.String(120), nullable=False)
    field_type = db.Column(db.String(20), nullable=False, default="text")
    options_json = db.Column(db.Text, nullable=False, default="[]")
    help_text = db.Column(db.String(240), nullable=False, default="")
    placeholder = db.Column(db.String(160), nullable=False, default="")
    required = db.Column(db.Boolean, nullable=False, default=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    display_order = db.Column(db.Integer, nullable=False, default=100)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class HandoverRecord(db.Model):
    """Immutable roster and field snapshot shared between watch managers."""
    __tablename__ = "handover_record"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    status = db.Column(db.String(20), nullable=False, default="published", index=True)
    created_by_id = db.Column(db.Integer, nullable=False, index=True)
    created_by_name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    target_shift_day = db.Column(db.Date, nullable=True, index=True)
    target_shift_code = db.Column(db.String(10), nullable=False, default="")
    target_shift_name = db.Column(db.String(80), nullable=False, default="")
    target_shift_start = db.Column(db.DateTime, nullable=True)
    next_shift_json = db.Column(db.Text, nullable=False, default="{}")
    responses_json = db.Column(db.Text, nullable=False, default="[]")


class HandoverOperationalState(db.Model):
    __tablename__ = "handover_operational_state"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, unique=True, index=True)
    runway_in_use = db.Column(db.String(40), nullable=False, default="")
    runway_options_json = db.Column(db.Text, nullable=False, default="[]")
    metar_icao = db.Column(db.String(4), nullable=False, default="")
    updated_by_id = db.Column(db.Integer)
    updated_by_name = db.Column(db.String(80), nullable=False, default="")
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class HandoverEquipment(db.Model):
    __tablename__ = "handover_equipment"

    id = db.Column(db.Integer, primary_key=True)
    unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    status = db.Column(db.String(10), nullable=False, default="green")
    note = db.Column(db.String(240), nullable=False, default="")
    active = db.Column(db.Boolean, nullable=False, default=True)
    display_order = db.Column(db.Integer, nullable=False, default=100)
    updated_by_id = db.Column(db.Integer)
    updated_by_name = db.Column(db.String(80), nullable=False, default="")
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


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
    # Compatibility note: ``code`` remains the legacy materialised value while
    # readers are moved in stages to ``effective_code``.
    generated_code = db.Column(db.String(10))
    override_code = db.Column(db.String(10))
    generated_from_pattern_id = db.Column(db.Integer)
    generated_from_pattern_day_index = db.Column(db.Integer)
    generated_at = db.Column(db.DateTime)
    generation_event_id = db.Column(db.Integer)
    generation_version = db.Column(db.String(40))
    override_type = db.Column(db.String(40))
    override_reason = db.Column(db.String(500), nullable=False, default="")
    override_by_user_id = db.Column(db.Integer)
    override_at = db.Column(db.DateTime)
    override_classification = db.Column(db.String(50))
    override_classified_at = db.Column(db.Date)
    source = db.Column(db.String(10), default="auto")
    note = db.Column(db.String(140), default="")
    # Annotation code (managed via AnnotationType, optional suffix like A6M)
    annotation = db.Column(db.String(20), default="")
    # User-facing detail for the annotation. Kept separate from system notes.
    annotation_note = db.Column(db.String(140), default="")
    version = db.Column(db.Integer, nullable=False, default=1)
    lock_status = db.Column(db.String(20), nullable=False, default="UNLOCKED")
    locked_by_user_id = db.Column(db.Integer)
    locked_at = db.Column(db.DateTime)
    lock_reason = db.Column(db.String(250), nullable=False, default="")

    @hybrid_property
    def effective_code(self):
        """Return the editor override, then baseline, then legacy fallback."""
        if self.override_code is not None:
            return self.override_code
        if self.generated_code is not None:
            return self.generated_code
        return self.code

    @effective_code.expression
    def effective_code(cls):
        return db.func.coalesce(cls.override_code, cls.generated_code, cls.code)

    def materialise_effective_code(self):
        """Keep the legacy column aligned during the compatibility rollout."""
        effective = self.effective_code
        self.code = effective if effective is not None else "OFF"
        return self.code

    def set_generated_baseline(
        self,
        code,
        *,
        generated_at=None,
        generation_version=None,
        pattern_id=None,
        pattern_day_index=None,
        generation_event_id=None,
    ):
        self.generated_code = code
        self.generated_at = generated_at or utcnow()
        self.generation_version = generation_version
        self.generated_from_pattern_id = pattern_id
        self.generated_from_pattern_day_index = pattern_day_index
        self.generation_event_id = generation_event_id
        return self.materialise_effective_code()

    def set_editor_override(self, code, *, actor_id=None, reason="", override_type="MANUAL"):
        self.override_code = code
        self.override_type = override_type
        self.override_reason = (reason or "")[:500]
        self.override_by_user_id = actor_id
        self.override_at = utcnow()
        return self.materialise_effective_code()

    def clear_editor_override(self):
        self.override_code = None
        self.override_type = None
        self.override_reason = ""
        self.override_by_user_id = None
        self.override_at = None
        return self.materialise_effective_code()
    __table_args__ = (db.UniqueConstraint(
        "unit_id", "staff_id", "day", name="uniq_unit_staff_day"),
        db.Index("ix_assignment_unit_day", "unit_id", "day"),
        db.CheckConstraint(
            "lock_status IN ('UNLOCKED','SOFT_LOCKED','HARD_LOCKED')",
            name="ck_assignment_lock_status",
        ),)


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
    effective_to = db.Column(db.Date)
    reason = db.Column(db.String(500), nullable=False, default="")
    alignment_mode = db.Column(
        db.String(40), nullable=False, default="ALIGN_WITH_DESTINATION_WATCH"
    )
    starting_cycle_day = db.Column(db.Integer)
    pattern_anchor = db.Column(db.Date)
    staff = db.relationship("Staff", backref="watch_history")
    watch = db.relationship("Watch")
