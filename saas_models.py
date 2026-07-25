"""Control-plane and product-domain model registration.

Kept separate from the legacy UI module so the application can be split into
central and per-airport metadata without a frontend rewrite.
"""
from __future__ import annotations

from types import SimpleNamespace


def register_saas_models(db, utcnow):
    class PlatformIdentity(db.Model):
        __tablename__ = "platform_identity"
        id = db.Column(db.Integer, primary_key=True)
        public_id = db.Column(db.String(64), unique=True, nullable=False)
        username = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(255), nullable=False)
        mfa_secret_encrypted = db.Column(db.Text)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        last_active_at = db.Column(db.DateTime)

    class UnitMembership(db.Model):
        __tablename__ = "unit_membership"
        id = db.Column(db.Integer, primary_key=True)
        identity_id = db.Column(db.Integer, db.ForeignKey("platform_identity.id"), nullable=False)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        # Opaque operational person id. The referenced row lives in the
        # airport database, so a cross-database foreign key is impossible.
        person_id = db.Column(db.Integer)
        role = db.Column(db.String(30), nullable=False)
        status = db.Column(db.String(20), nullable=False, default="invited")
        permissions_json = db.Column(db.Text, nullable=False, default="{}")
        invited_at = db.Column(db.DateTime, default=utcnow)
        activated_at = db.Column(db.DateTime)
        suspended_at = db.Column(db.DateTime)
        __table_args__ = (
            db.UniqueConstraint("identity_id", "unit_id", name="uq_membership_identity_unit"),
        )

    class SecureInvitation(db.Model):
        __tablename__ = "secure_invitation"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        token_digest = db.Column(db.String(128), unique=True, nullable=False)
        role = db.Column(db.String(30), nullable=False)
        expires_at = db.Column(db.DateTime, nullable=False)
        accepted_at = db.Column(db.DateTime)
        disabled_at = db.Column(db.DateTime)

    class DatabaseRoutingMetadata(db.Model):
        __tablename__ = "database_routing_metadata"
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), primary_key=True)
        secret_name = db.Column(db.String(120), nullable=False)
        health = db.Column(db.String(20), nullable=False, default="unknown")
        migration_version = db.Column(db.String(64), nullable=False, default="")
        storage_bytes = db.Column(db.BigInteger, nullable=False, default=0)

    class FeatureFlag(db.Model):
        __tablename__ = "feature_flag"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        key = db.Column(db.String(80), nullable=False)
        enabled = db.Column(db.Boolean, nullable=False, default=False)
        __table_args__ = (db.UniqueConstraint("unit_id", "key", name="uq_feature_unit_key"),)

    class PlanHistory(db.Model):
        __tablename__ = "plan_history"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        plan = db.Column(db.String(40), nullable=False)
        active_user_limit = db.Column(db.Integer, nullable=False)
        effective_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        changed_by_identity_id = db.Column(db.Integer, db.ForeignKey("platform_identity.id"))

    class AggregateUsageEvent(db.Model):
        __tablename__ = "aggregate_usage_event"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        event_type = db.Column(db.String(60), nullable=False)
        count = db.Column(db.Integer, nullable=False, default=1)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class SuperAdminAudit(db.Model):
        __tablename__ = "super_admin_audit"
        id = db.Column(db.Integer, primary_key=True)
        actor_identity_id = db.Column(db.Integer, db.ForeignKey("platform_identity.id"), nullable=False)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False)
        action = db.Column(db.String(80), nullable=False)
        safe_summary = db.Column(db.String(500), nullable=False)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class QualificationType(db.Model):
        __tablename__ = "qualification_type"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        code = db.Column(db.String(30), nullable=False)
        label = db.Column(db.String(100), nullable=False)
        warning_days_csv = db.Column(db.String(100), nullable=False, default="180,90,60,30")
        expiry_required = db.Column(db.Boolean, nullable=False, default=True)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (db.UniqueConstraint("unit_id", "code", name="uq_qualification_unit_code"),)

    class PersonQualification(db.Model):
        __tablename__ = "person_qualification"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        qualification_type_id = db.Column(db.Integer, db.ForeignKey("qualification_type.id"), nullable=False)
        issued_on = db.Column(db.Date)
        valid_from = db.Column(db.Date)
        expires_on = db.Column(db.Date)
        status = db.Column(db.String(20), nullable=False, default="valid")
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "person_id", "qualification_type_id",
                name="uq_person_qualification_type",
            ),
        )

    class PersonQualificationHistory(db.Model):
        __tablename__ = "person_qualification_history"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_qualification_id = db.Column(
            db.Integer, db.ForeignKey("person_qualification.id"),
            nullable=False, index=True,
        )
        actor_id = db.Column(db.Integer, nullable=False)
        action = db.Column(db.String(30), nullable=False)
        snapshot_json = db.Column(db.Text, nullable=False)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class RosterPublication(db.Model):
        __tablename__ = "roster_publication"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        year = db.Column(db.Integer, nullable=False)
        month = db.Column(db.Integer, nullable=False)
        version = db.Column(db.Integer, nullable=False)
        state = db.Column(db.String(20), nullable=False, default="draft")
        snapshot_json = db.Column(db.Text, nullable=False, default="{}")
        published_at = db.Column(db.DateTime)
        superseded_at = db.Column(db.DateTime)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "year", "month", "version", name="uq_roster_publication_version"),
        )

    class RosterAcknowledgement(db.Model):
        __tablename__ = "roster_acknowledgement"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        publication_id = db.Column(db.Integer, db.ForeignKey("roster_publication.id"), nullable=False)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
        acknowledged_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class Scenario(db.Model):
        __tablename__ = "scenario"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        name = db.Column(db.String(120), nullable=False)
        changes_json = db.Column(db.Text, nullable=False, default="[]")
        created_by_id = db.Column(db.Integer, nullable=False)
        approved_by_id = db.Column(db.Integer)
        applied_at = db.Column(db.DateTime)

    class OperationalPosition(db.Model):
        __tablename__ = "operational_position"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        code = db.Column(db.String(30), nullable=False)
        label = db.Column(db.String(120), nullable=False)
        description = db.Column(db.Text, nullable=False, default="")
        is_safety_critical = db.Column(db.Boolean, nullable=False, default=True)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "code", name="uq_position_unit_code"),
        )

    class PositionEndorsement(db.Model):
        __tablename__ = "position_endorsement"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        position_id = db.Column(db.Integer, db.ForeignKey("operational_position.id"), nullable=False, index=True)
        valid_from = db.Column(db.Date, nullable=False)
        valid_until = db.Column(db.Date)
        status = db.Column(db.String(20), nullable=False, default="valid")
        restrictions = db.Column(db.Text, nullable=False, default="")
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "person_id", "position_id",
                name="uq_position_endorsement_person",
            ),
        )

    class PositionRequirement(db.Model):
        __tablename__ = "position_requirement"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        day = db.Column(db.Date, nullable=False, index=True)
        shift_code = db.Column(db.String(10), nullable=False)
        position_id = db.Column(db.Integer, db.ForeignKey("operational_position.id"), nullable=False)
        required_count = db.Column(db.Integer, nullable=False, default=1)
        contingency_count = db.Column(db.Integer, nullable=False, default=0)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "day", "shift_code", "position_id",
                name="uq_position_requirement_day_shift",
            ),
        )

    class BreakPlan(db.Model):
        __tablename__ = "break_plan"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        day = db.Column(db.Date, nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        position_id = db.Column(db.Integer, db.ForeignKey("operational_position.id"))
        start_time = db.Column(db.Time, nullable=False)
        end_time = db.Column(db.Time, nullable=False)
        kind = db.Column(db.String(20), nullable=False, default="break")
        state = db.Column(db.String(20), nullable=False, default="planned")
        recorded_by_id = db.Column(db.Integer, nullable=False)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class AchievedDuty(db.Model):
        __tablename__ = "achieved_duty"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        day = db.Column(db.Date, nullable=False, index=True)
        planned_assignment_id = db.Column(db.Integer, db.ForeignKey("assignment.id"))
        actual_start = db.Column(db.DateTime, nullable=False)
        actual_end = db.Column(db.DateTime, nullable=False)
        duty_type = db.Column(db.String(30), nullable=False, default="operational")
        variance_reason = db.Column(db.String(500), nullable=False, default="")
        recorded_by_id = db.Column(db.Integer, nullable=False)
        recorded_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "person_id", "day", name="uq_achieved_duty_person_day"),
        )

    class FatigueReport(db.Model):
        __tablename__ = "fatigue_report"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        duty_day = db.Column(db.Date, nullable=False, index=True)
        severity = db.Column(db.String(20), nullable=False)
        summary = db.Column(db.String(500), nullable=False)
        status = db.Column(db.String(20), nullable=False, default="open")
        reported_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        manager_response = db.Column(db.String(1000), nullable=False, default="")
        reviewed_by_id = db.Column(db.Integer)
        reviewed_at = db.Column(db.DateTime)
        closed_at = db.Column(db.DateTime)

    class RosterRuleVersion(db.Model):
        __tablename__ = "roster_rule_version"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        version = db.Column(db.Integer, nullable=False)
        name = db.Column(db.String(120), nullable=False)
        rules_json = db.Column(db.Text, nullable=False, default="{}")
        state = db.Column(db.String(20), nullable=False, default="draft")
        effective_from = db.Column(db.Date)
        change_reference = db.Column(db.String(120), nullable=False, default="")
        consultation_summary = db.Column(db.Text, nullable=False, default="")
        approved_by_id = db.Column(db.Integer)
        approved_at = db.Column(db.DateTime)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "version", name="uq_roster_rule_unit_version"),
        )

    class MfaCredential(db.Model):
        __tablename__ = "mfa_credential"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, unique=True)
        encrypted_secret = db.Column(db.Text, nullable=False)
        enabled = db.Column(db.Boolean, nullable=False, default=False)
        enrolled_at = db.Column(db.DateTime)
        last_used_step = db.Column(db.BigInteger)
        recovery_codes_digest = db.Column(db.Text, nullable=False, default="[]")

    return SimpleNamespace(**locals())
