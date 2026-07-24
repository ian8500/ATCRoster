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
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"))
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
        __table_args__ = (db.UniqueConstraint("unit_id", "code", name="uq_qualification_unit_code"),)

    class PersonQualification(db.Model):
        __tablename__ = "person_qualification"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        person_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True)
        qualification_type_id = db.Column(db.Integer, db.ForeignKey("qualification_type.id"), nullable=False)
        expires_on = db.Column(db.Date)
        status = db.Column(db.String(20), nullable=False, default="valid")

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

    return SimpleNamespace(**locals())
