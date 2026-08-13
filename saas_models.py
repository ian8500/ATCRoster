"""Control-plane and product-domain model registration.

Kept separate from the legacy UI module so the application can be split into
central and per-airport metadata without a frontend rewrite.
"""
from __future__ import annotations

from types import SimpleNamespace

from atcroster.extensions import register_unit_membership_model


def register_saas_models(db, utcnow):
    class PlatformIdentity(db.Model):
        __tablename__ = "platform_identity"
        id = db.Column(db.Integer, primary_key=True)
        public_id = db.Column(db.String(64), unique=True, nullable=False)
        username = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(255), nullable=False)
        email = db.Column(db.String(254), nullable=False, default="")
        mfa_secret_encrypted = db.Column(db.Text)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        last_active_at = db.Column(db.DateTime)

        unit_id = 0
        name = "Platform Administrator"

        @property
        def role(self):
            return (
                "superadmin"
                if self.public_id.startswith("platform-") else "inactive"
            )

        @property
        def membership_status(self):
            return (
                "active"
                if self.public_id.startswith("platform-") else "pending"
            )

        @property
        def is_authenticated(self):
            return True

        @property
        def is_active(self):
            return True

        @property
        def is_anonymous(self):
            return False

        def get_id(self):
            return f"platform-identity:{self.id}"

        def check_password(self, password):
            from werkzeug.security import check_password_hash
            return check_password_hash(self.password_hash, password)

        def set_password(self, password):
            from werkzeug.security import generate_password_hash
            self.password_hash = generate_password_hash(password)

    class PlatformMfaCredential(db.Model):
        __tablename__ = "platform_mfa_credential"
        id = db.Column(db.Integer, primary_key=True)
        identity_id = db.Column(
            db.Integer, db.ForeignKey("platform_identity.id"),
            nullable=False, unique=True,
        )
        encrypted_secret = db.Column(db.Text, nullable=False)
        enabled = db.Column(db.Boolean, nullable=False, default=False)
        enrolled_at = db.Column(db.DateTime)
        last_used_step = db.Column(db.BigInteger)
        recovery_codes_digest = db.Column(db.Text, nullable=False, default="[]")
        reset_required = db.Column(db.Boolean, nullable=False, default=False)

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

    register_unit_membership_model(UnitMembership)

    class SecureInvitation(db.Model):
        __tablename__ = "secure_invitation"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        token_digest = db.Column(db.String(128), unique=True, nullable=False)
        role = db.Column(db.String(30), nullable=False)
        # A nullable uniqueness sentinel makes the "one active bootstrap"
        # invariant portable across PostgreSQL and SQLite. It is cleared when
        # the invitation is consumed or revoked.
        active_bootstrap_key = db.Column(db.String(20))
        # Opaque id of an already configured roster person. The operational
        # row may live in the airport database, so no cross-database FK.
        target_person_id = db.Column(db.Integer, index=True)
        expires_at = db.Column(db.DateTime, nullable=False)
        accepted_at = db.Column(db.DateTime)
        disabled_at = db.Column(db.DateTime)
        issued_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "role", "active_bootstrap_key",
                name="uq_active_bootstrap_invitation",
            ),
        )

    class SignupWorkflow(db.Model):
        __tablename__ = "signup_workflow"
        id = db.Column(db.Integer, primary_key=True)
        invitation_id = db.Column(
            db.Integer, db.ForeignKey("secure_invitation.id"),
            nullable=False, unique=True,
        )
        idempotency_key = db.Column(db.String(64), nullable=False, unique=True)
        state = db.Column(db.String(40), nullable=False, default="pending")
        normalized_username = db.Column(db.String(120), nullable=False)
        identity_id = db.Column(db.Integer, db.ForeignKey("platform_identity.id"))
        operational_person_id = db.Column(db.Integer)
        membership_id = db.Column(db.Integer, db.ForeignKey("unit_membership.id"))
        attempt_count = db.Column(db.Integer, nullable=False, default=0)
        last_error_code = db.Column(db.String(80), nullable=False, default="")
        compensation_state = db.Column(db.String(40), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class RecoveryRequest(db.Model):
        __tablename__ = "recovery_request"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), index=True)
        identity_id = db.Column(
            db.Integer, db.ForeignKey("platform_identity.id"), index=True
        )
        person_id = db.Column(db.Integer)
        approval_token_digest = db.Column(
            db.String(64), unique=True, nullable=False
        )
        reset_token_digest = db.Column(db.String(64), unique=True)
        state = db.Column(
            db.String(24), nullable=False, default="pending_approval"
        )
        expires_at = db.Column(db.DateTime, nullable=False)
        approved_at = db.Column(db.DateTime)
        completed_at = db.Column(db.DateTime)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class DatabaseRoutingMetadata(db.Model):
        __tablename__ = "database_routing_metadata"
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), primary_key=True)
        secret_name = db.Column(db.String(120), nullable=False)
        health = db.Column(db.String(20), nullable=False, default="unknown")
        migration_version = db.Column(db.String(64), nullable=False, default="")
        storage_bytes = db.Column(db.BigInteger, nullable=False, default=0)
        provisioning_state = db.Column(
            db.String(40), nullable=False, default="pending"
        )
        last_error_code = db.Column(db.String(80), nullable=False, default="")
        attempt_count = db.Column(db.Integer, nullable=False, default=0)
        last_attempt_at = db.Column(db.DateTime)
        ready_at = db.Column(db.DateTime)

    class ProvisioningJob(db.Model):
        __tablename__ = "provisioning_job"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"),
            nullable=False, index=True,
        )
        idempotency_key = db.Column(db.String(64), nullable=False, unique=True)
        state = db.Column(db.String(30), nullable=False, default="queued")
        # Only active jobs carry this sentinel. Completed/cancelled/failed
        # history uses NULL and therefore remains unrestricted.
        active_key = db.Column(db.String(20))
        attempt_count = db.Column(db.Integer, nullable=False, default=0)
        next_attempt_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        locked_at = db.Column(db.DateTime)
        worker_id = db.Column(db.String(64), nullable=False, default="")
        lease_owner = db.Column(db.String(64), nullable=False, default="")
        lease_expires_at = db.Column(db.DateTime, index=True)
        cancel_requested = db.Column(db.Boolean, nullable=False, default=False)
        last_error_code = db.Column(db.String(80), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "active_key", name="uq_active_provisioning_job"
            ),
        )

    class WorkerHeartbeat(db.Model):
        __tablename__ = "worker_heartbeat"
        worker_id = db.Column(db.String(64), primary_key=True)
        process_type = db.Column(db.String(30), nullable=False, default="provisioning")
        state = db.Column(db.String(30), nullable=False, default="starting")
        last_seen_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        started_at = db.Column(db.DateTime, nullable=False, default=utcnow)


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
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"))
        action = db.Column(db.String(80), nullable=False)
        safe_summary = db.Column(db.String(500), nullable=False)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class CentralSecurityAudit(db.Model):
        __tablename__ = "central_security_audit"
        id = db.Column(db.Integer, primary_key=True)
        identity_id = db.Column(db.Integer, db.ForeignKey("platform_identity.id"))
        event_type = db.Column(db.String(80), nullable=False)
        principal_digest = db.Column(db.String(32), nullable=False, default="")
        outcome = db.Column(db.String(20), nullable=False)
        safe_detail = db.Column(db.String(200), nullable=False, default="")
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
        valid_to = db.Column(db.Date)
        suspended_from = db.Column(db.Date)
        suspended_to = db.Column(db.Date)
        evidence_reference = db.Column(db.String(500), nullable=False, default="")
        created_by_user_id = db.Column(db.Integer)
        status = db.Column(db.String(20), nullable=False, default="valid")
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "person_id", "qualification_type_id",
                name="uq_person_qualification_type",
            ),
            db.CheckConstraint(
                "valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from",
                name="ck_person_qualification_valid_range",
            ),
            db.CheckConstraint(
                "suspended_to IS NULL OR suspended_from IS NULL OR "
                "suspended_to >= suspended_from",
                name="ck_person_qualification_suspension_range",
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

    class OperationalPositionGroup(db.Model):
        __tablename__ = "operational_position_group"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        name = db.Column(db.String(80), nullable=False)
        display_order = db.Column(db.Integer, nullable=False, default=100)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "name", name="uq_position_group_unit_name"),
        )

    class OperationalPosition(db.Model):
        __tablename__ = "operational_position"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        code = db.Column(db.String(30), nullable=False)
        label = db.Column(db.String(120), nullable=False)
        description = db.Column(db.Text, nullable=False, default="")
        display_order = db.Column(db.Integer, nullable=False, default=100)
        group_name = db.Column(db.String(80), nullable=False, default="")
        maximum_session_duration_minutes = db.Column(
            db.Integer, nullable=False, default=120
        )
        currency_category_id = db.Column(
            db.Integer, db.ForeignKey("position_currency_category.id")
        )
        supporting_participants_allowed = db.Column(
            db.Boolean, nullable=False, default=True
        )
        multiple_supporting_participants_allowed = db.Column(
            db.Boolean, nullable=False, default=True
        )
        training_supported = db.Column(db.Boolean, nullable=False, default=True)
        assessment_supported = db.Column(db.Boolean, nullable=False, default=True)
        is_safety_critical = db.Column(db.Boolean, nullable=False, default=True)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "code", name="uq_position_unit_code"),
            db.UniqueConstraint("unit_id", "id", name="uq_position_unit_id"),
        )

    class OperationalPositionTimeAllowance(db.Model):
        __tablename__ = "operational_position_time_allowance"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False)
        position_id = db.Column(db.Integer, nullable=False)
        weekday = db.Column(db.Integer, nullable=False)
        start_hour = db.Column(db.Integer, nullable=False)
        maximum_duration_minutes = db.Column(db.Integer, nullable=False)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "position_id"],
                ["operational_position.unit_id", "operational_position.id"],
                name="fk_position_allowance_position_unit",
                ondelete="CASCADE",
            ),
            db.UniqueConstraint(
                "unit_id", "position_id", "weekday", "start_hour",
                name="uq_position_allowance_slot",
            ),
            db.CheckConstraint("weekday >= 0 AND weekday <= 6", name="ck_position_allowance_weekday"),
            db.CheckConstraint("start_hour >= 0 AND start_hour <= 23", name="ck_position_allowance_start_hour"),
            db.CheckConstraint(
                "maximum_duration_minutes >= 1 AND maximum_duration_minutes <= 1440",
                name="ck_position_allowance_duration",
            ),
            db.Index(
                "ix_position_allowance_lookup",
                "unit_id", "position_id", "weekday", "start_hour",
            ),
        )

    class PositionCurrencyCategory(db.Model):
        __tablename__ = "position_currency_category"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        code = db.Column(db.String(30), nullable=False)
        label = db.Column(db.String(120), nullable=False)
        description = db.Column(db.Text, nullable=False, default="")
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "code", name="uq_position_currency_category_code"
            ),
        )

    class PositionParticipantRole(db.Model):
        __tablename__ = "position_participant_role"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        code = db.Column(db.String(30), nullable=False)
        label = db.Column(db.String(80), nullable=False)
        is_primary = db.Column(db.Boolean, nullable=False, default=False)
        counts_for_currency = db.Column(db.Boolean, nullable=False, default=False)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        __table_args__ = (
            db.UniqueConstraint(
                "unit_id", "code", name="uq_position_participant_role_code"
            ),
        )

    class PositionStatusEvent(db.Model):
        __tablename__ = "position_status_event"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        position_id = db.Column(
            db.Integer, db.ForeignKey("operational_position.id"),
            nullable=False, index=True,
        )
        status = db.Column(db.String(20), nullable=False)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
        actor_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
        reason = db.Column(db.String(250), nullable=False, default="")
        transaction_key = db.Column(db.String(64), nullable=False, unique=True)

    class PositionSession(db.Model):
        __tablename__ = "position_session"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        position_id = db.Column(
            db.Integer, db.ForeignKey("operational_position.id"),
            nullable=False, index=True,
        )
        primary_person_id = db.Column(
            db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True
        )
        session_type = db.Column(db.String(20), nullable=False, default="operational")
        started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
        ended_at = db.Column(db.DateTime, index=True)
        ended_reason = db.Column(db.String(40), nullable=False, default="")
        maximum_duration_seconds = db.Column(db.Integer)
        warning_threshold_seconds = db.Column(db.Integer)
        due_off_at = db.Column(db.DateTime)
        currency_category_id = db.Column(
            db.Integer, db.ForeignKey("position_currency_category.id")
        )
        created_by_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
        corrected_at = db.Column(db.DateTime)
        corrected_by_id = db.Column(db.Integer, db.ForeignKey("staff.id"))
        correction_reason = db.Column(db.String(500), nullable=False, default="")
        is_void = db.Column(db.Boolean, nullable=False, default=False)
        version = db.Column(db.Integer, nullable=False, default=1)
        transaction_key = db.Column(db.String(64), nullable=False, unique=True)
        __table_args__ = (
            db.CheckConstraint(
                "ended_at IS NULL OR ended_at >= started_at",
                name="ck_position_session_time_order",
            ),
        )

    class PositionSessionParticipant(db.Model):
        __tablename__ = "position_session_participant"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        session_id = db.Column(
            db.Integer, db.ForeignKey("position_session.id"), nullable=False, index=True
        )
        person_id = db.Column(
            db.Integer, db.ForeignKey("staff.id"), nullable=False, index=True
        )
        role_id = db.Column(
            db.Integer, db.ForeignKey("position_participant_role.id"),
            nullable=False, index=True,
        )
        started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
        ended_at = db.Column(db.DateTime, index=True)
        ended_reason = db.Column(db.String(40), nullable=False, default="")
        transaction_key = db.Column(db.String(64), nullable=False, unique=True)
        __table_args__ = (
            db.CheckConstraint(
                "ended_at IS NULL OR ended_at >= started_at",
                name="ck_position_participant_time_order",
            ),
        )

    class ControllerKioskCredential(db.Model):
        __tablename__ = "controller_kiosk_credential"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        person_id = db.Column(
            db.Integer, db.ForeignKey("staff.id"), nullable=False, unique=True
        )
        pin_hash = db.Column(db.String(255), nullable=False)
        enabled = db.Column(db.Boolean, nullable=False, default=True)
        failed_attempts = db.Column(db.Integer, nullable=False, default=0)
        locked_until = db.Column(db.DateTime)
        changed_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    class PositionSessionAudit(db.Model):
        __tablename__ = "position_session_audit"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(
            db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True
        )
        session_id = db.Column(db.Integer, db.ForeignKey("position_session.id"), index=True)
        position_id = db.Column(
            db.Integer, db.ForeignKey("operational_position.id"), index=True
        )
        actor_id = db.Column(db.Integer, db.ForeignKey("staff.id"), nullable=False)
        action = db.Column(db.String(40), nullable=False, index=True)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
        old_value_json = db.Column(db.Text, nullable=False, default="{}")
        new_value_json = db.Column(db.Text, nullable=False, default="{}")
        reason = db.Column(db.String(500), nullable=False, default="")
        transaction_key = db.Column(db.String(64), nullable=False, index=True)

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

    class ToilTransaction(db.Model):
        __tablename__ = "toil_transaction"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        person_id = db.Column(db.Integer, nullable=False, index=True)
        delta_half_days = db.Column(db.Integer, nullable=False)
        balance_after_half_days = db.Column(db.Integer, nullable=False)
        reason = db.Column(db.String(500), nullable=False)
        source_type = db.Column(db.String(40), nullable=False)
        source_id = db.Column(db.Integer)
        actor_id = db.Column(db.Integer, nullable=False)
        transaction_key = db.Column(db.String(64), nullable=False)
        occurred_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "person_id"],
                ["staff.unit_id", "staff.id"],
                name="fk_toil_transaction_person_unit",
            ),
            db.UniqueConstraint(
                "unit_id", "transaction_key", name="uq_toil_transaction_unit_key"
            ),
            db.CheckConstraint(
                "delta_half_days <> 0", name="ck_toil_transaction_nonzero"
            ),
        )

    class WorkPattern(db.Model):
        __tablename__ = "work_pattern"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        name = db.Column(db.String(120), nullable=False)
        description = db.Column(db.Text, nullable=False, default="")
        cycle_length_days = db.Column(db.Integer, nullable=False)
        contracted_minutes_per_cycle = db.Column(db.Integer, nullable=False, default=0)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "name", name="uq_work_pattern_unit_name"),
            db.UniqueConstraint("unit_id", "id", name="uq_work_pattern_unit_id"),
            db.CheckConstraint("cycle_length_days > 0", name="ck_work_pattern_cycle_positive"),
            db.CheckConstraint(
                "contracted_minutes_per_cycle >= 0",
                name="ck_work_pattern_minutes_nonnegative",
            ),
        )

    class WorkPatternDay(db.Model):
        __tablename__ = "work_pattern_day"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        work_pattern_id = db.Column(db.Integer, nullable=False, index=True)
        day_index = db.Column(db.Integer, nullable=False)
        day_type = db.Column(db.String(32), nullable=False)
        fixed_shift_type_id = db.Column(db.Integer)
        required_work = db.Column(db.Boolean, nullable=False, default=False)
        notes = db.Column(db.String(500), nullable=False, default="")
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "work_pattern_id"],
                ["work_pattern.unit_id", "work_pattern.id"],
                name="fk_work_pattern_day_pattern_unit", ondelete="CASCADE",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "fixed_shift_type_id"],
                ["shift_type.unit_id", "shift_type.id"],
                name="fk_work_pattern_day_shift_unit",
            ),
            db.UniqueConstraint(
                "unit_id", "work_pattern_id", "day_index",
                name="uq_work_pattern_day_index",
            ),
            db.UniqueConstraint("unit_id", "id", name="uq_work_pattern_day_unit_id"),
            db.CheckConstraint("day_index >= 0", name="ck_work_pattern_day_index_nonnegative"),
            db.CheckConstraint(
                "day_type IN ('FIXED_SHIFT','WORK_ANY','WORK_ALLOWED_SET','OFF',"
                "'OPTIONAL_WORK','PROTECTED_NON_OPERATIONAL')",
                name="ck_work_pattern_day_type",
            ),
            db.CheckConstraint(
                "(day_type = 'FIXED_SHIFT' AND fixed_shift_type_id IS NOT NULL) OR "
                "(day_type <> 'FIXED_SHIFT' AND fixed_shift_type_id IS NULL)",
                name="ck_work_pattern_day_fixed_shift",
            ),
        )

    class WorkPatternDayAllowedShift(db.Model):
        __tablename__ = "work_pattern_day_allowed_shift"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        work_pattern_day_id = db.Column(db.Integer, nullable=False, index=True)
        shift_type_id = db.Column(db.Integer, nullable=False, index=True)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "work_pattern_day_id"],
                ["work_pattern_day.unit_id", "work_pattern_day.id"],
                name="fk_pattern_allowed_day_unit", ondelete="CASCADE",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "shift_type_id"],
                ["shift_type.unit_id", "shift_type.id"],
                name="fk_pattern_allowed_shift_unit",
            ),
            db.UniqueConstraint(
                "unit_id", "work_pattern_day_id", "shift_type_id",
                name="uq_pattern_day_allowed_shift",
            ),
        )

    class StaffPatternAssignment(db.Model):
        __tablename__ = "staff_pattern_assignment"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        staff_id = db.Column(db.Integer, nullable=False, index=True)
        work_pattern_id = db.Column(db.Integer, nullable=False, index=True)
        effective_from = db.Column(db.Date, nullable=False, index=True)
        effective_to = db.Column(db.Date, index=True)
        anchor_date = db.Column(db.Date, nullable=False)
        anchor_day_index = db.Column(db.Integer, nullable=False, default=0)
        contracted_minutes_override = db.Column(db.Integer)
        change_type = db.Column(db.String(40), nullable=False, default="WORK_PATTERN_CHANGE")
        contracted_minutes_per_week = db.Column(db.Integer)
        notes = db.Column(db.String(500), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "staff_id"], ["staff.unit_id", "staff.id"],
                name="fk_staff_pattern_person_unit", ondelete="CASCADE",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "work_pattern_id"],
                ["work_pattern.unit_id", "work_pattern.id"],
                name="fk_staff_pattern_pattern_unit",
            ),
            db.CheckConstraint(
                "effective_to IS NULL OR effective_to >= effective_from",
                name="ck_staff_pattern_effective_range",
            ),
            db.CheckConstraint("anchor_day_index >= 0", name="ck_staff_pattern_anchor_index"),
            db.CheckConstraint(
                "contracted_minutes_override IS NULL OR contracted_minutes_override >= 0",
                name="ck_staff_pattern_minutes_nonnegative",
            ),
            db.CheckConstraint(
                "change_type IN ('WORK_PATTERN_CHANGE','PART_TIME_CHANGE','FULL_TIME_CHANGE')",
                name="ck_staff_pattern_change_type",
            ),
            db.CheckConstraint(
                "contracted_minutes_per_week IS NULL OR contracted_minutes_per_week >= 0",
                name="ck_staff_pattern_weekly_minutes_nonnegative",
            ),
        )

    class StaffRule(db.Model):
        __tablename__ = "staff_rule"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        staff_id = db.Column(db.Integer, nullable=False, index=True)
        rule_type = db.Column(db.String(40), nullable=False, index=True)
        hardness = db.Column(db.String(8), nullable=False)
        effective_from = db.Column(db.Date, nullable=False, index=True)
        effective_to = db.Column(db.Date, index=True)
        shift_type_id = db.Column(db.Integer)
        shift_group = db.Column(db.String(20))
        maximum_count = db.Column(db.Integer)
        rolling_period_days = db.Column(db.Integer)
        weekdays_mask = db.Column(db.Integer)
        penalty_weight = db.Column(db.Integer, nullable=False, default=1)
        reason = db.Column(db.String(500), nullable=False, default="")
        authorised_by_user_id = db.Column(db.Integer)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "staff_id"], ["staff.unit_id", "staff.id"],
                name="fk_staff_rule_person_unit", ondelete="CASCADE",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "shift_type_id"],
                ["shift_type.unit_id", "shift_type.id"],
                name="fk_staff_rule_shift_unit",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "authorised_by_user_id"],
                ["staff.unit_id", "staff.id"],
                name="fk_staff_rule_authoriser_unit",
            ),
            db.CheckConstraint("hardness IN ('HARD','SOFT')", name="ck_staff_rule_hardness"),
            db.CheckConstraint(
                "rule_type IN ('NO_NIGHT','AVOID_NIGHT','NO_EARLY','AVOID_EARLY',"
                "'ALLOWED_SHIFT','DISALLOWED_SHIFT','MAX_NIGHTS_PER_CYCLE',"
                "'MAX_SHIFTS_PER_CYCLE','AVAILABLE_WEEKDAYS','UNAVAILABLE_WEEKDAYS',"
                "'MAX_CONTRACTED_MINUTES','PREFERRED_SHIFT','PREFERRED_DAY_OFF')",
                name="ck_staff_rule_type",
            ),
            db.CheckConstraint(
                "effective_to IS NULL OR effective_to >= effective_from",
                name="ck_staff_rule_effective_range",
            ),
            db.CheckConstraint(
                "maximum_count IS NULL OR maximum_count >= 0",
                name="ck_staff_rule_maximum_nonnegative",
            ),
            db.CheckConstraint(
                "rolling_period_days IS NULL OR rolling_period_days > 0",
                name="ck_staff_rule_period_positive",
            ),
            db.CheckConstraint(
                "weekdays_mask IS NULL OR (weekdays_mask >= 0 AND weekdays_mask <= 127)",
                name="ck_staff_rule_weekdays_mask",
            ),
            db.CheckConstraint("penalty_weight >= 0", name="ck_staff_rule_penalty"),
        )

    class BankHoliday(db.Model):
        __tablename__ = "bank_holiday"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        day = db.Column(db.Date, nullable=False, index=True)
        name = db.Column(db.String(120), nullable=False)
        is_active = db.Column(db.Boolean, nullable=False, default=True)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "day", name="uq_bank_holiday_unit_day"),
        )

    class RosterProposal(db.Model):
        __tablename__ = "roster_proposal"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        start_date = db.Column(db.Date, nullable=False, index=True)
        end_date = db.Column(db.Date, nullable=False, index=True)
        status = db.Column(db.String(32), nullable=False)
        workflow_state = db.Column(db.String(20), nullable=False, default="draft")
        objective_score = db.Column(db.BigInteger, nullable=False, default=0)
        configuration_json = db.Column(db.Text, nullable=False, default="{}")
        warnings_json = db.Column(db.Text, nullable=False, default="[]")
        uncovered_json = db.Column(db.Text, nullable=False, default="[]")
        created_by_user_id = db.Column(db.Integer, nullable=False)
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        applied_by_user_id = db.Column(db.Integer)
        applied_at = db.Column(db.DateTime)
        discarded_by_user_id = db.Column(db.Integer)
        discarded_at = db.Column(db.DateTime)
        __table_args__ = (
            db.CheckConstraint(
                "end_date >= start_date", name="ck_roster_proposal_date_range"
            ),
            db.CheckConstraint(
                "workflow_state IN ('draft','applied','discarded')",
                name="ck_roster_proposal_workflow_state",
            ),
            db.UniqueConstraint("unit_id", "id", name="uq_roster_proposal_unit_id"),
        )

    class RosterProposalAssignment(db.Model):
        __tablename__ = "roster_proposal_assignment"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        proposal_id = db.Column(db.Integer, nullable=False, index=True)
        staff_id = db.Column(db.Integer, nullable=False, index=True)
        day = db.Column(db.Date, nullable=False, index=True)
        shift_type_id = db.Column(db.Integer, nullable=False)
        shift_code = db.Column(db.String(10), nullable=False)
        review_state = db.Column(db.String(20), nullable=False, default="pending")
        score = db.Column(db.BigInteger, nullable=False, default=0)
        explanations_json = db.Column(db.Text, nullable=False, default="[]")
        applied_assignment_id = db.Column(db.Integer)
        reviewed_by_user_id = db.Column(db.Integer)
        reviewed_at = db.Column(db.DateTime)
        __table_args__ = (
            db.ForeignKeyConstraint(
                ["unit_id", "proposal_id"],
                ["roster_proposal.unit_id", "roster_proposal.id"],
                name="fk_proposal_assignment_proposal_unit",
                ondelete="CASCADE",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "staff_id"],
                ["staff.unit_id", "staff.id"],
                name="fk_proposal_assignment_staff_unit",
            ),
            db.ForeignKeyConstraint(
                ["unit_id", "shift_type_id"],
                ["shift_type.unit_id", "shift_type.id"],
                name="fk_proposal_assignment_shift_unit",
            ),
            db.CheckConstraint(
                "review_state IN ('pending','accepted','rejected','applied')",
                name="ck_proposal_assignment_review_state",
            ),
            db.UniqueConstraint(
                "unit_id", "proposal_id", "staff_id", "day",
                name="uq_proposal_assignment_staff_day",
            ),
        )

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

    class RosterPeriod(db.Model):
        __tablename__ = "roster_period"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        year = db.Column(db.Integer, nullable=False)
        month = db.Column(db.Integer, nullable=False)
        status = db.Column(db.String(30), nullable=False, index=True)
        generated_at = db.Column(db.DateTime)
        generated_by_user_id = db.Column(db.Integer)
        generation_method = db.Column(db.String(40), nullable=False, default="AUTOMATIC")
        generation_version = db.Column(db.String(40), nullable=False, default="")
        notes = db.Column(db.String(1000), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "year", "month", name="uq_roster_period_unit_month"),
            db.CheckConstraint("month BETWEEN 1 AND 12", name="ck_roster_period_month"),
            db.CheckConstraint(
                "status IN ('CURRENT','PROTECTED','FUTURE_AUTOMATIC','HISTORICAL','CLOSED')",
                name="ck_roster_period_status",
            ),
        )

    class RosterImpactEvent(db.Model):
        __tablename__ = "roster_impact_event"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, db.ForeignKey("unit.id"), nullable=False, index=True)
        event_type = db.Column(db.String(50), nullable=False, index=True)
        effective_from = db.Column(db.Date, nullable=False, index=True)
        effective_to = db.Column(db.Date)
        staff_ids_json = db.Column(db.Text, nullable=False, default="[]")
        watch_ids_json = db.Column(db.Text, nullable=False, default="[]")
        rebuild_baseline = db.Column(db.Boolean, nullable=False, default=False)
        recalculate_coverage = db.Column(db.Boolean, nullable=False, default=True)
        preserve_overrides = db.Column(db.Boolean, nullable=False, default=True)
        reason = db.Column(db.String(500), nullable=False, default="")
        triggered_by_user_id = db.Column(db.Integer)
        status = db.Column(db.String(30), nullable=False, default="PENDING", index=True)
        protected_from = db.Column(db.Date)
        protected_to = db.Column(db.Date)
        automatic_from = db.Column(db.Date)
        automatic_to = db.Column(db.Date)
        result_json = db.Column(db.Text, nullable=False, default="{}")
        started_at = db.Column(db.DateTime)
        affected_dates = db.Column(db.Integer, nullable=False, default=0)
        assignments_created = db.Column(db.Integer, nullable=False, default=0)
        baselines_changed = db.Column(db.Integer, nullable=False, default=0)
        baselines_removed = db.Column(db.Integer, nullable=False, default=0)
        overrides_retained = db.Column(db.Integer, nullable=False, default=0)
        redundant_overrides_found = db.Column(db.Integer, nullable=False, default=0)
        exceptions_created = db.Column(db.Integer, nullable=False, default=0)
        warnings_created = db.Column(db.Integer, nullable=False, default=0)
        error_message = db.Column(db.String(2000), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        completed_at = db.Column(db.DateTime)
        __table_args__ = (
            db.UniqueConstraint("unit_id", "id", name="uq_roster_impact_event_unit_id"),
            db.CheckConstraint(
                "status IN ('PENDING','RUNNING','COMPLETED','COMPLETED_WITH_WARNINGS','FAILED')",
                name="ck_roster_impact_event_status",
            ),
            db.CheckConstraint("effective_to IS NULL OR effective_to >= effective_from", name="ck_roster_impact_event_range"),
        )

    class RosterImpactException(db.Model):
        __tablename__ = "roster_impact_exception"
        id = db.Column(db.Integer, primary_key=True)
        unit_id = db.Column(db.Integer, nullable=False, index=True)
        event_id = db.Column(db.Integer, nullable=False, index=True)
        staff_id = db.Column(db.Integer, index=True)
        watch_id = db.Column(db.Integer, db.ForeignKey("watch.id"), index=True)
        effective_from = db.Column(db.Date, nullable=False, index=True)
        effective_to = db.Column(db.Date, nullable=False)
        exception_type = db.Column(db.String(40), nullable=False, index=True)
        severity = db.Column(db.String(20), nullable=False, default="WARNING")
        description = db.Column(db.String(1000), nullable=False)
        status = db.Column(db.String(20), nullable=False, default="OPEN", index=True)
        resolved_by_user_id = db.Column(db.Integer)
        resolved_at = db.Column(db.DateTime)
        resolution_note = db.Column(db.String(1000), nullable=False, default="")
        created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
        __table_args__ = (
            db.ForeignKeyConstraint(["unit_id", "event_id"], ["roster_impact_event.unit_id", "roster_impact_event.id"], name="fk_roster_impact_exception_event_unit", ondelete="CASCADE"),
            db.ForeignKeyConstraint(["unit_id", "staff_id"], ["staff.unit_id", "staff.id"], name="fk_roster_impact_exception_staff_unit"),
            db.CheckConstraint("effective_to >= effective_from", name="ck_roster_impact_exception_range"),
            db.CheckConstraint("severity IN ('INFO','WARNING','CRITICAL')", name="ck_roster_impact_exception_severity"),
            db.CheckConstraint("status IN ('OPEN','ACKNOWLEDGED','RESOLVED','NOT_APPLICABLE')", name="ck_roster_impact_exception_status"),
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
