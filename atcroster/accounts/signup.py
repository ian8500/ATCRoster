"""Cross-database invitation signup saga."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.exc import IntegrityError


class SignupWorkflowError(RuntimeError):
    """A safe, user-presentable signup workflow failure."""


@dataclass(frozen=True)
class SignupSagaDependencies:
    db: Any
    ShiftType: Any
    SignupWorkflow: Any
    PlatformIdentity: Any
    Staff: Any
    UnitMembership: Any
    Unit: Any
    SecureInvitation: Any
    DatabaseRoutingMetadata: Any
    now: Callable[[], Any]
    valid_email: Callable[[str], str]
    password_hash: Callable[[str], str]


def normalized_login(value: str) -> str:
    return value.strip().casefold()


def run_invitation_signup(
    dependencies: SignupSagaDependencies,
    invitation: Any,
    unit: Any,
    name: str,
    username: str,
    password: str,
    email: str = "",
    fail_after: str | None = None,
):
    db = dependencies.db
    ShiftType = dependencies.ShiftType
    SignupWorkflow = dependencies.SignupWorkflow
    PlatformIdentity = dependencies.PlatformIdentity
    Staff = dependencies.Staff
    UnitMembership = dependencies.UnitMembership
    Unit = dependencies.Unit
    SecureInvitation = dependencies.SecureInvitation
    DatabaseRoutingMetadata = dependencies.DatabaseRoutingMetadata
    utcnow = dependencies.now
    _valid_email = dependencies.valid_email
    generate_password_hash = dependencies.password_hash
    """Resume an invitation saga without claiming cross-DB atomicity."""
    # Every airport begins with a dependable rest-day code, even when it was
    # provisioned after the original application seed ran.
    off_shift = ShiftType.query.filter_by(unit_id=unit.id, code="OFF").first()
    if not off_shift:
        db.session.add(
            ShiftType(
                unit_id=unit.id,
                code="OFF",
                name="Day off",
                is_working=False,
                is_active=True,
                is_requestable=False,
            )
        )
        db.session.commit()
    normalized = normalized_login(username)
    workflow = SignupWorkflow.query.filter_by(invitation_id=invitation.id).first()
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
                raise SignupWorkflowError("That login identifier is unavailable.")
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
                "UnitAdmin": "admin",
                "RosterEditor": "editor",
                "WatchManager": "user",
                "StaffUser": "user",
                "ReadOnlyAuditor": "auditor",
                "PositionMonitor": "position_monitor",
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
                    raise SignupWorkflowError("That login identifier is unavailable.")
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
                    raise SignupWorkflowError("That login identifier is unavailable.")
                staff = Staff(
                    unit_id=unit.id,
                    username=normalized,
                    name=name[:80],
                    staff_no=marker,
                    role=role_map[invitation.role],
                    is_wm=invitation.role == "WatchManager",
                    is_operational=False,
                    membership_status="pending",
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
                    identity_id=workflow.identity_id,
                    unit_id=unit.id,
                    person_id=workflow.operational_person_id,
                    role=invitation.role,
                    status="invited",
                )
                db.session.add(membership)
                db.session.flush()
                from account_limits import activate_membership

                activate_membership(db, Unit, UnitMembership, membership.id)
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
            staff = db.session.get(Staff, workflow.operational_person_id)
            if not staff:
                raise SignupWorkflowError(
                    "Operational account requires reconciliation."
                )
            staff.membership_status = "active"
            db.session.commit()
            workflow = db.session.get(SignupWorkflow, workflow.id)
            invitation = db.session.get(SecureInvitation, workflow.invitation_id)
            invitation.accepted_at = utcnow()
            invitation.active_bootstrap_key = None
            if invitation.role == "UnitAdmin":
                unit.status = "active"
                routing = db.session.get(DatabaseRoutingMetadata, unit.id)
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
            str(exc) if str(exc).startswith("injected_") else "stage_interrupted"
        )
        workflow.updated_at = utcnow()
        db.session.commit()
        raise SignupWorkflowError(
            "Account setup was interrupted safely. Retry this invitation "
            "or ask an administrator to reconcile it."
        ) from exc
