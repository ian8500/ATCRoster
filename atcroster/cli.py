"""Operational and platform maintenance CLI commands."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Callable

import click


@dataclass(frozen=True)
class CliDependencies:
    db: Any
    PlatformIdentity: Any
    PlatformMfaCredential: Any
    Unit: Any
    SignupWorkflow: Any
    SecureInvitation: Any
    DatabaseRoutingMetadata: Any
    Staff: Any
    UnitMembership: Any
    MfaCredential: Any
    now: Callable[[], Any]
    central_security_event: Callable[..., None]
    encrypt_field: Callable[[str], str]
    decrypt_field: Callable[[str], str]
    generate_password_hash: Callable[[str], str]
    operational_unit_context: Callable[[int, str], Any]


def create_cli_commands(dependencies: CliDependencies) -> tuple[Any, ...]:
    """Create CLI commands while leaving app assembly responsible for registration."""
    db = dependencies.db
    PlatformIdentity = dependencies.PlatformIdentity
    PlatformMfaCredential = dependencies.PlatformMfaCredential
    Unit = dependencies.Unit
    SignupWorkflow = dependencies.SignupWorkflow
    SecureInvitation = dependencies.SecureInvitation
    DatabaseRoutingMetadata = dependencies.DatabaseRoutingMetadata
    Staff = dependencies.Staff
    UnitMembership = dependencies.UnitMembership
    MfaCredential = dependencies.MfaCredential
    utcnow = dependencies.now
    _central_security_event = dependencies.central_security_event
    _encrypt_field = dependencies.encrypt_field
    _decrypt_field = dependencies.decrypt_field
    generate_password_hash = dependencies.generate_password_hash
    operational_unit_context = dependencies.operational_unit_context

    @click.command("bootstrap-platform")
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
                code="PLATFORM",
                name="ATCRoster Platform",
                status="platform_control",
                plan="internal",
                active_user_limit=5,
            )
            db.session.add(control)
            db.session.flush()
        password_hash = generate_password_hash(password)
        db.session.add(
            PlatformIdentity(
                public_id=f"platform-{secrets.token_hex(12)}",
                username=username,
                password_hash=password_hash,
            )
        )
        db.session.commit()
        click.echo(f"Platform Super Admin {username} created.")

    @click.command("reset-platform-mfa")
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
            "platform_mfa_reset",
            "success",
            identity.id,
            hashlib.sha256(normalized.encode()).hexdigest()[:16],
            "Re-enrolment required by trusted operator.",
        )
        db.session.commit()
        click.echo("Platform MFA reset; re-enrolment is required at next login.")

    @click.command("reconcile-signups")
    @click.option("--apply", "apply_changes", is_flag=True)
    @click.option(
        "--confirm",
        default="",
        help="Required with --apply: enter RECONCILE-INCOMPLETE-SIGNUPS",
    )
    def reconcile_signups(apply_changes, confirm):
        """Report or safely reconcile interrupted cross-database signups."""
        if apply_changes and confirm != "RECONCILE-INCOMPLETE-SIGNUPS":
            raise click.UsageError(
                "--apply requires --confirm RECONCILE-INCOMPLETE-SIGNUPS"
            )
        rows = (
            SignupWorkflow.query.filter(SignupWorkflow.state != "completed")
            .order_by(SignupWorkflow.id)
            .all()
        )
        for row in rows:
            invitation = db.session.get(SecureInvitation, row.invitation_id)
            routing = (
                db.session.get(DatabaseRoutingMetadata, invitation.unit_id)
                if invitation
                else None
            )
            click.echo(
                f"workflow={row.id} state={row.state} "
                f"error={row.last_error_code or 'none'}"
            )
            if not apply_changes or not invitation or not routing:
                continue
            if row.membership_id and row.operational_person_id:
                with operational_unit_context(invitation.unit_id, routing.secret_name):
                    staff = db.session.get(Staff, row.operational_person_id)
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
                        staff = db.session.get(Staff, row.operational_person_id)
                        if staff and invitation.target_person_id:
                            staff.membership_status = "active"
                            db.session.commit()
                        elif staff and staff.membership_status != "active":
                            db.session.delete(staff)
                            db.session.commit()
                    row.operational_person_id = None
                if row.identity_id:
                    identity = db.session.get(PlatformIdentity, row.identity_id)
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

    @click.command("rotate-field-encryption")
    @click.option(
        "--confirm",
        default="",
        help="Required: enter ROTATE-FIELD-ENCRYPTION",
    )
    def rotate_field_encryption(confirm):
        """Re-encrypt MFA secrets with the first configured versioned key."""
        if confirm != "ROTATE-FIELD-ENCRYPTION":
            raise click.UsageError("--confirm ROTATE-FIELD-ENCRYPTION is required")
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

    return (
        bootstrap_platform,
        reset_platform_mfa,
        reconcile_signups,
        rotate_field_encryption,
    )
