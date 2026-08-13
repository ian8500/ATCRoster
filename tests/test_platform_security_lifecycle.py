import hashlib

import pyotp
import pytest
from sqlalchemy import create_engine, inspect

import app
from app import (
    DatabaseRoutingMetadata,
    PlatformIdentity,
    PlatformMfaCredential,
    SecureInvitation,
    SignupWorkflow,
    Staff,
    Unit,
    UnitMembership,
    db,
)
from scripts.migrate_all_databases import upgrade_database
from tenancy import dispose_operational_engines, operational_unit_context


def _reset():
    dispose_operational_engines()
    with app.app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()


def _platform_account():
    control = Unit(
        code="CTRL", name="Platform Control",
        status="platform_control", active_user_limit=2,
    )
    db.session.add(control)
    db.session.flush()
    user = Staff(
        unit_id=control.id, username="security.platform",
        name="Platform Security", staff_no="CTRL-SEC",
        role="superadmin", is_operational=False,
    )
    user.set_password("Platform-Security-2026!")
    db.session.add(user)
    db.session.flush()
    identity = PlatformIdentity(
        public_id="platform-security-test",
        username=user.username, password_hash=user.password_hash,
    )
    db.session.add(identity)
    db.session.commit()
    return user, identity


def _csrf(client):
    with client.session_transaction() as session:
        return session["_csrf_token"]


def test_superadmin_requires_central_mfa_before_platform_access():
    _reset()
    with app.app.app_context():
        _platform_account()
    client = app.app.test_client()
    client.get("/login")
    with client.session_transaction() as session:
        login_token = session["_csrf_token"]
    password = client.post("/login", data={
        "_csrf_token": login_token,
        "username": "security.platform",
        "password": "Platform-Security-2026!",
    })
    assert password.status_code == 302
    blocked = client.get("/platform/admin")
    assert blocked.status_code == 302
    assert "/login" in blocked.headers["Location"]
    setup = client.get("/login/platform-mfa/setup")
    assert setup.status_code == 200
    with client.session_transaction() as session:
        secret = session["_pending_platform_mfa_secret"]
        token = session["_csrf_token"]
    enabled = client.post("/login/platform-mfa/setup", data={
        "_csrf_token": token, "code": pyotp.TOTP(secret).now(),
    })
    assert enabled.status_code == 302
    with app.app.app_context():
        credential = PlatformMfaCredential.query.one()
        assert credential.enabled
        assert secret not in credential.encrypted_secret
        assert credential.recovery_codes_digest == "[]"
    client.get("/login/platform-mfa")
    verified = client.post("/login/platform-mfa", data={
        "_csrf_token": _csrf(client),
        "code": pyotp.TOTP(secret).now(),
    })
    assert verified.status_code == 302
    assert client.get("/platform/admin").status_code == 200


def test_provisioning_failure_is_safe_and_retryable(tmp_path, monkeypatch):
    monkeypatch.setenv("ATCROSTER_DISABLE_LOCAL_AUTO_PROVISION", "1")
    _reset()
    with app.app.app_context():
        _user, identity = _platform_account()
        unit = Unit(
            code="PRV", name="Provisioning Test",
            status="provisioning", active_user_limit=3,
        )
        db.session.add(unit)
        db.session.flush()
        route = DatabaseRoutingMetadata(
            unit_id=unit.id,
            secret_name=f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL",
            provisioning_state="pending",
        )
        db.session.add(route)
        db.session.commit()
        identity_id = identity.id
        unit_id, secret_name = unit.id, route.secret_name
    monkeypatch.delenv(secret_name, raising=False)
    client = app.app.test_client()
    with client.session_transaction() as session:
        session["_user_id"] = f"platform-identity:{identity_id}"
        session["_fresh"] = True
        session["_platform_mfa_verified"] = True
    client.get("/platform/admin")
    failed = client.post("/platform/admin", data={
        "_csrf_token": _csrf(client),
        "action": "provision_unit", "unit_id": str(unit_id),
    })
    assert failed.status_code == 302
    from platform_provisioning import ProvisioningWorker
    worker = ProvisioningWorker(app.app)
    assert worker.run_once()
    with app.app.app_context():
        route = db.session.get(DatabaseRoutingMetadata, unit_id)
        assert route.provisioning_state == "retry_wait"
        assert route.last_error_code == "database_secret_unavailable"
        assert SecureInvitation.query.filter_by(unit_id=unit_id).count() == 0
        job = app.ProvisioningJob.query.filter_by(unit_id=unit_id).one()
        job.next_attempt_at = app.utcnow()
        db.session.commit()
    operational_url = f"sqlite:///{tmp_path / 'provisioned.db'}"
    monkeypatch.setenv(secret_name, operational_url)
    import platform_provisioning
    real_upgrade = platform_provisioning.upgrade_database
    monkeypatch.setattr(
        platform_provisioning, "upgrade_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic migration failure")
        ),
    )
    client.get("/platform/admin")
    migration_failed = client.post("/platform/admin", data={
        "_csrf_token": _csrf(client),
        "action": "provision_unit", "unit_id": str(unit_id),
    })
    assert migration_failed.status_code == 302
    assert worker.run_once()
    with app.app.app_context():
        route = db.session.get(DatabaseRoutingMetadata, unit_id)
        assert route.provisioning_state == "retry_wait"
        assert route.last_error_code == "database_provisioning_failed"
        assert SecureInvitation.query.filter_by(unit_id=unit_id).count() == 0
        job = app.ProvisioningJob.query.filter_by(unit_id=unit_id).one()
        job.next_attempt_at = app.utcnow()
        db.session.commit()
    monkeypatch.setattr(
        platform_provisioning, "upgrade_database", real_upgrade
    )
    client.get("/platform/admin")
    retried = client.post("/platform/admin", data={
        "_csrf_token": _csrf(client),
        "action": "provision_unit", "unit_id": str(unit_id),
    })
    assert retried.status_code == 302
    assert worker.run_once()
    with app.app.app_context():
        route = db.session.get(DatabaseRoutingMetadata, unit_id)
        assert route.provisioning_state == "invitation_issued"
        assert route.attempt_count == 3
        assert SecureInvitation.query.filter_by(unit_id=unit_id).count() == 1
    engine = create_engine(operational_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert "staff" in tables
        assert "platform_identity" not in tables
        assert "unit" not in tables
    finally:
        engine.dispose()


@pytest.mark.parametrize("stage", [
    "identity_created",
    "operational_account_created",
    "membership_created",
])
def test_signup_saga_recovers_idempotently_after_each_stage(
    stage, tmp_path, monkeypatch
):
    _reset()
    operational_url = f"sqlite:///{tmp_path / f'{stage}.db'}"
    upgrade_database(operational_url, "operational")
    with app.app.app_context():
        unit = Unit(
            code="SGA", name="Saga Airport", status="active",
            active_user_limit=5,
        )
        db.session.add(unit)
        db.session.flush()
        secret_name = f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL"
        monkeypatch.setenv(secret_name, operational_url)
        route = DatabaseRoutingMetadata(
            unit_id=unit.id, secret_name=secret_name,
            provisioning_state="invitation_issued",
        )
        invitation = SecureInvitation(
            unit_id=unit.id,
            token_digest=hashlib.sha256(stage.encode()).hexdigest(),
            role="StaffUser",
            expires_at=app.utcnow() + app.timedelta(days=1),
        )
        db.session.add_all([route, invitation])
        db.session.commit()
        with operational_unit_context(unit.id, secret_name):
            with pytest.raises(app.SignupWorkflowError):
                app._run_invitation_signup(
                    invitation, unit, "Saga Person",
                    f"{stage}-test".replace("_", "-"),
                    "Saga-Password-2026!", fail_after=stage,
                )
            workflow = SignupWorkflow.query.filter_by(
                invitation_id=invitation.id
            ).one()
            assert workflow.state == "failed"
            app._run_invitation_signup(
                invitation, unit, "Saga Person",
                f"{stage}-test".replace("_", "-"),
                "Saga-Password-2026!",
            )
            assert Staff.query.count() == 1
        workflow = SignupWorkflow.query.filter_by(
            invitation_id=invitation.id
        ).one()
        assert workflow.state == "completed"
        assert UnitMembership.query.filter_by(unit_id=unit.id).count() == 1
        assert db.session.get(SecureInvitation, invitation.id).accepted_at


def test_global_identity_duplicate_creates_no_operational_staff(
    tmp_path, monkeypatch
):
    _reset()
    operational_url = f"sqlite:///{tmp_path / 'duplicate.db'}"
    upgrade_database(operational_url, "operational")
    with app.app.app_context():
        unit = Unit(
            code="DUP", name="Duplicate Test", status="active",
            active_user_limit=2,
        )
        db.session.add(unit)
        db.session.flush()
        secret_name = f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL"
        monkeypatch.setenv(secret_name, operational_url)
        db.session.add_all([
            DatabaseRoutingMetadata(
                unit_id=unit.id, secret_name=secret_name,
                provisioning_state="invitation_issued",
            ),
            PlatformIdentity(
                public_id="existing-global",
                username="global.user", password_hash="not-used",
            ),
        ])
        invitation = SecureInvitation(
            unit_id=unit.id,
            token_digest=hashlib.sha256(b"duplicate").hexdigest(),
            role="StaffUser",
            expires_at=app.utcnow() + app.timedelta(days=1),
        )
        db.session.add(invitation)
        db.session.commit()
        with operational_unit_context(unit.id, secret_name):
            with pytest.raises(
                app.SignupWorkflowError,
                match="login identifier is unavailable",
            ):
                app._run_invitation_signup(
                    invitation, unit, "Duplicate Person",
                    "GLOBAL.USER", "Duplicate-Password-2026!",
                )
            assert Staff.query.count() == 0
            workflow = SignupWorkflow.query.filter_by(
                invitation_id=invitation.id
            ).one()
            assert workflow.state == "failed"
            assert workflow.compensation_state == "pending"
            assert workflow.identity_id is None

            app._run_invitation_signup(
                invitation,
                unit,
                "Duplicate Person",
                "available.user",
                "Duplicate-Password-2026!",
            )
            assert Staff.query.filter_by(
                username="available.user"
            ).count() == 1
        workflow = SignupWorkflow.query.filter_by(
            invitation_id=invitation.id
        ).one()
        assert workflow.state == "completed"
        assert workflow.normalized_username == "available.user"
        assert db.session.get(SecureInvitation, invitation.id).accepted_at
