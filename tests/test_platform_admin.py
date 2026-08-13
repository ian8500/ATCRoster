import re

import pyotp

import app
from tenancy import operational_unit_context
from app import (
    PlatformIdentity,
    Staff,
    Unit,
    UnitMembership,
    Watch,
    db,
)


def _csrf(client, path):
    response = client.get(path)
    assert response.status_code == 200
    with client.session_transaction() as session:
        return session["_csrf_token"]


def _login(client, username, password):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    response = client.post("/login", data={
        "_csrf_token": token, "username": username, "password": password,
    })
    assert response.status_code == 302


def _login_platform_with_mfa(client, username, password):
    _login(client, username, password)
    setup = client.get("/login/platform-mfa/setup")
    assert setup.status_code == 200
    with client.session_transaction() as session:
        secret = session["_pending_platform_mfa_secret"]
        token = session["_csrf_token"]
    enrolled = client.post(
        "/login/platform-mfa/setup",
        data={"_csrf_token": token, "code": pyotp.TOTP(secret).now()},
    )
    assert enrolled.status_code == 302
    challenge = client.get("/login/platform-mfa")
    assert challenge.status_code == 200
    with client.session_transaction() as session:
        token = session["_csrf_token"]
    verified = client.post(
        "/login/platform-mfa",
        data={"_csrf_token": token, "code": pyotp.TOTP(secret).now()},
    )
    assert verified.status_code == 302


def test_super_admin_provisions_airport_and_account_limit_is_transactional(
    tmp_path, monkeypatch
):
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        control = Unit(
            code="CTRL",
            name="Platform Control",
            status="platform_control",
            active_user_limit=1,
        )
        db.session.add(control)
        db.session.flush()
        platform_user = Staff(
            unit_id=control.id,
            username="platform.admin",
            name="Platform Administrator",
            staff_no="CTRL-1",
            role="superadmin",
            is_operational=False,
        )
        platform_user.set_password("Platform-Test-2026!")
        db.session.add(platform_user)
        db.session.flush()
        db.session.add(
            PlatformIdentity(
                public_id="platform-admin-test",
                username=platform_user.username,
                password_hash=platform_user.password_hash,
            )
        )
        db.session.commit()
        app.migrate_add_role_and_calendar_token()
        assert db.session.get(Staff, platform_user.id).role == "superadmin"

    super_client = app.app.test_client()
    _login_platform_with_mfa(super_client, "platform.admin", "Platform-Test-2026!")
    token = _csrf(super_client, "/platform/admin")
    created = super_client.post(
        "/platform/admin",
        data={
            "_csrf_token": token,
            "action": "create_unit",
            "code": "TST",
            "name": "Test Airport",
            "plan": "starter",
            "active_user_limit": "2",
            "admin_name": "Initial Unit Admin",
            "admin_username": "tst.admin",
            "admin_password": "UnitAdmin-Test-2026!",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    assert b"Test Airport metadata created" in created.data
    assert b"Training" in created.data
    assert b"Competency" in created.data
    assert b"Advanced coverage" not in created.data
    assert b"Scenario planning" not in created.data
    assert b"Calendar exports" not in created.data
    assert b'name="key" value="training_module"' in created.data
    assert b'name="key" value="competency_module"' in created.data
    # The control-plane listing does not display personnel details.
    assert b"tst.admin" not in created.data
    with app.app.app_context():
        unit = Unit.query.filter_by(code="TST").one()
        secret_name = f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL"
    monkeypatch.setenv(secret_name, f"sqlite:///{tmp_path / 'tst-operational.db'}")
    provision_token = _csrf(super_client, "/platform/admin")
    provisioned = super_client.post(
        "/platform/admin",
        data={
            "_csrf_token": provision_token,
            "action": "provision_unit",
            "unit_id": str(unit.id),
        },
        follow_redirects=True,
    )
    assert provisioned.status_code == 200
    from platform_provisioning import ProvisioningWorker

    assert ProvisioningWorker(app.app).run_once()
    with app.app.app_context():
        job = app.ProvisioningJob.query.filter_by(unit_id=unit.id).one()
        job_id = job.id
        assert job.state == "completed"
    revealed = super_client.post(
        "/platform/admin",
        data={
            "_csrf_token": _csrf(super_client, "/platform/admin"),
            "action": "reveal_bootstrap",
            "job_id": str(job_id),
        },
        follow_redirects=True,
    )
    bootstrap_match = re.search(rb"/invite/([A-Za-z0-9_-]+)", revealed.data)
    assert bootstrap_match
    bootstrap_path = bootstrap_match.group(0).decode()

    with app.app.app_context():
        unit = Unit.query.filter_by(code="TST").one()
        assert unit.active_user_limit == 2
        memberships = UnitMembership.query.filter_by(unit_id=unit.id).all()
        assert memberships == []

    unit_client = app.app.test_client()
    bootstrap_csrf = _csrf(unit_client, bootstrap_path)
    accepted_bootstrap = unit_client.post(
        bootstrap_path,
        data={
            "_csrf_token": bootstrap_csrf,
            "name": "Initial Unit Admin",
            "username": "tst.admin",
            "email": "tst.admin@example.test",
            "password": "UnitAdmin-Test-2026!",
        },
    )
    assert accepted_bootstrap.status_code == 302
    _login(unit_client, "tst.admin", "UnitAdmin-Test-2026!")
    setup = unit_client.get("/security/mfa")
    assert setup.status_code == 200
    assert b"data:image/svg+xml;base64," in setup.data
    assert b"Scan QR code" in setup.data
    with unit_client.session_transaction() as session:
        secret = session["_pending_mfa_secret"]
        mfa_csrf = session["_csrf_token"]
    enrolled = unit_client.post(
        "/security/mfa",
        data={
            "_csrf_token": mfa_csrf,
            "code": pyotp.TOTP(secret).now(),
        },
        follow_redirects=True,
    )
    assert enrolled.status_code == 200
    assert b"Save your recovery codes" not in enrolled.data
    unit_token = _csrf(unit_client, "/unit/accounts")
    with app.app.app_context():
        db.session.add(
            PlatformIdentity(
                public_id="other-airport-duplicate",
                username="tst.duplicate",
                password_hash="not-a-login-secret",
            )
        )
        db.session.commit()
    duplicate = unit_client.post(
        "/unit/accounts",
        data={
            "_csrf_token": unit_token,
            "action": "create_account",
            "name": "Must Not Be Created",
            "username": "TST.DUPLICATE",
            "password": "Duplicate-Test-2026!",
        },
        follow_redirects=True,
    )
    assert b"login identifier is unavailable" in duplicate.data
    with app.app.app_context():
        assert (
            UnitMembership.query.join(PlatformIdentity)
            .filter(PlatformIdentity.username == "tst.duplicate")
            .count()
            == 0
        )
    second = unit_client.post(
        "/unit/accounts",
        data={
            "_csrf_token": unit_token,
            "action": "create_account",
            "name": "Second Account",
            "username": "tst.user2",
            "password": "Second-Test-2026!",
        },
        follow_redirects=True,
    )
    assert b"Account activated" in second.data
    assert b"2 of 2" in second.data

    redundant_bootstrap = super_client.post(
        "/platform/admin",
        data={
            "_csrf_token": _csrf(super_client, "/platform/admin"),
            "action": "provision_unit",
            "unit_id": str(unit.id),
        },
        follow_redirects=True,
    )
    assert b"already has active accounts" in redundant_bootstrap.data
    assert b"Provision / retry" not in redundant_bootstrap.data

    blocked = unit_client.post(
        "/unit/accounts",
        data={
            "_csrf_token": unit_token,
            "action": "create_account",
            "name": "Blocked Account",
            "username": "tst.user3",
            "password": "Blocked-Test-2026!",
        },
        follow_redirects=True,
    )
    assert b"Active account limit reached" in blocked.data
    with app.app.app_context():
        assert (
            db.session.query(Staff)
            .execution_options(skip_tenant_scope=True)
            .filter_by(username="tst.user3")
            .first()
            is None
        )
        unit = Unit.query.filter_by(code="TST").one()
        assert (
            UnitMembership.query.filter_by(unit_id=unit.id, status="active").count()
            == 2
        )

        unit.active_user_limit = 3
        with operational_unit_context(unit.id, secret_name):
            watch = Watch.query.filter_by(unit_id=unit.id).first()
            roster_person = Staff(
                unit_id=unit.id,
                username="person-before-access",
                name="Airport Auditor",
                staff_no="AUD-001",
                role="user",
                watch_id=watch.id if watch else None,
                is_operational=True,
                pattern_override=True,
                pattern_csv="D,OFF",
            )
            roster_person.set_password("No-Login-Placeholder-2026!")
            db.session.add(roster_person)
            db.session.commit()
            roster_person_id = roster_person.id

    invitation = unit_client.post(
        "/unit/accounts",
        data={
            "_csrf_token": unit_token,
            "action": "create_invitation",
            "role": "ReadOnlyAuditor",
            "person_id": str(roster_person_id),
        },
        follow_redirects=True,
    )
    assert invitation.status_code == 200
    match = re.search(rb"/invite/([A-Za-z0-9_-]+)", invitation.data)
    assert match, invitation.get_data(as_text=True)
    invitation_path = match.group(0).decode()
    invited_client = app.app.test_client()
    invite_token = _csrf(invited_client, invitation_path)
    accepted = invited_client.post(
        invitation_path,
        data={
            "_csrf_token": invite_token,
            "username": "tst.auditor",
            "email": "tst.auditor@example.test",
            "password": "Auditor-Test-2026!",
        },
        follow_redirects=True,
    )
    assert accepted.status_code == 200
    assert b"Account created" in accepted.data
    _login(invited_client, "tst.auditor", "Auditor-Test-2026!")
    with app.app.app_context():
        unit = Unit.query.filter_by(code="TST").one()
        membership = UnitMembership.query.filter_by(
            unit_id=unit.id,
            status="active",
            role="ReadOnlyAuditor",
            person_id=roster_person_id,
        ).one()
        assert membership.person_id == roster_person_id
        with operational_unit_context(unit.id, secret_name):
            preserved = db.session.get(Staff, roster_person_id)
            assert preserved.staff_no == "AUD-001"
            assert preserved.pattern_csv == "D,OFF"
            assert preserved.is_operational


def test_platform_admin_onboarding_contains_no_personal_identity_fields():
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        control = Unit(
            code="CTRL",
            name="Platform Control",
            status="platform_control",
            active_user_limit=1,
        )
        db.session.add(control)
        db.session.flush()
        platform_user = Staff(
            unit_id=control.id,
            username="privacy.platform",
            name="Platform Operator",
            staff_no="CTRL-PRIV",
            role="superadmin",
            is_operational=False,
        )
        platform_user.set_password("Platform-Privacy-2026!")
        db.session.add(platform_user)
        db.session.flush()
        db.session.add(
            PlatformIdentity(
                public_id="platform-privacy-test",
                username=platform_user.username,
                password_hash=platform_user.password_hash,
            )
        )
        db.session.commit()
    client = app.app.test_client()
    _login_platform_with_mfa(client, "privacy.platform", "Platform-Privacy-2026!")
    page = client.get("/platform/admin")
    assert page.status_code == 200
    prohibited = (
        b"admin_name",
        b"admin_username",
        b"admin_password",
        b"email",
        b"phone",
        b"staff_no",
        b"impersonat",
    )
    lower = page.data.lower()
    for value in prohibited:
        assert value not in lower

    created = client.post(
        "/platform/admin",
        data={
            "_csrf_token": _csrf(client, "/platform/admin"),
            "action": "create_unit",
            "code": "DEL",
            "name": "Deletion Test Airport",
            "plan": "starter",
            "active_user_limit": "2",
        },
        follow_redirects=True,
    )
    assert created.status_code == 200
    with app.app.app_context():
        unit_id = Unit.query.filter_by(code="DEL").one().id

    rejected = client.post(
        "/platform/admin",
        data={
            "_csrf_token": _csrf(client, "/platform/admin"),
            "action": "delete_unit",
            "unit_id": str(unit_id),
            "confirmation_code": "WRONG",
            "database_retained": "yes",
        },
        follow_redirects=True,
    )
    assert b"Type DEL exactly" in rejected.data
    with app.app.app_context():
        assert db.session.get(Unit, unit_id) is not None

    deleted = client.post(
        "/platform/admin",
        data={
            "_csrf_token": _csrf(client, "/platform/admin"),
            "action": "delete_unit",
            "unit_id": str(unit_id),
            "confirmation_code": "DEL",
            "database_retained": "yes",
        },
        follow_redirects=True,
    )
    assert deleted.status_code == 200
    assert b"DEL airport metadata deleted" in deleted.data
    with app.app.app_context():
        assert db.session.get(Unit, unit_id) is None
        audit = app.SuperAdminAudit.query.filter_by(action="airport_deleted").one()
        assert audit.unit_id is None
