import re

import app
from app import (
    PlatformIdentity,
    Staff,
    Unit,
    UnitMembership,
    db,
)


def _csrf(client, path):
    response = client.get(path)
    assert response.status_code == 200
    with client.session_transaction() as session:
        return session["_csrf_token"]


def _login(client, username, password):
    response = client.post(
        "/login", data={"username": username, "password": password}
    )
    assert response.status_code == 302


def test_super_admin_provisions_airport_and_account_limit_is_transactional():
    with app.app.app_context():
        db.drop_all()
        db.create_all()
        control = Unit(
            code="CTRL", name="Platform Control", status="platform_control",
            active_user_limit=1,
        )
        db.session.add(control)
        db.session.flush()
        platform_user = Staff(
            unit_id=control.id, username="platform.admin",
            name="Platform Administrator", staff_no="CTRL-1",
            role="superadmin", is_operational=False,
        )
        platform_user.set_password("Platform-Test-2026!")
        db.session.add(platform_user)
        db.session.flush()
        db.session.add(PlatformIdentity(
            public_id="platform-admin-test",
            username=platform_user.username,
            password_hash=platform_user.password_hash,
        ))
        db.session.commit()
        app.migrate_add_role_and_calendar_token()
        assert db.session.get(Staff, platform_user.id).role == "superadmin"

    super_client = app.app.test_client()
    _login(super_client, "platform.admin", "Platform-Test-2026!")
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
    assert b"Test Airport created" in created.data
    # The control-plane listing does not display personnel details.
    assert b"tst.admin" not in created.data

    with app.app.app_context():
        unit = Unit.query.filter_by(code="TST").one()
        assert unit.active_user_limit == 2
        memberships = UnitMembership.query.filter_by(unit_id=unit.id).all()
        assert len(memberships) == 1
        assert memberships[0].status == "active"

    unit_client = app.app.test_client()
    _login(unit_client, "tst.admin", "UnitAdmin-Test-2026!")
    unit_token = _csrf(unit_client, "/unit/accounts")
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
        assert db.session.query(Staff).execution_options(
            skip_tenant_scope=True
        ).filter_by(username="tst.user3").first() is None
        unit = Unit.query.filter_by(code="TST").one()
        assert UnitMembership.query.filter_by(
            unit_id=unit.id, status="active"
        ).count() == 2

        unit.active_user_limit = 3
        db.session.commit()

    invitation = unit_client.post(
        "/unit/accounts",
        data={
            "_csrf_token": unit_token,
            "action": "create_invitation",
            "role": "ReadOnlyAuditor",
        },
        follow_redirects=True,
    )
    assert invitation.status_code == 200
    match = re.search(rb"/invite/([A-Za-z0-9_-]+)", invitation.data)
    assert match
    invitation_path = match.group(0).decode()
    invited_client = app.app.test_client()
    invite_token = _csrf(invited_client, invitation_path)
    accepted = invited_client.post(
        invitation_path,
        data={
            "_csrf_token": invite_token,
            "name": "Airport Auditor",
            "username": "tst.auditor",
            "password": "Auditor-Test-2026!",
        },
        follow_redirects=True,
    )
    assert accepted.status_code == 200
    assert b"Account created" in accepted.data
    _login(invited_client, "tst.auditor", "Auditor-Test-2026!")
    with app.app.app_context():
        unit = Unit.query.filter_by(code="TST").one()
        assert UnitMembership.query.filter_by(
            unit_id=unit.id, status="active",
            role="ReadOnlyAuditor",
        ).count() == 1
