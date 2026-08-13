"""Global test safety boundaries.

Tests must never inherit the developer's live DATABASE_URL. Real PostgreSQL
integration runs opt in through ATCROSTER_TEST_CONTROL_DATABASE_URL; every
other run receives a disposable, process-specific SQLite control database.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import tempfile


def finish_operational_login(client) -> None:
    """Complete the real airport MFA flow for test administrator accounts."""
    import pyotp

    import app

    with client.session_transaction() as session:
        challenge_user_id = int(session.get("_mfa_user_id") or 0)
        challenge_unit_id = int(session.get("_mfa_unit_id") or 0)
        authenticated = "_user_id" in session

    if challenge_user_id:
        with app.app.app_context():
            credential = app.MfaCredential.query.filter_by(
                person_id=challenge_user_id, enabled=True, reset_required=False,
            ).first()
            secret = app._decrypt_mfa_secret(credential) if credential else ""
            # Tests can sign the same account in repeatedly within one TOTP
            # interval. Reset replay tracking only inside this test helper.
            if credential:
                credential.last_used_step = None
                app.db.session.commit()
            with app.app.test_request_context(
                "/", environ_base={"REMOTE_ADDR": "127.0.0.1"}
            ):
                app._reset_rate_limit(
                        "airport-mfa",
                        f"{challenge_unit_id}:{challenge_user_id}",
                )
        if not credential:
            client.get("/security/mfa")
            with client.session_transaction() as session:
                secret = session["_pending_mfa_secret"]
                token = session["_csrf_token"]
            response = client.post(
                "/security/mfa",
                data={"_csrf_token": token, "code": pyotp.TOTP(secret).now()},
                follow_redirects=False,
            )
            assert response.status_code == 302
            return
        client.get("/login/mfa")
        with client.session_transaction() as session:
            token = session["_csrf_token"]
        response = client.post(
            "/login/mfa",
            data={"_csrf_token": token, "code": pyotp.TOTP(secret).now()},
            follow_redirects=False,
        )
        assert response.status_code == 302
        return

    if not authenticated:
        return

    client.get("/security/mfa")
    with client.session_transaction() as session:
        pending = session.get("_pending_mfa_secret")
        token = session.get("_csrf_token")
    if not pending:
        return
    with app.app.test_request_context(
        "/", environ_base={"REMOTE_ADDR": "127.0.0.1"}
    ):
        with client.session_transaction() as session:
            membership_id = int(str(session["_user_id"]).split(":")[-1])
        with app.app.app_context():
            membership = app.db.session.get(
                app.UnitMembership, membership_id
            )
            app._reset_rate_limit(
                "airport-mfa-enrolment",
                f"{membership.unit_id}:{membership.person_id}",
            )
    response = client.post(
        "/security/mfa",
        data={"_csrf_token": token, "code": pyotp.TOTP(pending).now()},
        follow_redirects=False,
    )
    assert response.status_code == 302


integration_url = os.environ.get("ATCROSTER_TEST_CONTROL_DATABASE_URL")
if integration_url:
    os.environ["DATABASE_URL"] = integration_url
    os.environ["CONTROL_DATABASE_URL"] = integration_url
else:
    test_database = (
        Path(tempfile.gettempdir()) / f"atcroster-pytest-control-{os.getpid()}.db"
    )
    test_url = f"sqlite:///{test_database}"
    os.environ["DATABASE_URL"] = test_url
    os.environ["CONTROL_DATABASE_URL"] = test_url

    def _remove_test_database() -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                test_database.with_name(test_database.name + suffix).unlink()
            except FileNotFoundError:
                pass

    atexit.register(_remove_test_database)
