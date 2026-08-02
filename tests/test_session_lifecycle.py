from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from flask import Flask
from flask_login import LoginManager, UserMixin, current_user, login_user

from atcroster.security.sessions import (
    SessionLifecycle,
    SessionLifecycleDependencies,
)


@dataclass
class Credential:
    enabled: bool = True
    reset_required: bool = False
    encrypted_secret: str = "encrypted-secret"
    enrolled_at: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)


class BrowserUser(UserMixin):
    id = 7
    password_hash = "password-hash"
    role = "admin"
    membership_status = "active"


def session_application(monkeypatch, *, idle_minutes="30", absolute_minutes="720"):
    application = Flask(__name__)
    application.secret_key = "isolated-session-tests"
    manager = LoginManager(application)
    user = BrowserUser()
    credential = Credential()
    clock = {"now": datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)}
    events = []
    lifecycle = SessionLifecycle(
        SessionLifecycleDependencies(
            now=lambda: clock["now"],
            credential_for_user=lambda _user: credential,
            security_event=lambda event, **facts: events.append((event, facts)),
        )
    )
    monkeypatch.setenv("ATCROSTER_SESSION_IDLE_MINUTES", idle_minutes)
    monkeypatch.setenv("ATCROSTER_SESSION_ABSOLUTE_MINUTES", absolute_minutes)

    @manager.user_loader
    def load_user(user_id):
        return user if user_id == str(user.id) else None

    @application.before_request
    def enforce_session():
        return lifecycle.enforce_request(current_user)

    @application.get("/login", endpoint="login")
    def login():
        login_user(user)
        lifecycle.initialize(user)
        return "logged-in"

    @application.get("/private")
    def private():
        return "private"

    return application, user, credential, clock, events, lifecycle


def test_initialize_sets_revocation_stamp_and_session_times(monkeypatch):
    application, user, _credential, clock, _events, lifecycle = session_application(
        monkeypatch
    )
    client = application.test_client()
    client.get("/login")
    with client.session_transaction() as browser_session:
        assert browser_session["_auth_stamp"] == lifecycle.auth_stamp(user)
        assert browser_session["_last_seen_epoch"] == int(clock["now"].timestamp())
        assert browser_session["_session_started_at"] == clock["now"].isoformat()
        assert browser_session["_session_nonce"]


def test_idle_timeout_revokes_session(monkeypatch):
    application, _user, _credential, clock, events, _lifecycle = session_application(
        monkeypatch, idle_minutes="30"
    )
    client = application.test_client()
    client.get("/login")
    clock["now"] += timedelta(minutes=31)
    response = client.get("/private")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")
    assert events[0][0] == "session_expired"
    assert events[0][1]["reason"] == "idle"


def test_absolute_timeout_wins_over_recent_activity(monkeypatch):
    application, _user, _credential, clock, events, _lifecycle = session_application(
        monkeypatch, absolute_minutes="60"
    )
    client = application.test_client()
    client.get("/login")
    clock["now"] += timedelta(minutes=61)
    with client.session_transaction() as browser_session:
        browser_session["_last_seen_epoch"] = int(clock["now"].timestamp())
    assert client.get("/private").status_code == 302
    assert events[0][1]["reason"] == "absolute"


def test_role_or_mfa_change_forces_revocation(monkeypatch):
    application, user, credential, _clock, events, _lifecycle = session_application(
        monkeypatch
    )
    role_client = application.test_client()
    role_client.get("/login")
    user.role = "user"
    assert role_client.get("/private").status_code == 302
    assert events[-1][0] == "session_forced_invalidation"

    user.role = "admin"
    mfa_client = application.test_client()
    mfa_client.get("/login")
    credential.encrypted_secret = "replacement-secret"
    assert mfa_client.get("/private").status_code == 302
    assert events[-1][0] == "session_forced_invalidation"
