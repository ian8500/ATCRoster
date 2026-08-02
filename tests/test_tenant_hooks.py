from __future__ import annotations

from dataclasses import dataclass

from flask import Flask, g

from atcroster.tenancy_hooks import TenantHookDependencies, register_tenant_hooks


@dataclass
class BrowserUser:
    is_authenticated: bool
    role: str = "user"
    unit_id: int | None = None


@dataclass
class Route:
    secret_name: str


def tenant_application(user: BrowserUser, *, production=False, explode=False):
    application = Flask(__name__)
    calls = []

    def record(name, result=None):
        def callback(*args):
            calls.append((name, args))
            return result

        return callback

    register_tenant_hooks(
        application,
        TenantHookDependencies(
            deployment_environment="production" if production else "test",
            current_user=lambda: user,
            enforce_session=lambda _user: None,
            routing_for_unit=lambda unit_id: Route(f"UNIT_{unit_id}_DATABASE_URL"),
            clear_context=record("clear"),
            bind_authenticated_unit=record("bind_unit", "unit-token"),
            reset_authenticated_unit=record("reset_unit"),
            bind_platform_control=record("bind_platform", "platform-token"),
            reset_platform_control=record("reset_platform"),
        ),
    )

    @application.get("/")
    def index():
        if explode:
            raise RuntimeError("expected request failure")
        return {
            "unit_token": getattr(g, "tenant_context_token", None),
            "platform_token": getattr(g, "platform_control_token", None),
        }

    return application, calls


def test_anonymous_requests_remain_unbound_and_clear_context_twice():
    application, calls = tenant_application(BrowserUser(False))
    assert application.test_client().get("/").status_code == 200
    assert [name for name, _args in calls] == ["clear", "clear"]


def test_verified_unit_is_bound_from_user_and_reset_after_request():
    application, calls = tenant_application(BrowserUser(True, unit_id=12))
    response = application.test_client().get("/")
    assert response.json == {"platform_token": None, "unit_token": "unit-token"}
    assert ("bind_unit", (12, "UNIT_12_DATABASE_URL")) in calls
    assert ("reset_unit", ("unit-token",)) in calls


def test_superadmin_binds_only_platform_control():
    application, calls = tenant_application(BrowserUser(True, "superadmin", 99))
    response = application.test_client().get("/")
    assert response.json == {
        "platform_token": "platform-token",
        "unit_token": None,
    }
    assert not any(name == "bind_unit" for name, _args in calls)
    assert ("reset_platform", ("platform-token",)) in calls


def test_tenant_context_is_reset_after_exception():
    application, calls = tenant_application(BrowserUser(True, unit_id=3), explode=True)
    application.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    assert application.test_client().get("/").status_code == 500
    assert ("reset_unit", ("unit-token",)) in calls
    assert [name for name, _args in calls][-1] == "clear"
