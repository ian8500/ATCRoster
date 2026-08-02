from __future__ import annotations

from flask import Flask, abort, g
from flask_login import LoginManager, UserMixin, login_user
from jinja2 import DictLoader
from werkzeug.exceptions import BadRequest

from atcroster.errors import ErrorHandlerDependencies, register_error_handlers


class PlatformUser(UserMixin):
    id = 17
    unit_id = None
    role = "superadmin"


def error_application() -> tuple[Flask, list[tuple[str, dict[str, object]]]]:
    application = Flask(__name__)
    application.secret_key = "isolated-error-handler-tests"
    application.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    application.jinja_loader = DictLoader(
        {
            "error.html": (
                "{{ status_code }}|{{ error_title }}|{{ error_message }}|"
                "{{ request_id }}|{{ home_url }}|{{ home_label }}"
            )
        }
    )
    manager = LoginManager(application)
    platform_user = PlatformUser()

    @manager.user_loader
    def load_user(user_id):
        return platform_user if user_id == str(platform_user.id) else None

    @application.before_request
    def request_context():
        g.request_id = "request-errors"

    @application.get("/", endpoint="index")
    def index():
        return "index"

    @application.get("/platform/admin", endpoint="platform_admin")
    def platform_admin():
        return "platform"

    @application.get("/briefing", endpoint="briefing.home")
    def briefing_home():
        return "briefing"

    @application.get("/training", endpoint="training_home")
    def training_home():
        return "training"

    @application.get("/competency", endpoint="competency_home")
    def competency_home():
        return "competency"

    @application.get("/csrf")
    def csrf_failure():
        raise BadRequest("Invalid CSRF token.")

    @application.get("/generic-bad")
    def generic_bad_request():
        abort(400)

    @application.get("/denied")
    def denied():
        abort(403)

    @application.get("/platform-denied")
    def platform_denied():
        login_user(platform_user)
        abort(403)

    @application.get("/explode")
    def explode():
        raise RuntimeError("expected test failure")

    events: list[tuple[str, dict[str, object]]] = []
    register_error_handlers(
        application,
        ErrorHandlerDependencies(
            security_event=lambda event, **facts: events.append((event, facts))
        ),
    )
    return application, events


def test_csrf_400_preserves_message_request_id_and_security_event():
    application, events = error_application()
    response = application.test_client().get("/csrf")
    assert response.status_code == 400
    assert b"This page or form has expired" in response.data
    assert b"request-errors" in response.data
    assert events == [("csrf_rejected", {"route": "csrf_failure"})]


def test_generic_400_preserves_safe_user_message():
    application, _events = error_application()
    response = application.test_client().get("/generic-bad")
    assert response.status_code == 400
    assert b"The request was not valid" in response.data


def test_403_records_event_and_preserves_platform_admin_navigation():
    application, events = error_application()
    response = application.test_client().get("/platform-denied")
    assert response.status_code == 403
    assert b"Platform administrators cannot access airport" in response.data
    assert b"/platform/admin" in response.data
    assert events == [
        (
            "forbidden_role_action",
            {
                "route": "platform_denied",
                "unit_id": None,
                "actor_id": 17,
            },
        )
    ]


def test_404_preserves_module_navigation_and_request_id():
    application, _events = error_application()
    response = application.test_client().get("/briefing/missing")
    assert response.status_code == 404
    assert b"That page or record was not found" in response.data
    assert b"/briefing" in response.data
    assert b"Return to briefing" in response.data
    assert b"request-errors" in response.data


def test_500_preserves_request_id_and_logs_failure(caplog):
    application, _events = error_application()
    response = application.test_client().get("/explode")
    assert response.status_code == 500
    assert b"request-errors" in response.data
    assert "unhandled_request_error request_id=request-errors" in caplog.text
