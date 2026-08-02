"""Tenant context and secret-backed operational database routing.

The browser never chooses a tenant or database. ``bind_authenticated_unit`` is
called with a trusted membership value after authentication and stores it in a
context variable for the lifetime of the request.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_unit_context: ContextVar[int | None] = ContextVar("atc_roster_unit", default=None)
_route_context: ContextVar[DatabaseRoute | None] = ContextVar(
    "atc_roster_database_route", default=None
)
_operational_access_forbidden: ContextVar[bool] = ContextVar(
    "atc_roster_operational_access_forbidden", default=False
)


def bind_authenticated_unit(
    unit_id: int, secret_name: str | None = None
):
    if not isinstance(unit_id, int) or unit_id < 1:
        raise ValueError("A trusted positive unit id is required")
    unit_token = _unit_context.set(unit_id)
    route_token = _route_context.set(
        DatabaseRoute(unit_id, secret_name) if secret_name else None
    )
    return unit_token, route_token


def reset_authenticated_unit(token) -> None:
    if isinstance(token, tuple):
        unit_token, route_token = token
        _route_context.reset(route_token)
        _unit_context.reset(unit_token)
    else:
        _unit_context.reset(token)


def authenticated_unit_id() -> int:
    unit_id = _unit_context.get()
    if unit_id is None:
        raise RuntimeError("No authenticated tenant context")
    return unit_id


@dataclass(frozen=True)
class DatabaseRoute:
    unit_id: int
    secret_name: str


class OperationalDatabaseRouter:
    """Resolve unit databases through deployment secret names only."""

    def __init__(self, route_lookup: Callable[[int], DatabaseRoute]):
        self.route_lookup = route_lookup
        self._engines: dict[int, Engine] = {}

    def engine_for_authenticated_unit(self) -> Engine:
        unit_id = authenticated_unit_id()
        route = self.route_lookup(unit_id)
        if route.unit_id != unit_id:
            raise PermissionError("Database route does not match authenticated unit")
        database_url = os.environ.get(route.secret_name)
        if not database_url:
            raise RuntimeError(f"Deployment secret {route.secret_name!r} is unavailable")
        if unit_id not in self._engines:
            options = {
                "pool_pre_ping": True,
                "pool_recycle": 280,
                "pool_size": int(os.environ.get("ATCROSTER_OPERATIONAL_POOL_SIZE", "5")),
                "max_overflow": int(
                    os.environ.get("ATCROSTER_OPERATIONAL_MAX_OVERFLOW", "5")
                ),
                "pool_timeout": int(
                    os.environ.get("ATCROSTER_DB_POOL_TIMEOUT_SECONDS", "10")
                ),
            }
            if database_url.startswith("postgresql"):
                options["connect_args"] = {
                    "connect_timeout": int(
                        os.environ.get("ATCROSTER_DB_CONNECT_TIMEOUT_SECONDS", "5")
                    ),
                    "options": (
                        "-c statement_timeout="
                        + str(int(os.environ.get(
                            "ATCROSTER_DB_STATEMENT_TIMEOUT_MS", "15000"
                        )))
                    ),
                }
            self._engines[unit_id] = create_engine(
                database_url, **options
            )
        return self._engines[unit_id]

    def dispose(self) -> None:
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()


def authenticated_database_route() -> DatabaseRoute:
    route = _route_context.get()
    unit_id = authenticated_unit_id()
    if route is None:
        raise RuntimeError("No operational database route is bound")
    if route.unit_id != unit_id:
        raise PermissionError("Operational database route is inconsistent")
    return route


def authenticated_database_route_optional() -> DatabaseRoute | None:
    """Return the trusted route when one is configured for this unit."""
    unit_id = authenticated_unit_id()
    route = _route_context.get()
    if route is not None and route.unit_id != unit_id:
        raise PermissionError("Operational database route is inconsistent")
    return route


_context_router = OperationalDatabaseRouter(
    lambda unit_id: authenticated_database_route()
)


def operational_engine_for_authenticated_unit() -> Engine:
    if _operational_access_forbidden.get():
        raise PermissionError(
            "Platform control context cannot open an operational database"
        )
    return _context_router.engine_for_authenticated_unit()


def dispose_operational_engines() -> None:
    _context_router.dispose()


def bind_platform_control():
    return _operational_access_forbidden.set(True)


def reset_platform_control(token) -> None:
    _operational_access_forbidden.reset(token)


def clear_request_context() -> None:
    """Force a clean boundary at HTTP request start/end."""
    _unit_context.set(None)
    _route_context.set(None)
    _operational_access_forbidden.set(False)


@contextmanager
def operational_unit_context(
    unit_id: int, secret_name: str
):
    """Required boundary for CLI jobs, exports and background operations."""
    token = bind_authenticated_unit(unit_id, secret_name)
    try:
        yield operational_engine_for_authenticated_unit()
    finally:
        reset_authenticated_unit(token)


@contextmanager
def authenticated_unit_context(unit_id: int, secret_name: str | None = None):
    """Restore a previously authenticated tenant across a deferred boundary."""
    token = bind_authenticated_unit(unit_id, secret_name)
    try:
        yield
    finally:
        reset_authenticated_unit(token)


class TenantRepository:
    """Small service-layer guard used by operational repositories."""

    def __init__(self, model, session):
        self.model = model
        self.session = session

    def query(self):
        return self.session.query(self.model).filter(
            self.model.unit_id == authenticated_unit_id()
        )

    def get(self, record_id: int):
        return self.query().filter(self.model.id == int(record_id)).one_or_none()

    def add(self, record):
        unit_id = authenticated_unit_id()
        if getattr(record, "unit_id", unit_id) != unit_id:
            raise PermissionError("Cross-unit writes are forbidden")
        record.unit_id = unit_id
        self.session.add(record)
        return record
