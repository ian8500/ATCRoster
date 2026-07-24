"""Tenant context and secret-backed operational database routing.

The browser never chooses a tenant or database. ``bind_authenticated_unit`` is
called with a trusted membership value after authentication and stores it in a
context variable for the lifetime of the request.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import os
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


_unit_context: ContextVar[int | None] = ContextVar("atc_roster_unit", default=None)


def bind_authenticated_unit(unit_id: int):
    if not isinstance(unit_id, int) or unit_id < 1:
        raise ValueError("A trusted positive unit id is required")
    return _unit_context.set(unit_id)


def reset_authenticated_unit(token) -> None:
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
            self._engines[unit_id] = create_engine(
                database_url, pool_pre_ping=True, pool_recycle=280
            )
        return self._engines[unit_id]

    def dispose(self) -> None:
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()


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
