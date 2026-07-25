
import pytest

from tenancy import (
    DatabaseRoute,
    OperationalDatabaseRouter,
    bind_authenticated_unit,
    reset_authenticated_unit,
)


def test_operational_database_router_uses_authenticated_unit_only(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIT_A_DATABASE_URL", f"sqlite:///{tmp_path / 'a.db'}")
    monkeypatch.setenv("UNIT_B_DATABASE_URL", f"sqlite:///{tmp_path / 'b.db'}")
    routes = {
        1: DatabaseRoute(1, "UNIT_A_DATABASE_URL"),
        2: DatabaseRoute(2, "UNIT_B_DATABASE_URL"),
    }
    router = OperationalDatabaseRouter(routes.__getitem__)
    token = bind_authenticated_unit(1)
    try:
        engine_a = router.engine_for_authenticated_unit()
    finally:
        reset_authenticated_unit(token)
    token = bind_authenticated_unit(2)
    try:
        engine_b = router.engine_for_authenticated_unit()
    finally:
        reset_authenticated_unit(token)

    assert engine_a.url.database != engine_b.url.database
    router.dispose()


def test_operational_database_router_rejects_mismatched_route(monkeypatch):
    monkeypatch.setenv("UNIT_B_DATABASE_URL", "sqlite://")
    router = OperationalDatabaseRouter(
        lambda _unit_id: DatabaseRoute(2, "UNIT_B_DATABASE_URL")
    )
    token = bind_authenticated_unit(1)
    try:
        with pytest.raises(PermissionError):
            router.engine_for_authenticated_unit()
    finally:
        reset_authenticated_unit(token)

