"""Cached roster shift lookup and selector grouping."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from .shifts import shift_groups_snapshot


class ShiftLookupService:
    """Bind shift catalogue lookups to an application's tenant context."""

    def __init__(
        self,
        *,
        ShiftType: Any,
        current_unit_id: Callable[[], int],
        banned_codes: Callable[[], set[str]],
    ) -> None:
        self.ShiftType = ShiftType
        self.current_unit_id = current_unit_id
        self.banned_codes = banned_codes

    @lru_cache(maxsize=256)
    def by_code(self, unit_id: int, code: str) -> Any:
        return self.ShiftType.query.filter_by(unit_id=unit_id, code=code).first()

    def get(self, code: str, unit_id: int | None = None) -> Any:
        return self.by_code(
            int(unit_id or self.current_unit_id() or 1), (code or "").upper()
        )

    def refresh(self) -> None:
        self.by_code.cache_clear()

    @lru_cache(maxsize=128)
    def groups(self, unit_id: int) -> tuple[list[Any], list[Any], list[Any]]:
        return shift_groups_snapshot(self.ShiftType, unit_id, self.banned_codes)


def create_shift_lookup_service(
    *, operational_models: Any, **services: Any
) -> ShiftLookupService:
    """Bind shift catalogue records inside the roster domain."""
    return ShiftLookupService(
        ShiftType=operational_models.ShiftType,
        **services,
    )
