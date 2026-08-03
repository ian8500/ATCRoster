"""Small, explicitly invalidated cache for expensive monthly roster analysis."""

from __future__ import annotations

from threading import RLock
from time import monotonic
from typing import Any


class RosterMonthCache:
    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._items: dict[tuple[int, int, int], tuple[float, Any]] = {}
        self._lock = RLock()

    def get(self, unit_id: int, year: int, month: int) -> Any | None:
        key = (int(unit_id), int(year), int(month))
        now = monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            return value

    def set(self, unit_id: int, year: int, month: int, value: Any) -> None:
        with self._lock:
            self._items[(int(unit_id), int(year), int(month))] = (
                monotonic(), value
            )

    def invalidate_unit(self, unit_id: int) -> None:
        resolved = int(unit_id)
        with self._lock:
            for key in tuple(self._items):
                if key[0] == resolved:
                    self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
