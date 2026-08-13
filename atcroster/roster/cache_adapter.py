"""Compatibility adapter for optional Flask-Caching roster results."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable


def memoize(cache: Any, seconds: int = 60) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return the configured cache decorator, or a no-op when unavailable."""
    def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
        return cache.memoize(timeout=seconds)(function) if cache else function
    return wrap


def invalidate_month_for_day(
    cache: Any, loader: Callable[..., Any], unit_id: int, day: date | None,
) -> None:
    """Best-effort invalidation for one affected roster month."""
    if not cache or not day:
        return
    try:
        cache.delete_memoized(loader, int(unit_id or 1), day.year, day.month)
    except Exception:
        return
