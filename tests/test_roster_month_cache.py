from roster_month_cache import RosterMonthCache


def test_month_cache_is_scoped_and_explicitly_invalidated():
    cache = RosterMonthCache(ttl_seconds=30)
    cache.set(1, 2026, 8, {"value": "a"})
    cache.set(1, 2026, 9, {"value": "b"})
    cache.set(2, 2026, 8, {"value": "c"})

    assert cache.get(1, 2026, 8) == {"value": "a"}
    cache.invalidate_unit(1)
    assert cache.get(1, 2026, 8) is None
    assert cache.get(1, 2026, 9) is None
    assert cache.get(2, 2026, 8) == {"value": "c"}


def test_month_cache_clear_removes_every_unit():
    cache = RosterMonthCache(ttl_seconds=30)
    cache.set(1, 2026, 8, "a")
    cache.set(2, 2026, 8, "b")
    cache.clear()
    assert cache.get(1, 2026, 8) is None
    assert cache.get(2, 2026, 8) is None
