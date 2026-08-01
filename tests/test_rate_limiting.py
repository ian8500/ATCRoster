import pytest

from rate_limiting import (
    LimiterUnavailable, MemoryRateLimiter, RedisRateLimiter, privacy_key,
)


def test_memory_limiter_enforces_and_expires():
    now = [0.0]
    limiter = MemoryRateLimiter(clock=lambda: now[0])
    assert limiter.consume("opaque", 2, 10)
    assert limiter.consume("opaque", 2, 10)
    assert not limiter.consume("opaque", 2, 10)
    now[0] = 11
    assert limiter.consume("opaque", 2, 10)


def test_privacy_key_is_keyed_and_contains_no_identifier():
    first = privacy_key("secret-a", "login", "person@example.test")
    second = privacy_key("secret-b", "login", "person@example.test")
    assert first != second
    assert "person" not in first
    assert len(first) == 64


class _FailingRedis:
    def pipeline(self, transaction=True):
        raise ConnectionError("offline")

    def ping(self):
        raise TimeoutError("timed out")


def test_redis_limiter_reports_unavailability_without_leaking_details():
    limiter = RedisRateLimiter(_FailingRedis())
    with pytest.raises(LimiterUnavailable, match="shared limiter unavailable"):
        limiter.consume("opaque", 3, 60)
    with pytest.raises(LimiterUnavailable, match="shared limiter unavailable"):
        limiter.verify()


class _Pipeline:
    def __init__(self, result):
        self.result = result

    def incr(self, _key):
        return self

    def expire(self, _key, _seconds, nx=True):
        return self

    def execute(self):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _RecoveringRedis:
    def __init__(self, results):
        self.results = iter(results)

    def pipeline(self, transaction=True):
        return _Pipeline(next(self.results))

    def ping(self):
        return True


@pytest.mark.parametrize("result", [["not-a-number", True], [], None])
def test_redis_limiter_rejects_malformed_responses(result):
    limiter = RedisRateLimiter(_RecoveringRedis([result]))
    with pytest.raises(LimiterUnavailable, match="shared limiter unavailable"):
        limiter.consume("opaque", 3, 60)


def test_redis_limiter_recovers_after_intermittent_timeout():
    limiter = RedisRateLimiter(
        _RecoveringRedis([TimeoutError("slow"), [1, True]])
    )
    with pytest.raises(LimiterUnavailable):
        limiter.consume("opaque", 3, 60)
    assert limiter.consume("opaque", 3, 60) is True
