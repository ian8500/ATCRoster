import os
import secrets

import pytest

from rate_limiting import RedisRateLimiter


pytestmark = pytest.mark.skipif(
    not os.environ.get("REDIS_URL"),
    reason="REDIS_URL is required for the distributed integration suite",
)


def test_rate_limit_is_shared_across_independent_instances():
    import redis

    client_a = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    client_b = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    prefix = f"atcroster:test:{secrets.token_hex(8)}"
    first = RedisRateLimiter(client_a, prefix=prefix)
    second = RedisRateLimiter(client_b, prefix=prefix)
    assert first.consume("opaque-subject", 2, 60)
    assert second.consume("opaque-subject", 2, 60)
    assert not first.consume("opaque-subject", 2, 60)
    second.reset("opaque-subject")
    assert first.consume("opaque-subject", 2, 60)
