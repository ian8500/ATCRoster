import os
import secrets

import pytest

from rate_limiting import RedisRateLimiter, privacy_key


pytestmark = pytest.mark.skipif(
    not os.environ.get("REDIS_URL"),
    reason="REDIS_URL is required for the distributed integration suite",
)


def test_rate_limit_is_shared_across_independent_instances():
    import redis

    client_a = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    client_b = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    prefix = f"atcroster:test:{secrets.token_hex(8)}"
    subject = privacy_key("integration-secret", "person@example.test")
    first = RedisRateLimiter(client_a, prefix=prefix)
    second = RedisRateLimiter(client_b, prefix=prefix)
    assert first.consume(subject, 2, 60)
    assert second.consume(subject, 2, 60)
    assert not first.consume(subject, 2, 60)
    keys = client_a.keys(f"{prefix}:*")
    assert len(keys) == 1
    assert client_a.ttl(keys[0]) > 0
    assert "person@example.test" not in keys[0]
    assert subject in keys[0]
    second.reset(subject)
    assert first.consume(subject, 2, 60)
