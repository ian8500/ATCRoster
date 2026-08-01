"""Privacy-safe distributed rate limiting for security-sensitive operations."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass
from typing import Protocol


class LimiterUnavailable(RuntimeError):
    pass


class RateLimiter(Protocol):
    def consume(self, key: str, limit: int, window_seconds: int) -> bool: ...
    def reset(self, key: str) -> None: ...


class MemoryRateLimiter:
    """Process-local implementation for tests and explicit local development."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, limit: int, window_seconds: int) -> bool:
        now = self.clock()
        with self._lock:
            expires_at, count = self._entries.get(key, (now + window_seconds, 0))
            if now >= expires_at:
                expires_at, count = now + window_seconds, 0
            count += 1
            self._entries[key] = (expires_at, count)
            return count <= limit

    def reset(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)


class RedisRateLimiter:
    """Atomic fixed-window limiter shared by every web and worker process."""

    def __init__(self, redis_client, prefix: str = "atcroster:limit"):
        self.redis = redis_client
        self.prefix = prefix

    def consume(self, key: str, limit: int, window_seconds: int) -> bool:
        redis_key = f"{self.prefix}:{key}"
        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds, nx=True)
            result = pipe.execute()
            if not isinstance(result, (list, tuple)) or len(result) != 2:
                raise ValueError("malformed limiter response")
            count = int(result[0])
            if count < 1:
                raise ValueError("invalid limiter counter")
        except Exception as exc:
            raise LimiterUnavailable("shared limiter unavailable") from exc
        return count <= int(limit)

    def reset(self, key: str) -> None:
        try:
            self.redis.delete(f"{self.prefix}:{key}")
        except Exception as exc:
            raise LimiterUnavailable("shared limiter unavailable") from exc

    def verify(self) -> None:
        try:
            if self.redis.ping() is not True:
                raise ValueError("unexpected Redis PING response")
        except Exception as exc:
            raise LimiterUnavailable("shared limiter unavailable") from exc


@dataclass(frozen=True)
class RateLimitPolicy:
    limit: int
    window_seconds: int
    fail_closed: bool = False


def privacy_key(secret: str, *parts: object) -> str:
    """Return an opaque keyed digest; raw identifiers never enter Redis."""
    material = "\x1f".join(str(part or "") for part in parts).encode()
    return hmac.new(secret.encode(), material, hashlib.sha256).hexdigest()
