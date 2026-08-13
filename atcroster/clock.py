"""Application time source shared by models and domain services."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the timezone-aware UTC timestamp used by the application."""
    return datetime.now(timezone.utc)
