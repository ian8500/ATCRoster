"""Application time source shared by models and domain services."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the timezone-aware UTC timestamp used by the application."""
    return datetime.now(timezone.utc)


def as_naive_utc(value: datetime) -> datetime:
    """Normalize a timestamp to the naive UTC form used by database columns."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)
