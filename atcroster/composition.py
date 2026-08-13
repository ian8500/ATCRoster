"""Small, explicit primitives used while composing the application.

The application has a few genuine construction cycles: reporting needs the
work-pattern service, while planning needs reporting.  A deferred reference
makes those cycles visible and fails clearly if startup order is wrong, rather
than relying on an untyped lookup in ``application`` globals.
"""

from __future__ import annotations

from typing import Generic, TypeVar


T = TypeVar("T")


class DeferredReference(Generic[T]):
    """A single-assignment dependency populated during application assembly."""

    def __init__(self, name: str):
        self.name = name
        self._value: T | None = None

    def set(self, value: T) -> None:
        if self._value is not None:
            raise RuntimeError(f"Deferred dependency already configured: {self.name}")
        self._value = value

    def get(self) -> T:
        if self._value is None:
            raise RuntimeError(f"Deferred dependency is not configured: {self.name}")
        return self._value
