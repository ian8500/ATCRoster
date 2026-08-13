"""Canonical shift-request states and permitted transitions."""

from __future__ import annotations


REQUEST_STATUSES = frozenset({
    "pending", "approved", "rejected", "fulfilled", "cancelled",
})
REQUEST_TRANSITIONS = {
    "pending": frozenset({"approved", "rejected", "cancelled"}),
    "approved": frozenset({"rejected", "cancelled"}),
    "rejected": frozenset(),
    "cancelled": frozenset(),
    "fulfilled": frozenset(),
}
