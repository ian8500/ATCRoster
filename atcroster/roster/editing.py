"""Roster assignment edit-protection policy."""

from __future__ import annotations

from typing import Any


LOCKED_SOURCES = frozenset({"manual", "leave", "sickness"})


def cell_is_protected(assignment: Any) -> bool:
    """Return whether a materialised assignment may not be overwritten."""
    return bool(assignment.effective_code and assignment.source in LOCKED_SOURCES)
