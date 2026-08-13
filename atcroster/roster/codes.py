"""Roster shift-code normalization and classification."""

from __future__ import annotations

from typing import Any, Callable, Iterable


def normalize_code(value: object) -> str:
    return str(value or "").strip().upper()


def is_non_working(value: str, non_working_codes: Callable[[], Iterable[str]]) -> bool:
    return normalize_code(value) in non_working_codes()


def is_working_with_prefix(value: str, prefix: str, non_working_codes: Callable[[], Iterable[str]], shift_lookup: Callable[[str], Any]) -> bool:
    """Classify a known working shift by normalized code prefix."""
    code = normalize_code(value)
    if not code or code in non_working_codes():
        return False
    shift = shift_lookup(code)
    return bool(getattr(shift, "is_working", True)) and code.startswith(prefix)
