"""Roster shift-code normalization and classification."""

from __future__ import annotations

from typing import Callable, Iterable


def normalize_code(value: object) -> str:
    return str(value or "").strip().upper()


def is_non_working(value: str, non_working_codes: Callable[[], Iterable[str]]) -> bool:
    return normalize_code(value) in non_working_codes()
