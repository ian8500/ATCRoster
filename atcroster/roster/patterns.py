"""Roster pattern parsing and validation adapters."""

from __future__ import annotations

from typing import Callable


def expand(raw_value: str | None, expand_pattern: Callable[[str | None], list[str]]) -> list[str]:
    return expand_pattern(raw_value)


def validate(raw_value: str | None, validated_pattern: Callable[[str | None], list[str]]) -> list[str]:
    return validated_pattern(raw_value)
