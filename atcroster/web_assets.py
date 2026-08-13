"""Jinja helpers for static asset cache busting."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Callable

from flask import Flask, url_for


def register_template_helpers(
    app: Flask,
) -> tuple[Callable[[str], int | None], Callable[..., str]]:
    """Register time and versioned-static-URL helpers on the Jinja environment."""

    @lru_cache(maxsize=256)
    def asset_version(filename: str) -> int | None:
        static_folder = app.static_folder
        if not static_folder:
            return None
        try:
            return int(os.path.getmtime(os.path.join(static_folder, filename)))
        except (OSError, TypeError, ValueError):
            return None

    def asset_url(filename: str, **extra: object) -> str:
        version = asset_version(filename)
        if version is not None:
            return url_for("static", filename=filename, v=version, **extra)
        return url_for("static", filename=filename, **extra)

    app.jinja_env.globals["now"] = datetime.now
    app.jinja_env.globals["asset_url"] = asset_url
    return asset_version, asset_url
