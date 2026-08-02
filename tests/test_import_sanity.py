"""Detect circular imports and import-time bootstrap regressions."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_application_modules_import_in_clean_process():
    root_modules = sorted(
        path.stem
        for path in ROOT.glob("*.py")
        if path.stem not in {"app", "wsgi"}
    )
    import_script = f"""
import importlib
import pkgutil
import atcroster

modules = {root_modules!r}
modules.extend(
    info.name for info in pkgutil.walk_packages(
        atcroster.__path__, prefix='atcroster.'
    )
)
app = importlib.import_module('app')
for name in sorted(set(modules)):
    importlib.import_module(name)
wsgi = importlib.import_module('wsgi')
assert wsgi.application is app.app
"""
    environment = os.environ.copy()
    environment["ATCROSTER_SKIP_BOOTSTRAP"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
