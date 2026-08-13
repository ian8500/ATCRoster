"""Regression coverage for the exact E2E launcher invocation used by CI."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_e2e_server_direct_script_invocation_seeds_database_and_becomes_ready(tmp_path):
    port = _available_port()
    database = tmp_path / "e2e.db"
    environment = os.environ.copy()
    environment.update(
        {
            "ATCROSTER_E2E_DATABASE": str(database),
            "ATCROSTER_ENV": "development",
            "ATCROSTER_SKIP_RUNTIME_SCHEMA": "1",
            "FLASK_SECRET_KEY": "e2e-only-not-a-production-secret-123456",
            "PORT": str(port),
        }
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("DATABASE_URL", None)
    process = subprocess.Popen(
        [sys.executable, "scripts/run_e2e_server.py"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready_url = f"http://127.0.0.1:{port}/health/ready"
        for _ in range(60):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"E2E launcher exited early:\n{output}")
            try:
                with urlopen(ready_url, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise AssertionError("E2E launcher did not expose /health/ready")
        assert database.exists()
        assert database.stat().st_size > 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
