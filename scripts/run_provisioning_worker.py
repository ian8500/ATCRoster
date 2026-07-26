#!/usr/bin/env python3
"""Run the database-backed airport provisioning worker."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
from platform_provisioning import ProvisioningWorker

stopping = False


def _stop(_signum, _frame):
    global stopping
    stopping = True


def main() -> None:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker = ProvisioningWorker(app)
    worker.recover_stale_jobs()
    while not stopping:
        if not worker.run_once():
            time.sleep(2)


if __name__ == "__main__":
    main()
