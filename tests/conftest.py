"""Global test safety boundaries.

Tests must never inherit the developer's live DATABASE_URL. Real PostgreSQL
integration runs opt in through ATCROSTER_TEST_CONTROL_DATABASE_URL; every
other run receives a disposable, process-specific SQLite control database.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import tempfile


integration_url = os.environ.get("ATCROSTER_TEST_CONTROL_DATABASE_URL")
if integration_url:
    os.environ["DATABASE_URL"] = integration_url
    os.environ["CONTROL_DATABASE_URL"] = integration_url
else:
    test_database = (
        Path(tempfile.gettempdir()) / f"atcroster-pytest-control-{os.getpid()}.db"
    )
    test_url = f"sqlite:///{test_database}"
    os.environ["DATABASE_URL"] = test_url
    os.environ["CONTROL_DATABASE_URL"] = test_url

    def _remove_test_database() -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                test_database.with_name(test_database.name + suffix).unlink()
            except FileNotFoundError:
                pass

    atexit.register(_remove_test_database)


# Most route tests intentionally exercise pre-control-plane fixture accounts.
# Production and normal development remain fail-closed unless this migration
# escape hatch is explicitly enabled.
os.environ.setdefault("ATCROSTER_ENABLE_LEGACY_LOGIN", "true")
