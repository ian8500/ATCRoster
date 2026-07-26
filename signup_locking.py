"""Cross-process lock used while advancing an invitation signup saga."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from sqlalchemy import text

_local_guard = threading.Lock()
_local_locks: dict[int, threading.Lock] = {}


@contextmanager
def invitation_signup_lock(db, invitation_id: int):
    """Serialize one invitation while allowing unrelated signups to proceed."""
    engine = db.engine
    if engine.dialect.name == "postgresql":
        # Hold a dedicated connection for the lifetime of the multi-database
        # saga. Session commits cannot release this session advisory lock.
        with engine.connect() as connection:
            connection.execute(
                text("SELECT pg_advisory_lock(:key)"),
                {"key": int(invitation_id)},
            )
            try:
                yield
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": int(invitation_id)},
                )
    else:
        with _local_guard:
            lock = _local_locks.setdefault(int(invitation_id), threading.Lock())
        with lock:
            yield
