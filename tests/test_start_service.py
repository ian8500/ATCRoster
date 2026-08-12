import os

from scripts.start_service import waitress_threads


def test_waitress_threads_defaults_and_is_bounded(monkeypatch):
    monkeypatch.delenv("ATCROSTER_WAITRESS_THREADS", raising=False)
    assert waitress_threads() == 4
    monkeypatch.setenv("ATCROSTER_WAITRESS_THREADS", "0")
    assert waitress_threads() == 1
    monkeypatch.setenv("ATCROSTER_WAITRESS_THREADS", "99")
    assert waitress_threads() == 16
    monkeypatch.setenv("ATCROSTER_WAITRESS_THREADS", "not-a-number")
    assert waitress_threads() == 4
