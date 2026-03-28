"""Desktop launcher for ATC Roster (Windows/macOS/Linux).

This wrapper starts the Flask app locally and opens it in a native webview
window so users can run it like a desktop app.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from urllib.request import urlopen

import webview

from app import app


def _find_free_port(start: int = 5001, end: int = 5999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free localhost port found in range 5001-5999")


def _wait_for_server(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.5):
                return True
        except Exception:
            time.sleep(0.2)
    return False


def _run_server(port: int) -> None:
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def main() -> None:
    os.environ.setdefault("FLASK_ENV", "production")
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/"

    server_thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
    server_thread.start()

    if not _wait_for_server(url):
        raise RuntimeError("ATC Roster server failed to start")

    # If webview cannot initialize in a given environment, fall back to browser.
    try:
        webview.create_window("ATC Roster", url, width=1366, height=900, min_size=(1024, 700))
        webview.start()
    except Exception:
        webbrowser.open(url)
        # Keep process alive while the Flask thread serves requests.
        while server_thread.is_alive():
            time.sleep(1)


if __name__ == "__main__":
    main()
