"""Run the provisioning worker with a minimal Railway health endpoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import threading


def main() -> None:
    worker = subprocess.Popen(
        ["python", "scripts/run_provisioning_worker.py"],
    )

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            healthy = worker.poll() is None
            if self.path not in {"/health/live", "/health/ready"}:
                status, payload = 404, {"status": "not_found"}
            elif healthy:
                status, payload = 200, {"status": "ready"}
            else:
                status, payload = 503, {"status": "not_ready"}
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(
        # Railway must reach this health-only listener from outside the
        # container; it exposes no operational or administrative routes.
        ("0.0.0.0", int(os.environ.get("PORT", "8080"))),  # nosec B104
        HealthHandler,
    )

    def stop_when_worker_exits() -> None:
        return_code = worker.wait()
        if return_code:
            os._exit(return_code)
        server.shutdown()

    threading.Thread(target=stop_when_worker_exits, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=15)


if __name__ == "__main__":
    main()
