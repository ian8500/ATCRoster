"""Start the configured Railway process without duplicating service manifests."""

from __future__ import annotations

import os


def main() -> None:
    process_type = os.environ.get("ATCROSTER_PROCESS_TYPE", "web").strip().lower()
    if process_type == "web":
        os.execvp(
            "waitress-serve",
            [
                "waitress-serve",
                "--host=0.0.0.0",
                "--port=8080",
                "--threads=8",
                "wsgi:application",
            ],
        )
    if process_type == "worker":
        os.execvp(
            "python",
            ["python", "scripts/run_worker_service.py"],
        )
    raise SystemExit("ATCROSTER_PROCESS_TYPE must be either 'web' or 'worker'.")


if __name__ == "__main__":
    main()
