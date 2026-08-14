#!/usr/bin/env python3
"""Seed and serve ATCRoster exclusively for local/CI browser tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E2E_MFA_SECRET = "JBSWY3DPEHPK3PXP"
E2E_MFA_USERNAMES = ("lba.admin", "lba.editor", "lba.atco01")

# Python puts ``scripts/`` first when this file is launched directly. Ensure
# the root-level WSGI compatibility module is importable in CI and locally.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    database = os.environ.get("ATCROSTER_E2E_DATABASE")
    if not database:
        raise SystemExit("ATCROSTER_E2E_DATABASE is required for browser tests.")
    database_path = Path(database).resolve()
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"
    os.environ["CONTROL_DATABASE_URL"] = f"sqlite:///{database_path}"
    os.environ.setdefault("ATCROSTER_ENV", "development")
    os.environ.setdefault("ATCROSTER_SKIP_RUNTIME_SCHEMA", "1")
    os.environ.setdefault("FLASK_SECRET_KEY", "e2e-only-not-a-production-secret-123456")
    subprocess.run(
        [sys.executable, "scripts/seed_acceptance_data.py", "--database", str(database_path), "--reset"],
        cwd=ROOT,
        check=True,
    )
    import app as roster

    with roster.app.app_context():
        users = roster.Staff.query.filter(
            roster.Staff.username.in_(E2E_MFA_USERNAMES)
        ).all()
        if len(users) != len(E2E_MFA_USERNAMES):
            raise RuntimeError("The isolated acceptance fixture is missing a required MFA test user.")
        for user in users:
            roster.MfaCredential.query.filter_by(person_id=user.id).delete()
            roster.db.session.add(
                roster.MfaCredential(
                    unit_id=user.unit_id,
                    person_id=user.id,
                    encrypted_secret=roster._encrypt_field(E2E_MFA_SECRET),
                    enabled=True,
                    reset_required=False,
                    recovery_codes_digest="[]",
                )
            )
        roster.db.session.commit()
    roster.app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))


if __name__ == "__main__":
    main()
