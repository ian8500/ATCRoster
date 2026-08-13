#!/usr/bin/env python3
"""Seed and serve ATCRoster exclusively for local/CI browser tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E2E_MFA_SECRET = "JBSWY3DPEHPK3PXP"


def main() -> None:
    database = os.environ.get("ATCROSTER_E2E_DATABASE")
    if not database:
        raise SystemExit("ATCROSTER_E2E_DATABASE is required for browser tests.")
    os.environ.setdefault("ATCROSTER_ENV", "development")
    os.environ.setdefault("ATCROSTER_SKIP_RUNTIME_SCHEMA", "1")
    os.environ.setdefault("FLASK_SECRET_KEY", "e2e-only-not-a-production-secret-123456")
    subprocess.run(
        [sys.executable, "scripts/seed_acceptance_data.py", "--database", database, "--reset"],
        cwd=ROOT,
        check=True,
    )
    import app as roster

    with roster.app.app_context():
        user = roster.Staff.query.filter_by(username="lba.admin").one()
        roster.MfaCredential.query.filter_by(person_id=user.id).delete()
        roster.db.session.add(
            roster.MfaCredential(
                person_id=user.id,
                encrypted_secret=roster._encrypt_mfa_secret(E2E_MFA_SECRET),
                enabled=True,
                recovery_codes_digest="[]",
            )
        )
        roster.db.session.commit()
    roster.app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))


if __name__ == "__main__":
    main()
