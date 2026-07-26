import hashlib

from cryptography.fernet import Fernet
import pytest

import app
from app import (
    DatabaseRoutingMetadata,
    ProvisioningJob,
    SecureInvitation,
    Unit,
    db,
)
import platform_provisioning
from platform_provisioning import (
    ProvisioningWorker,
    TokenEnvelopeError,
    pop_one_time_token,
    store_one_time_token,
)


def _job(tmp_path, monkeypatch):
    db.drop_all()
    db.create_all()
    unit = Unit(
        code="WRK",
        name="Worker Test",
        status="provisioning",
        active_user_limit=3,
    )
    db.session.add(unit)
    db.session.flush()
    secret_name = f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL"
    monkeypatch.setenv(secret_name, f"sqlite:///{tmp_path / 'operational.db'}")
    db.session.add(
        DatabaseRoutingMetadata(
            unit_id=unit.id,
            secret_name=secret_name,
            provisioning_state="queued",
        )
    )
    job = ProvisioningJob(
        unit_id=unit.id,
        idempotency_key=hashlib.sha256(b"worker").hexdigest(),
        state="queued",
        active_key="active",
        next_attempt_at=app.utcnow(),
    )
    db.session.add(job)
    db.session.commit()
    return unit.id, job.id


def test_worker_completes_once_and_token_is_one_time(tmp_path, monkeypatch):
    with app.app.app_context():
        unit_id, job_id = _job(tmp_path, monkeypatch)
    worker = ProvisioningWorker(app.app)
    assert worker.run_once()
    assert not worker.run_once()
    with app.app.app_context():
        job = db.session.get(ProvisioningJob, job_id)
        assert job.state == "completed"
        assert job.active_key is None
        invitations = SecureInvitation.query.filter_by(unit_id=unit_id).all()
        assert len(invitations) == 1
        assert invitations[0].active_bootstrap_key == "active"
    assert pop_one_time_token(job_id, unit_id)
    assert pop_one_time_token(job_id, unit_id) is None


def test_worker_recovers_abandoned_job(tmp_path, monkeypatch):
    with app.app.app_context():
        _unit_id, job_id = _job(tmp_path, monkeypatch)
        job = db.session.get(ProvisioningJob, job_id)
        job.state = "running"
        job.lease_owner = "crashed-worker"
        job.lease_expires_at = app.utcnow() - app.timedelta(seconds=1)
        db.session.commit()
    worker = ProvisioningWorker(app.app)
    assert worker.recover_stale_jobs() == 1
    assert worker.run_once()
    with app.app.app_context():
        assert db.session.get(ProvisioningJob, job_id).state == "completed"


def test_unexpired_lease_is_not_recovered(tmp_path, monkeypatch):
    with app.app.app_context():
        _unit_id, job_id = _job(tmp_path, monkeypatch)
        job = db.session.get(ProvisioningJob, job_id)
        job.state = "running"
        job.lease_owner = "live-worker"
        job.lease_expires_at = app.utcnow() + app.timedelta(minutes=30)
        db.session.commit()
    assert ProvisioningWorker(app.app).recover_stale_jobs() == 0
    with app.app.app_context():
        assert db.session.get(ProvisioningJob, job_id).state == "running"


def test_token_envelope_is_ciphertext_versioned_and_bound(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("ATCROSTER_TOKEN_ENCRYPTION_KEYS", f"v7:{key}")
    raw = "sensitive-bootstrap-token"
    store_one_time_token(91, 17, raw)
    stored = platform_provisioning._development_tokens[91]
    assert stored.startswith("v7.")
    assert raw not in stored
    with pytest.raises(TokenEnvelopeError, match="binding"):
        pop_one_time_token(91, 18)


def test_token_envelope_rejects_unknown_key_version(monkeypatch):
    first = Fernet.generate_key().decode()
    second = Fernet.generate_key().decode()
    monkeypatch.setenv("ATCROSTER_TOKEN_ENCRYPTION_KEYS", f"v1:{first}")
    store_one_time_token(92, 17, "one-time")
    monkeypatch.setenv("ATCROSTER_TOKEN_ENCRYPTION_KEYS", f"v2:{second}")
    with pytest.raises(TokenEnvelopeError, match="version"):
        pop_one_time_token(92, 17)


def test_token_envelope_store_failure_is_safe(monkeypatch):
    class FailedCache:
        def set(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(platform_provisioning, "_token_cache", lambda: FailedCache())
    monkeypatch.setenv(
        "ATCROSTER_TOKEN_ENCRYPTION_KEYS",
        f"v1:{Fernet.generate_key().decode()}",
    )
    with pytest.raises(TokenEnvelopeError, match="store"):
        store_one_time_token(93, 17, "one-time")
