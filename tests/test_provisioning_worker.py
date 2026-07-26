import hashlib

import app
from app import (
    DatabaseRoutingMetadata,
    ProvisioningJob,
    SecureInvitation,
    Unit,
    db,
)
from platform_provisioning import ProvisioningWorker, pop_one_time_token


def _job(tmp_path, monkeypatch):
    db.drop_all()
    db.create_all()
    unit = Unit(
        code="WRK", name="Worker Test", status="provisioning",
        active_user_limit=3,
    )
    db.session.add(unit)
    db.session.flush()
    secret_name = f"ATCROSTER_UNIT_{unit.id}_DATABASE_URL"
    monkeypatch.setenv(
        secret_name, f"sqlite:///{tmp_path / 'operational.db'}"
    )
    db.session.add(DatabaseRoutingMetadata(
        unit_id=unit.id, secret_name=secret_name,
        provisioning_state="queued",
    ))
    job = ProvisioningJob(
        unit_id=unit.id, idempotency_key=hashlib.sha256(b"worker").hexdigest(),
        state="queued", active_key="active", next_attempt_at=app.utcnow(),
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
    assert pop_one_time_token(job_id)
    assert pop_one_time_token(job_id) is None


def test_worker_recovers_abandoned_job(tmp_path, monkeypatch):
    with app.app.app_context():
        _unit_id, job_id = _job(tmp_path, monkeypatch)
        job = db.session.get(ProvisioningJob, job_id)
        job.state = "running"
        job.locked_at = app.utcnow() - app.timedelta(minutes=20)
        db.session.commit()
    worker = ProvisioningWorker(app.app)
    assert worker.recover_stale_jobs() == 1
    assert worker.run_once()
    with app.app.app_context():
        assert db.session.get(ProvisioningJob, job_id).state == "completed"
