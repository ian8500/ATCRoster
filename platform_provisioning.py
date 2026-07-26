"""Durable, idempotent airport database provisioning.

The HTTP application only queues jobs. This module is run by a separate
worker process and deliberately records only privacy-safe state and error
codes in the control database.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import socket
from datetime import timedelta

from sqlalchemy import create_engine, inspect

from scripts.migrate_all_databases import (
    _canonical_database_url,
    upgrade_database,
)

ACTIVE_STATES = ("queued", "running", "retry_wait")
SECRET_NAME_PATTERN = re.compile(r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL")
REQUIRED_OPERATIONAL_TABLES = frozenset({"staff", "assignment", "shift_type"})
_development_tokens: dict[int, str] = {}


def _token_cache():
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        if os.environ.get("ATCROSTER_ENVIRONMENT", "development") == "production":
            raise RuntimeError("token_cache_unavailable")
        return None
    import redis

    return redis.from_url(
        redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
        decode_responses=True,
    )


def store_one_time_token(job_id: int, raw_token: str) -> None:
    cache = _token_cache()
    if cache is None:
        _development_tokens[job_id] = raw_token
        return
    cache.set(f"atcroster:provisioning-token:{job_id}", raw_token, ex=3600)


def pop_one_time_token(job_id: int) -> str | None:
    cache = _token_cache()
    if cache is None:
        return _development_tokens.pop(job_id, None)
    key = f"atcroster:provisioning-token:{job_id}"
    pipeline = cache.pipeline(transaction=True)
    pipeline.get(key)
    pipeline.delete(key)
    value, _deleted = pipeline.execute()
    return value


class ProvisioningWorker:
    def __init__(self, application):
        self.app = application
        self.worker_id = hashlib.sha256(
            f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(8)}".encode()
        ).hexdigest()[:32]

    def recover_stale_jobs(self) -> int:
        """Release jobs abandoned by a worker that stopped unexpectedly."""
        with self.app.app_context():
            import app as application

            cutoff = application.utcnow() - timedelta(minutes=15)
            rows = application.ProvisioningJob.query.filter(
                application.ProvisioningJob.state == "running",
                application.ProvisioningJob.locked_at < cutoff,
            ).all()
            for job in rows:
                job.state = "retry_wait"
                job.worker_id = ""
                job.locked_at = None
                job.next_attempt_at = application.utcnow()
                job.last_error_code = "worker_interrupted"
            application.db.session.commit()
            return len(rows)

    def run_once(self) -> bool:
        with self.app.app_context():
            import app as application

            now = application.utcnow()
            job = (
                application.ProvisioningJob.query.filter(
                    application.ProvisioningJob.state.in_(("queued", "retry_wait")),
                    application.ProvisioningJob.next_attempt_at <= now,
                )
                .order_by(application.ProvisioningJob.created_at)
                .with_for_update(skip_locked=True)
                .first()
            )
            if not job:
                return False
            if job.cancel_requested:
                self._finish_cancelled(application, job)
                return True
            job.state = "running"
            job.worker_id = self.worker_id
            job.locked_at = now
            job.attempt_count = int(job.attempt_count or 0) + 1
            job.updated_at = now
            job_id = job.id
            application.db.session.commit()

        self._process(job_id)
        return True

    def _process(self, job_id: int) -> None:
        with self.app.app_context():
            import app as application

            job = application.db.session.get(application.ProvisioningJob, job_id)
            routing = application.db.session.get(
                application.DatabaseRoutingMetadata, job.unit_id
            )
            unit = application.db.session.get(application.Unit, job.unit_id)
            if not routing or not unit:
                self._fail(application, job, "routing_metadata_unavailable", False)
                return
            if job.cancel_requested:
                self._finish_cancelled(application, job)
                return
            secret_name = routing.secret_name or ""
            operational_url = (
                os.environ.get(secret_name)
                if SECRET_NAME_PATTERN.fullmatch(secret_name)
                else None
            )
            if not operational_url:
                self._fail(application, job, "database_secret_unavailable", True)
                return
            control_url = os.environ.get("CONTROL_DATABASE_URL") or os.environ.get(
                "DATABASE_URL", ""
            )
            if control_url and (
                _canonical_database_url(operational_url)
                == _canonical_database_url(control_url)
            ):
                self._fail(application, job, "database_route_conflict", False)
                return
            try:
                version = upgrade_database(operational_url, "operational")
                engine = create_engine(
                    operational_url,
                    pool_pre_ping=True,
                    pool_recycle=280,
                    pool_timeout=10,
                )
                try:
                    with engine.connect() as connection:
                        present = set(inspect(connection).get_table_names())
                    if not REQUIRED_OPERATIONAL_TABLES.issubset(present):
                        raise RuntimeError("operational_schema_incomplete")
                finally:
                    engine.dispose()
                if job.cancel_requested:
                    self._finish_cancelled(application, job)
                    return
                invitation = application.SecureInvitation.query.filter_by(
                    unit_id=unit.id,
                    role="UnitAdmin",
                    active_bootstrap_key="active",
                ).first()
                raw_token = None
                if not invitation:
                    raw_token = secrets.token_urlsafe(32)
                    invitation = application.SecureInvitation(
                        unit_id=unit.id,
                        token_digest=hashlib.sha256(raw_token.encode()).hexdigest(),
                        role="UnitAdmin",
                        active_bootstrap_key="active",
                        expires_at=application.utcnow() + timedelta(days=7),
                    )
                    application.db.session.add(invitation)
                routing.health = "healthy"
                routing.migration_version = version
                routing.provisioning_state = "invitation_issued"
                routing.last_error_code = ""
                routing.attempt_count = job.attempt_count
                routing.ready_at = application.utcnow()
                job.state = "completed"
                job.active_key = None
                job.locked_at = None
                job.worker_id = ""
                job.last_error_code = ""
                job.updated_at = application.utcnow()
                application.db.session.commit()
                if raw_token:
                    store_one_time_token(job.id, raw_token)
            except Exception:
                application.db.session.rollback()
                job = application.db.session.get(application.ProvisioningJob, job_id)
                self._fail(application, job, "database_provisioning_failed", True)

    @staticmethod
    def _finish_cancelled(application, job) -> None:
        job.state = "cancelled"
        job.active_key = None
        job.locked_at = None
        job.worker_id = ""
        job.updated_at = application.utcnow()
        routing = application.db.session.get(
            application.DatabaseRoutingMetadata, job.unit_id
        )
        if routing:
            routing.provisioning_state = "cancelled"
        application.db.session.commit()

    @staticmethod
    def _fail(application, job, code: str, retryable: bool) -> None:
        job.last_error_code = code
        job.locked_at = None
        job.worker_id = ""
        job.updated_at = application.utcnow()
        routing = application.db.session.get(
            application.DatabaseRoutingMetadata, job.unit_id
        )
        if retryable and job.attempt_count < 5:
            job.state = "retry_wait"
            delay = min(15 * (2 ** max(job.attempt_count - 1, 0)), 900)
            job.next_attempt_at = application.utcnow() + timedelta(seconds=delay)
            state = "retry_wait"
        else:
            job.state = "failed"
            job.active_key = None
            state = "failed"
        if routing:
            routing.provisioning_state = state
            routing.last_error_code = code
            routing.attempt_count = job.attempt_count
            routing.last_attempt_at = application.utcnow()
        application.db.session.commit()
