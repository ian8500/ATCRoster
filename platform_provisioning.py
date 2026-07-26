"""Durable, idempotent airport database provisioning.

The HTTP application only queues jobs. This module is run by a separate
worker process and deliberately records only privacy-safe state and error
codes in the control database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import threading
from contextlib import contextmanager
from datetime import timedelta

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine, inspect

from scripts.migrate_all_databases import (
    _canonical_database_url,
    upgrade_database,
)

ACTIVE_STATES = ("queued", "running", "retry_wait")
SECRET_NAME_PATTERN = re.compile(r"ATCROSTER_UNIT_[1-9][0-9]*_DATABASE_URL")
REQUIRED_OPERATIONAL_TABLES = frozenset({"staff", "assignment", "shift_type"})
_development_tokens: dict[int, str] = {}


class TokenEnvelopeError(RuntimeError):
    pass


def _token_keys() -> list[tuple[str, Fernet]]:
    configured = os.environ.get("ATCROSTER_TOKEN_ENCRYPTION_KEYS", "")
    if not configured:
        if os.environ.get("ATCROSTER_ENVIRONMENT", "development") == "production":
            raise TokenEnvelopeError("token_encryption_key_unavailable")
        secret = os.environ.get("FLASK_SECRET_KEY", "development-only")
        import base64

        key = base64.urlsafe_b64encode(
            hashlib.sha256(secret.encode()).digest()
        ).decode()
        configured = f"dev:{key}"
    result = []
    for item in configured.split(","):
        version, separator, key = item.strip().partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", version):
            raise TokenEnvelopeError("token_encryption_key_invalid")
        try:
            result.append((version, Fernet(key.encode())))
        except (TypeError, ValueError) as exc:
            raise TokenEnvelopeError("token_encryption_key_invalid") from exc
    if not result:
        raise TokenEnvelopeError("token_encryption_key_unavailable")
    return result


def validate_token_encryption_config() -> None:
    _token_keys()


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


def store_one_time_token(job_id: int, unit_id: int, raw_token: str) -> None:
    version, cipher = _token_keys()[0]
    payload = json.dumps(
        {"job_id": job_id, "unit_id": unit_id, "token": raw_token},
        separators=(",", ":"),
    ).encode()
    ciphertext = f"{version}.{cipher.encrypt(payload).decode()}"
    cache = _token_cache()
    if cache is None:
        _development_tokens[job_id] = ciphertext
        return
    ttl = max(60, int(os.environ.get("ATCROSTER_BOOTSTRAP_TOKEN_TTL_SECONDS", "900")))
    if not cache.set(f"atcroster:provisioning-token:{job_id}", ciphertext, ex=ttl):
        raise TokenEnvelopeError("token_envelope_store_failed")


def pop_one_time_token(job_id: int, unit_id: int) -> str | None:
    cache = _token_cache()
    if cache is None:
        value = _development_tokens.pop(job_id, None)
    else:
        value = cache.getdel(f"atcroster:provisioning-token:{job_id}")
    if not value:
        return None
    version, separator, ciphertext = value.partition(".")
    if not separator:
        raise TokenEnvelopeError("token_envelope_invalid")
    keys = dict(_token_keys())
    cipher = keys.get(version)
    if not cipher:
        raise TokenEnvelopeError("token_encryption_version_unknown")
    try:
        payload = json.loads(cipher.decrypt(ciphertext.encode()).decode())
    except (InvalidToken, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise TokenEnvelopeError("token_envelope_invalid") from exc
    if payload.get("job_id") != job_id or payload.get("unit_id") != unit_id:
        raise TokenEnvelopeError("token_envelope_binding_invalid")
    return str(payload["token"])


class ProvisioningWorker:
    def __init__(self, application):
        self.app = application
        self.worker_id = hashlib.sha256(
            f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(8)}".encode()
        ).hexdigest()[:32]
        self.lease_seconds = max(
            30, int(os.environ.get("ATCROSTER_PROVISIONING_LEASE_SECONDS", "120"))
        )

    def heartbeat(self, state: str = "idle") -> None:
        with self.app.app_context():
            import app as application

            row = application.db.session.get(
                application.WorkerHeartbeat, self.worker_id
            )
            if not row:
                row = application.WorkerHeartbeat(worker_id=self.worker_id)
                application.db.session.add(row)
            row.state = state[:30]
            row.last_seen_at = application.utcnow()
            application.db.session.commit()

    def recover_stale_jobs(self) -> int:
        """Release jobs abandoned by a worker that stopped unexpectedly."""
        with self.app.app_context():
            import app as application

            rows = application.ProvisioningJob.query.filter(
                application.ProvisioningJob.state == "running",
                application.ProvisioningJob.lease_expires_at.is_not(None),
                application.ProvisioningJob.lease_expires_at < application.utcnow(),
            ).all()
            for job in rows:
                job.state = "retry_wait"
                job.worker_id = ""
                job.lease_owner = ""
                job.lease_expires_at = None
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
            job.lease_owner = self.worker_id
            job.lease_expires_at = now + timedelta(seconds=self.lease_seconds)
            job.locked_at = now
            job.attempt_count = int(job.attempt_count or 0) + 1
            job.updated_at = now
            job_id = job.id
            application.db.session.commit()

        self.heartbeat("busy")
        try:
            self._process(job_id)
        finally:
            self.heartbeat("idle")
        return True

    def _process(self, job_id: int) -> None:
        stop_renewal = threading.Event()
        renewal = None
        with self.app.app_context():
            import app as application

            job = application.db.session.get(application.ProvisioningJob, job_id)
            if not self._owns_lease(job):
                return
            if application.db.engine.dialect.name == "postgresql":
                renewal = threading.Thread(
                    target=self._renew_lease_until,
                    args=(job_id, stop_renewal),
                    daemon=True,
                )
                renewal.start()
            try:
                with self._airport_advisory_lock(application, job.unit_id) as acquired:
                    if not acquired:
                        self._defer_lock_contention(application, job)
                        return
                    self._process_with_lock(application, job)
            finally:
                stop_renewal.set()
                if renewal:
                    renewal.join(timeout=5)

    def _process_with_lock(self, application, job) -> None:
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
            application.db.session.expire(job)
            if not self._owns_lease(job):
                return
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
            job.lease_owner = ""
            job.lease_expires_at = None
            job.last_error_code = ""
            job.updated_at = application.utcnow()
            if raw_token:
                store_one_time_token(job.id, unit.id, raw_token)
            application.db.session.commit()
        except Exception:
            application.db.session.rollback()
            job = application.db.session.get(application.ProvisioningJob, job.id)
            if self._owns_lease(job):
                self._fail(application, job, "database_provisioning_failed", True)

    def _renew_lease_until(self, job_id: int, stop: threading.Event) -> None:
        interval = max(5, self.lease_seconds // 3)
        while not stop.wait(interval):
            with self.app.app_context():
                import app as application

                updated = application.ProvisioningJob.query.filter_by(
                    id=job_id, state="running", lease_owner=self.worker_id
                ).update(
                    {
                        "lease_expires_at": (
                            application.utcnow() + timedelta(seconds=self.lease_seconds)
                        ),
                        "updated_at": application.utcnow(),
                    }
                )
                if updated != 1:
                    application.db.session.rollback()
                    return
                application.db.session.commit()
            self.heartbeat("busy")

    def _owns_lease(self, job) -> bool:
        return bool(
            job
            and job.state == "running"
            and job.lease_owner == self.worker_id
            and job.lease_expires_at
        )

    @contextmanager
    def _airport_advisory_lock(self, application, unit_id: int):
        if application.db.engine.dialect.name != "postgresql":
            yield True
            return
        with application.db.engine.connect() as connection:
            acquired = bool(
                connection.execute(
                    application.text("SELECT pg_try_advisory_lock(:key)"),
                    {"key": int(unit_id)},
                ).scalar()
            )
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute(
                        application.text("SELECT pg_advisory_unlock(:key)"),
                        {"key": int(unit_id)},
                    )

    def _defer_lock_contention(self, application, job) -> None:
        if not self._owns_lease(job):
            return
        job.state = "retry_wait"
        job.worker_id = ""
        job.lease_owner = ""
        job.lease_expires_at = None
        job.locked_at = None
        job.next_attempt_at = application.utcnow() + timedelta(seconds=15)
        job.last_error_code = "airport_lock_busy"
        application.db.session.commit()

    @staticmethod
    def _finish_cancelled(application, job) -> None:
        job.state = "cancelled"
        job.active_key = None
        job.locked_at = None
        job.worker_id = ""
        job.lease_owner = ""
        job.lease_expires_at = None
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
        job.lease_owner = ""
        job.lease_expires_at = None
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
