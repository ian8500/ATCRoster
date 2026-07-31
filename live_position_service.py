"""Transactional domain operations for Live Position Monitoring.

Routes deliberately stay thin: all linked position/session/audit mutations are
performed here using one authoritative timestamp and one transaction key.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from sqlalchemy.exc import IntegrityError


class LivePositionConflict(RuntimeError):
    """The requested state transition conflicts with current live state."""


class LivePositionValidationError(ValueError):
    """The requested state transition is not operationally valid."""


@dataclass(frozen=True)
class LivePositionModels:
    OperationalPosition: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    PositionSessionAudit: Any


class LivePositionService:
    def __init__(
        self, db: Any, models: LivePositionModels,
        now: Callable[[], datetime],
    ) -> None:
        self.db = db
        self.models = models
        self.now = now

    @staticmethod
    def transaction_key(value: str | None = None) -> str:
        value = (value or "").strip()
        return value[:64] if value else secrets.token_hex(24)

    def _position_for_update(self, unit_id: int, position_id: int) -> Any:
        position = (
            self.models.OperationalPosition.query
            .filter_by(id=position_id, unit_id=unit_id, is_active=True)
            .with_for_update()
            .first()
        )
        if not position:
            raise LivePositionValidationError("Unknown or inactive position.")
        return position

    def _open_session(self, unit_id: int, position_id: int) -> Any | None:
        return (
            self.models.PositionSession.query
            .filter_by(
                unit_id=unit_id, position_id=position_id,
                ended_at=None, is_void=False,
            )
            .with_for_update()
            .first()
        )

    def _latest_status(self, unit_id: int, position_id: int) -> str:
        event = (
            self.models.PositionStatusEvent.query
            .filter_by(unit_id=unit_id, position_id=position_id)
            .order_by(
                self.models.PositionStatusEvent.occurred_at.desc(),
                self.models.PositionStatusEvent.id.desc(),
            ).first()
        )
        return event.status if event else "closed"

    def _audit(
        self, *, unit_id: int, actor_id: int, action: str,
        occurred_at: datetime, transaction_key: str,
        session_id: int | None = None, position_id: int | None = None,
        old: dict[str, Any] | None = None, new: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        self.db.session.add(self.models.PositionSessionAudit(
            unit_id=unit_id, session_id=session_id, position_id=position_id,
            actor_id=actor_id, action=action, occurred_at=occurred_at,
            old_value_json=json.dumps(old or {}, sort_keys=True, default=str),
            new_value_json=json.dumps(new or {}, sort_keys=True, default=str),
            reason=reason, transaction_key=transaction_key,
        ))

    def set_position_open(
        self, *, unit_id: int, position_id: int, actor_id: int,
        open_position: bool, reason: str = "", request_key: str | None = None,
    ) -> Any:
        key = self.transaction_key(request_key)
        existing = self.models.PositionStatusEvent.query.filter_by(
            transaction_key=key
        ).first()
        if existing:
            return existing
        timestamp = self.now()
        target = "open" if open_position else "closed"
        try:
            self._position_for_update(unit_id, position_id)
            current = self._latest_status(unit_id, position_id)
            if target == "closed":
                active = self._open_session(unit_id, position_id)
                if active:
                    self._end_session_records(
                        active, timestamp, "position_closed", key
                    )
                    self._audit(
                        unit_id=unit_id, actor_id=actor_id,
                        action="session_ended", occurred_at=timestamp,
                        transaction_key=key, session_id=active.id,
                        position_id=position_id,
                        new={"ended_at": timestamp, "reason": "position_closed"},
                        reason=reason,
                    )
            event = self.models.PositionStatusEvent(
                unit_id=unit_id, position_id=position_id, status=target,
                occurred_at=timestamp, actor_id=actor_id, reason=reason,
                transaction_key=key,
            )
            self.db.session.add(event)
            self._audit(
                unit_id=unit_id, actor_id=actor_id,
                action="position_opened" if open_position else "position_closed",
                occurred_at=timestamp, transaction_key=key,
                position_id=position_id, old={"status": current},
                new={"status": target}, reason=reason,
            )
            self.db.session.commit()
            return event
        except IntegrityError as error:
            self.db.session.rollback()
            raise LivePositionConflict("Position state changed concurrently.") from error

    def start_session(
        self, *, unit_id: int, position_id: int, person_id: int,
        actor_id: int, session_type: str = "operational",
        currency_category_id: int | None = None,
        maximum_duration_seconds: int | None = None,
        warning_threshold_seconds: int | None = None,
        due_off_at: datetime | None = None, request_key: str | None = None,
    ) -> Any:
        if session_type not in {"operational", "training", "assessment", "supervised"}:
            raise LivePositionValidationError("Unsupported session type.")
        key = self.transaction_key(request_key)
        existing = self.models.PositionSession.query.filter_by(
            transaction_key=key
        ).first()
        if existing:
            return existing
        timestamp = self.now()
        try:
            position = self._position_for_update(unit_id, position_id)
            if self._latest_status(unit_id, position_id) != "open":
                raise LivePositionConflict("The position is closed.")
            if self._open_session(unit_id, position_id):
                raise LivePositionConflict("The position is already occupied.")
            if session_type == "training" and not position.training_supported:
                raise LivePositionValidationError("Training is not supported here.")
            if session_type == "assessment" and not position.assessment_supported:
                raise LivePositionValidationError("Assessment is not supported here.")
            session = self.models.PositionSession(
                unit_id=unit_id, position_id=position_id,
                primary_person_id=person_id, session_type=session_type,
                started_at=timestamp, currency_category_id=(
                    currency_category_id or position.currency_category_id
                ), maximum_duration_seconds=maximum_duration_seconds,
                warning_threshold_seconds=warning_threshold_seconds,
                due_off_at=due_off_at, created_by_id=actor_id,
                transaction_key=key,
            )
            self.db.session.add(session)
            self.db.session.flush()
            self._audit(
                unit_id=unit_id, actor_id=actor_id, action="session_started",
                occurred_at=timestamp, transaction_key=key,
                session_id=session.id, position_id=position_id,
                new={"person_id": person_id, "session_type": session_type,
                     "started_at": timestamp},
            )
            self.db.session.commit()
            return session
        except IntegrityError as error:
            self.db.session.rollback()
            raise LivePositionConflict(
                "The position or controller became occupied concurrently."
            ) from error

    def _end_session_records(
        self, session: Any, timestamp: datetime, reason: str, key: str
    ) -> None:
        participants = (
            self.models.PositionSessionParticipant.query
            .filter_by(session_id=session.id, unit_id=session.unit_id, ended_at=None)
            .with_for_update().all()
        )
        for participant in participants:
            participant.ended_at = timestamp
            participant.ended_reason = reason
        session.ended_at = timestamp
        session.ended_reason = reason
        session.version += 1

    def end_session(
        self, *, unit_id: int, position_id: int, actor_id: int,
        reason: str = "logoff", request_key: str | None = None,
    ) -> Any:
        key = self.transaction_key(request_key)
        audit = self.models.PositionSessionAudit.query.filter_by(
            transaction_key=key, action="session_ended"
        ).first()
        if audit and audit.session_id:
            return self.db.session.get(self.models.PositionSession, audit.session_id)
        timestamp = self.now()
        self._position_for_update(unit_id, position_id)
        session = self._open_session(unit_id, position_id)
        if not session:
            raise LivePositionConflict("The position is not occupied.")
        self._end_session_records(session, timestamp, reason, key)
        self._audit(
            unit_id=unit_id, actor_id=actor_id, action="session_ended",
            occurred_at=timestamp, transaction_key=key,
            session_id=session.id, position_id=position_id,
            new={"ended_at": timestamp, "reason": reason},
        )
        self.db.session.commit()
        return session
