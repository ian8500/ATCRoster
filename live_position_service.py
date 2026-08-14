"""Transactional domain operations for Live Position Monitoring.

Routes deliberately stay thin: all linked position/session/audit mutations are
performed here using one authoritative timestamp and one transaction key.
"""

from __future__ import annotations

import json
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy.exc import IntegrityError


class LivePositionConflict(RuntimeError):
    """The requested state transition conflicts with current live state."""


class LivePositionValidationError(ValueError):
    """The requested state transition is not operationally valid."""


@dataclass(frozen=True)
class LivePositionModels:
    Staff: Any
    OperationalPosition: Any
    PositionStatusEvent: Any
    PositionSession: Any
    PositionSessionParticipant: Any
    PositionParticipantRole: Any
    PositionSessionAudit: Any


class LivePositionService:
    def __init__(
        self,
        db: Any,
        models: LivePositionModels,
        now: Callable[[], datetime],
    ) -> None:
        self.db = db
        self.models = models
        self.now = now

    @staticmethod
    def transaction_key(value: str | None = None) -> str:
        value = (value or "").strip()
        return value[:64] if value else secrets.token_hex(24)

    @staticmethod
    def related_key(key: str, suffix: str) -> str:
        return hashlib.sha256(f"{key}:{suffix}".encode()).hexdigest()

    def _verified_actor(self, unit_id: int, actor_id: int) -> Any:
        actor = self.models.Staff.query.filter_by(
            id=actor_id,
            unit_id=unit_id,
            membership_status="active",
            role="position_monitor",
        ).first()
        if not actor:
            raise LivePositionValidationError(
                "An active kiosk account for this unit is required."
            )
        return actor

    def _participant_role(self, unit_id: int, role_id: int) -> Any:
        role = self.models.PositionParticipantRole.query.filter_by(
            id=role_id, unit_id=unit_id, is_active=True, is_primary=False
        ).first()
        if not role:
            raise LivePositionValidationError(
                "Unknown or inactive supporting participant role."
            )
        return role

    def _position_for_update(self, unit_id: int, position_id: int) -> Any:
        position = (
            self.models.OperationalPosition.query.filter_by(
                id=position_id, unit_id=unit_id, is_active=True
            )
            .with_for_update()
            .first()
        )
        if not position:
            raise LivePositionValidationError("Unknown or inactive position.")
        return position

    def _open_session(self, unit_id: int, position_id: int) -> Any | None:
        return (
            self.models.PositionSession.query.filter_by(
                unit_id=unit_id,
                position_id=position_id,
                ended_at=None,
                is_void=False,
            )
            .with_for_update()
            .first()
        )

    def _lock_available_person(self, unit_id: int, person_id: int) -> None:
        person = (
            self.models.Staff.query.filter_by(
                id=person_id,
                unit_id=unit_id,
                membership_status="active",
                is_operational=True,
            )
            .with_for_update()
            .first()
        )
        if not person:
            raise LivePositionValidationError("Unknown controller.")
        primary = self.models.PositionSession.query.filter_by(
            unit_id=unit_id,
            primary_person_id=person_id,
            ended_at=None,
            is_void=False,
        ).first()
        supporting = self.models.PositionSessionParticipant.query.filter_by(
            unit_id=unit_id, person_id=person_id, ended_at=None
        ).first()
        if primary or supporting:
            raise LivePositionConflict(
                f"{person.name} is already logged on to another position."
            )

    def _latest_status(self, unit_id: int, position_id: int) -> str:
        event = (
            self.models.PositionStatusEvent.query.filter_by(
                unit_id=unit_id, position_id=position_id
            )
            .order_by(
                self.models.PositionStatusEvent.occurred_at.desc(),
                self.models.PositionStatusEvent.id.desc(),
            )
            .first()
        )
        return event.status if event else "closed"

    def _participant_session_type(self, unit_id: int, session_id: int) -> str:
        rows = self.models.PositionSessionParticipant.query.filter_by(
            unit_id=unit_id, session_id=session_id, ended_at=None
        ).all()
        codes = {
            role.code
            for row in rows
            if (
                role := self.models.PositionParticipantRole.query.filter_by(
                    id=row.role_id, unit_id=unit_id
                ).first()
            )
        }
        if "assessor" in codes:
            return "assessment"
        if "ojti" in codes:
            return "training"
        return "operational"

    def _audit(
        self,
        *,
        unit_id: int,
        actor_id: int,
        action: str,
        occurred_at: datetime,
        transaction_key: str,
        session_id: int | None = None,
        position_id: int | None = None,
        old: dict[str, Any] | None = None,
        new: dict[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        self.db.session.add(
            self.models.PositionSessionAudit(
                unit_id=unit_id,
                session_id=session_id,
                position_id=position_id,
                actor_id=actor_id,
                action=action,
                occurred_at=occurred_at,
                old_value_json=json.dumps(old or {}, sort_keys=True, default=str),
                new_value_json=json.dumps(new or {}, sort_keys=True, default=str),
                reason=reason,
                transaction_key=transaction_key,
            )
        )

    def set_position_open(
        self,
        *,
        unit_id: int,
        position_id: int,
        actor_id: int,
        open_position: bool,
        reason: str = "",
        request_key: str | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        key = self.transaction_key(request_key)
        existing = self.models.PositionStatusEvent.query.filter_by(
            unit_id=unit_id, transaction_key=key
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
                    self._end_session_records(active, timestamp, "position_closed", key)
                    self._audit(
                        unit_id=unit_id,
                        actor_id=actor_id,
                        action="session_ended",
                        occurred_at=timestamp,
                        transaction_key=key,
                        session_id=active.id,
                        position_id=position_id,
                        new={"ended_at": timestamp, "reason": "position_closed"},
                        reason=reason,
                    )
            event = self.models.PositionStatusEvent(
                unit_id=unit_id,
                position_id=position_id,
                status=target,
                occurred_at=timestamp,
                actor_id=actor_id,
                reason=reason,
                transaction_key=key,
            )
            self.db.session.add(event)
            self._audit(
                unit_id=unit_id,
                actor_id=actor_id,
                action="position_opened" if open_position else "position_closed",
                occurred_at=timestamp,
                transaction_key=key,
                position_id=position_id,
                old={"status": current},
                new={"status": target},
                reason=reason,
            )
            self.db.session.commit()
            return event
        except IntegrityError as error:
            self.db.session.rollback()
            raise LivePositionConflict(
                "Position state changed concurrently."
            ) from error

    def start_session(
        self,
        *,
        unit_id: int,
        position_id: int,
        person_id: int,
        actor_id: int,
        session_type: str = "operational",
        currency_category_id: int | None = None,
        maximum_duration_seconds: int | None = None,
        warning_threshold_seconds: int | None = None,
        due_off_at: datetime | None = None,
        request_key: str | None = None,
        participants: list[dict[str, int]] | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        if session_type not in {"operational", "training", "assessment", "supervised"}:
            raise LivePositionValidationError("Unsupported session type.")
        key = self.transaction_key(request_key)
        existing = self.models.PositionSession.query.filter_by(
            unit_id=unit_id, transaction_key=key
        ).first()
        if existing:
            return existing
        timestamp = self.now()
        try:
            position = self._position_for_update(unit_id, position_id)
            position_was_closed = self._latest_status(unit_id, position_id) != "open"
            if self._open_session(unit_id, position_id):
                raise LivePositionConflict("The position is already occupied.")
            participant_ids = [int(item["person_id"]) for item in (participants or [])]
            for participant in participants or []:
                self._participant_role(unit_id, int(participant["role_id"]))
            if person_id in participant_ids or len(participant_ids) != len(
                set(participant_ids)
            ):
                raise LivePositionValidationError(
                    "Primary and secondary controllers must be different people."
                )
            for active_person_id in sorted([person_id, *participant_ids]):
                self._lock_available_person(unit_id, active_person_id)
            if participant_ids and not position.supporting_participants_allowed:
                raise LivePositionValidationError(
                    "Secondary controllers are not permitted on this position."
                )
            if (
                len(participant_ids) > 1
                and not position.multiple_supporting_participants_allowed
            ):
                raise LivePositionValidationError(
                    "This position permits only one secondary controller."
                )
            if session_type == "training" and not position.training_supported:
                raise LivePositionValidationError("Training is not supported here.")
            if session_type == "assessment" and not position.assessment_supported:
                raise LivePositionValidationError("Assessment is not supported here.")
            effective_maximum_duration_seconds = (
                maximum_duration_seconds
                if maximum_duration_seconds is not None
                else position.maximum_session_duration_minutes * 60
            )
            effective_due_off_at = due_off_at or (
                timestamp + timedelta(seconds=effective_maximum_duration_seconds)
            )
            if position_was_closed:
                open_key = self.related_key(key, "position-opened")
                self.db.session.add(
                    self.models.PositionStatusEvent(
                        unit_id=unit_id,
                        position_id=position_id,
                        status="open",
                        occurred_at=timestamp,
                        actor_id=actor_id,
                        reason="Opened automatically on controller logon",
                        transaction_key=open_key,
                    )
                )
                self._audit(
                    unit_id=unit_id,
                    actor_id=actor_id,
                    action="position_opened",
                    occurred_at=timestamp,
                    transaction_key=open_key,
                    position_id=position_id,
                    old={"status": "closed"},
                    new={"status": "open"},
                    reason="Opened automatically on controller logon",
                )
            session = self.models.PositionSession(
                unit_id=unit_id,
                position_id=position_id,
                primary_person_id=person_id,
                session_type=session_type,
                started_at=timestamp,
                currency_category_id=(
                    currency_category_id or position.currency_category_id
                ),
                maximum_duration_seconds=effective_maximum_duration_seconds,
                warning_threshold_seconds=warning_threshold_seconds,
                due_off_at=effective_due_off_at,
                created_by_id=actor_id,
                transaction_key=key,
            )
            self.db.session.add(session)
            self.db.session.flush()
            for participant in participants or []:
                self.db.session.add(
                    self.models.PositionSessionParticipant(
                        unit_id=unit_id,
                        session_id=session.id,
                        person_id=participant["person_id"],
                        role_id=participant["role_id"],
                        started_at=timestamp,
                        transaction_key=self.related_key(
                            key, f"participant:{participant['person_id']}"
                        ),
                    )
                )
            self._audit(
                unit_id=unit_id,
                actor_id=actor_id,
                action="session_started",
                occurred_at=timestamp,
                transaction_key=key,
                session_id=session.id,
                position_id=position_id,
                new={
                    "person_id": person_id,
                    "session_type": session_type,
                    "started_at": timestamp,
                },
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
            self.models.PositionSessionParticipant.query.filter_by(
                session_id=session.id, unit_id=session.unit_id, ended_at=None
            )
            .with_for_update()
            .all()
        )
        for participant in participants:
            participant.ended_at = timestamp
            participant.ended_reason = reason
        session.ended_at = timestamp
        session.ended_reason = reason
        session.version += 1

    def end_session(
        self,
        *,
        unit_id: int,
        position_id: int,
        actor_id: int,
        reason: str = "logoff",
        request_key: str | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        key = self.transaction_key(request_key)
        audit = self.models.PositionSessionAudit.query.filter_by(
            unit_id=unit_id, transaction_key=key, action="session_ended"
        ).first()
        if audit and audit.session_id:
            return self.models.PositionSession.query.filter_by(
                id=audit.session_id, unit_id=unit_id
            ).first()
        timestamp = self.now()
        self._position_for_update(unit_id, position_id)
        session = self._open_session(unit_id, position_id)
        if not session:
            raise LivePositionConflict("The position is not occupied.")
        self._end_session_records(session, timestamp, reason, key)
        self._audit(
            unit_id=unit_id,
            actor_id=actor_id,
            action="session_ended",
            occurred_at=timestamp,
            transaction_key=key,
            session_id=session.id,
            position_id=position_id,
            new={"ended_at": timestamp, "reason": reason},
        )
        self.db.session.commit()
        return session

    def add_participant(
        self,
        *,
        unit_id: int,
        position_id: int,
        person_id: int,
        role_id: int,
        actor_id: int,
        request_key: str | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        self._participant_role(unit_id, role_id)
        key = self.transaction_key(request_key)
        existing = self.models.PositionSessionParticipant.query.filter_by(
            unit_id=unit_id, transaction_key=key
        ).first()
        if existing:
            return existing
        timestamp = self.now()
        position = self._position_for_update(unit_id, position_id)
        if not position.supporting_participants_allowed:
            raise LivePositionValidationError(
                "Supporting participants are not permitted on this position."
            )
        session = self._open_session(unit_id, position_id)
        if not session:
            raise LivePositionConflict("The position is not occupied.")
        active = (
            self.models.PositionSessionParticipant.query.filter_by(
                unit_id=unit_id, session_id=session.id, ended_at=None
            )
            .with_for_update()
            .all()
        )
        if active and not position.multiple_supporting_participants_allowed:
            raise LivePositionConflict(
                "This position permits only one supporting participant."
            )
        if any(row.person_id == person_id for row in active):
            raise LivePositionConflict("This participant is already logged on.")
        self._lock_available_person(unit_id, person_id)
        participant = self.models.PositionSessionParticipant(
            unit_id=unit_id,
            session_id=session.id,
            person_id=person_id,
            role_id=role_id,
            started_at=timestamp,
            transaction_key=key,
        )
        self.db.session.add(participant)
        self.db.session.flush()
        session.session_type = self._participant_session_type(unit_id, session.id)
        self._audit(
            unit_id=unit_id,
            actor_id=actor_id,
            action="participant_added",
            occurred_at=timestamp,
            transaction_key=key,
            session_id=session.id,
            position_id=position_id,
            new={
                "participant_id": participant.id,
                "person_id": person_id,
                "role_id": role_id,
                "started_at": timestamp,
            },
        )
        try:
            self.db.session.commit()
            return participant
        except IntegrityError as error:
            self.db.session.rollback()
            raise LivePositionConflict(
                "The participant became active elsewhere concurrently."
            ) from error

    def end_participant(
        self,
        *,
        unit_id: int,
        position_id: int,
        participant_id: int,
        actor_id: int,
        request_key: str | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        key = self.transaction_key(request_key)
        prior = self.models.PositionSessionAudit.query.filter_by(
            unit_id=unit_id,
            transaction_key=key,
            action="participant_removed",
        ).first()
        if prior:
            return self.models.PositionSessionParticipant.query.filter_by(
                id=participant_id, unit_id=unit_id
            ).first()
        timestamp = self.now()
        session = self._open_session(unit_id, position_id)
        if not session:
            raise LivePositionConflict("The position is not occupied.")
        participant = (
            self.models.PositionSessionParticipant.query.filter_by(
                id=participant_id,
                unit_id=unit_id,
                session_id=session.id,
                ended_at=None,
            )
            .with_for_update()
            .first()
        )
        if not participant:
            raise LivePositionConflict("The participant is no longer active.")
        role = self.models.PositionParticipantRole.query.filter_by(
            id=participant.role_id, unit_id=unit_id
        ).first()
        primary = self.models.Staff.query.filter_by(
            id=session.primary_person_id, unit_id=unit_id
        ).first()
        if role and role.code == "ojti" and primary and primary.is_trainee:
            remaining_ojti = (
                self.models.PositionSessionParticipant.query.join(
                    self.models.PositionParticipantRole,
                    self.models.PositionSessionParticipant.role_id
                    == self.models.PositionParticipantRole.id,
                )
                .filter(
                    self.models.PositionSessionParticipant.unit_id == unit_id,
                    self.models.PositionSessionParticipant.session_id == session.id,
                    self.models.PositionSessionParticipant.id != participant.id,
                    self.models.PositionSessionParticipant.ended_at.is_(None),
                    self.models.PositionParticipantRole.code == "ojti",
                )
                .first()
            )
            if not remaining_ojti:
                raise LivePositionValidationError(
                    "An OJTI cannot be removed while the trainee remains logged on."
                )
        participant.ended_at = timestamp
        participant.ended_reason = "logoff"
        session.session_type = self._participant_session_type(unit_id, session.id)
        self._audit(
            unit_id=unit_id,
            actor_id=actor_id,
            action="participant_removed",
            occurred_at=timestamp,
            transaction_key=key,
            session_id=session.id,
            position_id=position_id,
            old={"participant_id": participant.id, "ended_at": None},
            new={"ended_at": timestamp},
        )
        self.db.session.commit()
        return participant

    def handover(
        self,
        *,
        unit_id: int,
        position_id: int,
        incoming_person_id: int,
        actor_id: int,
        session_type: str = "operational",
        maximum_duration_seconds: int | None = None,
        request_key: str | None = None,
        participants: list[dict[str, int]] | None = None,
    ) -> Any:
        self._verified_actor(unit_id, actor_id)
        key = self.transaction_key(request_key)
        existing = self.models.PositionSession.query.filter_by(
            unit_id=unit_id,
            transaction_key=self.related_key(key, "incoming"),
        ).first()
        if existing:
            return existing
        timestamp = self.now()
        position = self._position_for_update(unit_id, position_id)
        outgoing = self._open_session(unit_id, position_id)
        if not outgoing:
            raise LivePositionConflict("The position is not occupied.")
        participant_ids = [int(item["person_id"]) for item in (participants or [])]
        for participant in participants or []:
            self._participant_role(unit_id, int(participant["role_id"]))
        if incoming_person_id in participant_ids or len(participant_ids) != len(
            set(participant_ids)
        ):
            raise LivePositionValidationError(
                "Primary and secondary controllers must be different people."
            )
        for active_person_id in sorted([incoming_person_id, *participant_ids]):
            self._lock_available_person(unit_id, active_person_id)
        if participant_ids and not position.supporting_participants_allowed:
            raise LivePositionValidationError(
                "Secondary controllers are not permitted on this position."
            )
        if session_type == "training" and not position.training_supported:
            raise LivePositionValidationError("Training is not supported here.")
        if session_type == "assessment" and not position.assessment_supported:
            raise LivePositionValidationError("Assessment is not supported here.")
        self._end_session_records(outgoing, timestamp, "handover", key)
        effective_maximum_duration_seconds = (
            maximum_duration_seconds
            if maximum_duration_seconds is not None
            else position.maximum_session_duration_minutes * 60
        )
        incoming = self.models.PositionSession(
            unit_id=unit_id,
            position_id=position_id,
            primary_person_id=incoming_person_id,
            session_type=session_type,
            started_at=timestamp,
            currency_category_id=position.currency_category_id,
            maximum_duration_seconds=effective_maximum_duration_seconds,
            due_off_at=timestamp
            + timedelta(seconds=effective_maximum_duration_seconds),
            created_by_id=actor_id,
            transaction_key=self.related_key(key, "incoming"),
        )
        self.db.session.add(incoming)
        self.db.session.flush()
        for participant in participants or []:
            self.db.session.add(
                self.models.PositionSessionParticipant(
                    unit_id=unit_id,
                    session_id=incoming.id,
                    person_id=participant["person_id"],
                    role_id=participant["role_id"],
                    started_at=timestamp,
                    transaction_key=self.related_key(
                        key, f"incoming-participant:{participant['person_id']}"
                    ),
                )
            )
        self._audit(
            unit_id=unit_id,
            actor_id=actor_id,
            action="handover",
            occurred_at=timestamp,
            transaction_key=key,
            session_id=incoming.id,
            position_id=position_id,
            old={"session_id": outgoing.id, "person_id": outgoing.primary_person_id},
            new={"session_id": incoming.id, "person_id": incoming_person_id},
        )
        try:
            self.db.session.commit()
            return incoming
        except IntegrityError as error:
            self.db.session.rollback()
            raise LivePositionConflict(
                "The handover conflicted with another live action."
            ) from error
