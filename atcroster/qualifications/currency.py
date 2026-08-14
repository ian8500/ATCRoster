"""Operational-position currency policy and shortfall reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Mapping

from atcroster.clock import as_naive_utc


@dataclass(frozen=True)
class OperationalCurrencyRuntimeDependencies:
    db: Any
    Staff: Any
    PositionEndorsement: Any
    PositionSession: Any
    PositionParticipantRole: Any
    PositionSessionParticipant: Any
    current_unit_id: Callable[[], int]
    settings_snapshot: Callable[[int], Mapping[str, str]]
    save_setting: Callable[[str, str], None]
    live_position_enabled: Callable[[int], bool]
    now: Callable[[], datetime]
    setting_key: str
    defaults: Mapping[str, Any]


def create_operational_currency_runtime_dependencies(
    *, db: Any, operational_models: Any, saas_models: Any, **services: Any
) -> OperationalCurrencyRuntimeDependencies:
    """Bind currency reporting records within the qualifications domain."""
    return OperationalCurrencyRuntimeDependencies(
        db=db,
        Staff=operational_models.Staff,
        PositionEndorsement=saas_models.PositionEndorsement,
        PositionSession=saas_models.PositionSession,
        PositionParticipantRole=saas_models.PositionParticipantRole,
        PositionSessionParticipant=saas_models.PositionSessionParticipant,
        **services,
    )


class OperationalCurrencyRuntime:
    """Own persisted operational-currency policy and shortfall reporting."""

    def __init__(self, dependencies: OperationalCurrencyRuntimeDependencies) -> None:
        self.dependencies = dependencies

    def requirement(self, unit_id: int | None = None) -> dict[str, Any]:
        deps = self.dependencies
        return load_currency_requirement(
            unit_id,
            current_unit_id=deps.current_unit_id,
            settings_snapshot=deps.settings_snapshot,
            setting_key=deps.setting_key,
            defaults=deps.defaults,
        )

    def save_requirement(self, data: dict[str, Any]) -> None:
        requirement = dict(self.dependencies.defaults)
        requirement.update(data)
        self.dependencies.save_setting(
            self.dependencies.setting_key,
            json.dumps(requirement, sort_keys=True),
        )

    def window(
        self, requirement: dict[str, Any], today: date | None = None
    ) -> tuple[date, date]:
        return currency_window(requirement, today or self.dependencies.now().date())

    @staticmethod
    def minutes_between(start: datetime, end: datetime) -> int:
        return minutes_between(start, end)

    def shortfalls(self, unit_id: int) -> dict[str, Any]:
        deps = self.dependencies
        return operational_currency_shortfalls(
            unit_id,
            db=deps.db,
            Staff=deps.Staff,
            PositionEndorsement=deps.PositionEndorsement,
            PositionSession=deps.PositionSession,
            PositionParticipantRole=deps.PositionParticipantRole,
            PositionSessionParticipant=deps.PositionSessionParticipant,
            requirement_for=self.requirement,
            live_position_enabled=deps.live_position_enabled,
            now=deps.now,
        )


def load_currency_requirement(
    unit_id: int | None,
    *,
    current_unit_id: Callable[[], int],
    settings_snapshot: Callable[[int], Mapping[str, str]],
    setting_key: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and bound an airport's operational-time currency policy."""
    resolved_unit_id = int(unit_id or current_unit_id() or 1)
    result = dict(defaults)
    raw = settings_snapshot(resolved_unit_id).get(setting_key, "")
    try:
        saved = json.loads(raw) if raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        saved = {}
    if isinstance(saved, dict):
        result.update({key: saved[key] for key in result if key in saved})
    result["enabled"] = bool(result["enabled"])
    result["period_type"] = (
        "calendar_months"
        if result["period_type"] == "calendar_months"
        else "rolling_days"
    )
    for key, minimum, maximum in (
        ("period_days", 1, 731),
        ("period_months", 1, 24),
        ("hours_per_ue", 0.25, 1000),
        ("ojti_credit_percent", 0, 100),
    ):
        try:
            result[key] = max(minimum, min(maximum, float(result[key])))
        except (TypeError, ValueError):
            result[key] = defaults[key]
    return result


def currency_window(requirement: Mapping[str, Any], today: date) -> tuple[date, date]:
    """Resolve the inclusive reporting window for a currency policy."""
    end_day = today
    if requirement["period_type"] == "calendar_months":
        start_day = end_day.replace(day=1)
        for _ in range(int(requirement["period_months"]) - 1):
            start_day = (
                start_day.replace(year=start_day.year - 1, month=12)
                if start_day.month == 1
                else start_day.replace(month=start_day.month - 1)
            )
    else:
        start_day = end_day - timedelta(days=int(requirement["period_days"]) - 1)
    try:
        configured = date.fromisoformat(str(requirement.get("start_date") or ""))
        start_day = max(start_day, configured)
    except ValueError:
        pass
    return start_day, end_day


def minutes_between(start: datetime, end: datetime) -> int:
    return max(0, round((end - start).total_seconds() / 60))


def operational_currency_shortfalls(
    unit_id: int,
    *,
    db: Any,
    Staff: Any,
    PositionEndorsement: Any,
    PositionSession: Any,
    PositionParticipantRole: Any,
    PositionSessionParticipant: Any,
    requirement_for: Callable[[int], dict[str, Any]],
    live_position_enabled: Callable[[int], bool],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    """Calculate credited operational minutes and currency shortfalls."""
    requirement = requirement_for(unit_id)
    current_time = as_naive_utc(now())
    start_day, end_day = currency_window(requirement, current_time.date())
    if not requirement["enabled"] or not live_position_enabled(unit_id):
        return {
            "enabled": False,
            "start_day": start_day,
            "end_day": end_day,
            "rows": [],
        }
    people = (
        Staff.query.filter_by(
            unit_id=unit_id,
            membership_status="active",
            is_operational=True,
        )
        .filter(Staff.role != "position_monitor")
        .order_by(Staff.name)
        .all()
    )
    today = current_time.date()
    endorsement_counts = {
        person_id: count
        for person_id, count in db.session.query(
            PositionEndorsement.person_id,
            db.func.count(PositionEndorsement.id),
        )
        .filter(
            PositionEndorsement.unit_id == unit_id,
            PositionEndorsement.status == "valid",
            PositionEndorsement.valid_from <= today,
            db.or_(
                PositionEndorsement.valid_until.is_(None),
                PositionEndorsement.valid_until >= today,
            ),
        )
        .group_by(PositionEndorsement.person_id)
        .all()
    }
    range_start = datetime.combine(start_day, time.min)
    range_end = datetime.combine(end_day + timedelta(days=1), time.min)
    sessions = PositionSession.query.filter(
        PositionSession.unit_id == unit_id,
        PositionSession.is_void.is_(False),
        PositionSession.started_at < range_end,
        db.or_(
            PositionSession.ended_at.is_(None),
            PositionSession.ended_at > range_start,
        ),
    ).all()
    session_ids = [session.id for session in sessions]
    ojti_role_ids = {
        row.id
        for row in PositionParticipantRole.query.filter_by(unit_id=unit_id)
        .filter(PositionParticipantRole.code == "ojti")
        .all()
    }
    participants = (
        PositionSessionParticipant.query.filter(
            PositionSessionParticipant.unit_id == unit_id,
            PositionSessionParticipant.session_id.in_(session_ids),
            PositionSessionParticipant.role_id.in_(ojti_role_ids),
        ).all()
        if session_ids and ojti_role_ids
        else []
    )
    credited_minutes: dict[int, dict[str, int]] = {}
    for session in sessions:
        start = max(as_naive_utc(session.started_at), range_start)
        session_end = (
            as_naive_utc(session.ended_at) if session.ended_at else current_time
        )
        end = min(session_end, range_end)
        if end > start:
            credit = credited_minutes.setdefault(
                session.primary_person_id, {"operational": 0, "ojti": 0}
            )
            credit["operational"] += minutes_between(start, end)
    for participant in participants:
        start = max(as_naive_utc(participant.started_at), range_start)
        participant_end = (
            as_naive_utc(participant.ended_at)
            if participant.ended_at
            else current_time
        )
        end = min(participant_end, range_end)
        if end > start:
            credit = credited_minutes.setdefault(
                participant.person_id, {"operational": 0, "ojti": 0}
            )
            credit["ojti"] += minutes_between(start, end)

    rows = []
    for person in people:
        legacy_count = sum(
            bool(expiry and expiry >= today)
            for expiry in (
                person.tower_ue_expiry,
                person.radar_ue_expiry,
                person.met_ue_expiry,
            )
        )
        ue_count = endorsement_counts.get(person.id, legacy_count)
        if not ue_count:
            continue
        target = round(float(requirement["hours_per_ue"]) * 60 * ue_count)
        own = credited_minutes.get(person.id, {"operational": 0, "ojti": 0})
        ojti_cap = round(target * float(requirement["ojti_credit_percent"]) / 100)
        credited_ojti = min(own["ojti"], ojti_cap)
        credited = own["operational"] + credited_ojti
        if credited < target:
            rows.append(
                {
                    "person": person,
                    "ue_count": ue_count,
                    "target_minutes": target,
                    "operational_minutes": own["operational"],
                    "ojti_minutes": own["ojti"],
                    "credited_ojti_minutes": credited_ojti,
                    "credited_minutes": credited,
                    "shortfall_minutes": target - credited,
                }
            )
    return {
        "enabled": True,
        "start_day": start_day,
        "end_day": end_day,
        "rows": rows,
        "requirement": requirement,
    }
