"""Legacy migration and local-data bootstrap facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atcroster.roster.bootstrap import (
    ensure_shift as ensure_bootstrap_shift,
    ensure_watch as ensure_bootstrap_watch,
    seed_legacy_operational_data,
)

from .legacy_migrations import (
    add_assignment_annotation,
    add_columns_if_missing,
    add_invitation_target,
    add_performance_indexes,
    add_role_and_calendar_token,
    add_toil_and_leave_fields,
    add_unique_assignment_key,
    add_watch_pattern_configuration,
    upgrade_tenant_foundation,
)


@dataclass(frozen=True)
class LegacyBootstrapService:
    db: Any
    app: Any
    Unit: Any
    Staff: Any
    Watch: Any
    ShiftType: Any

    def migrate_tenant_foundation(self):
        return upgrade_tenant_foundation(db=self.db, Unit=self.Unit)

    def add_role_and_calendar_token(self):
        return add_role_and_calendar_token(db=self.db, Staff=self.Staff)

    def add_assignment_annotation(self):
        return add_assignment_annotation(db=self.db)

    def add_unique_assignment_key(self):
        return add_unique_assignment_key(db=self.db)

    def add_performance_indexes(self):
        return add_performance_indexes(db=self.db, app=self.app)

    def add_requirement_day_column(self):
        return add_columns_if_missing(
            db=self.db,
            table="requirement",
            columns={"req_d": "req_d INTEGER DEFAULT 0"},
        )

    def add_undertraining_flags(self):
        return add_columns_if_missing(
            db=self.db,
            table="staff",
            columns={
                "tower_ut": "tower_ut BOOLEAN DEFAULT 0",
                "radar_ut": "radar_ut BOOLEAN DEFAULT 0",
            },
        )

    def add_training_shift_flag(self):
        return add_columns_if_missing(
            db=self.db,
            table="shift_type",
            columns={"is_training": "is_training BOOLEAN DEFAULT 0"},
        )

    def add_workforce_flags(self):
        return add_columns_if_missing(
            db=self.db,
            table="staff",
            columns={
                "is_wm": "is_wm BOOLEAN DEFAULT 0",
                "is_dwm": "is_dwm BOOLEAN DEFAULT 0",
                "exclude_from_ot": "exclude_from_ot BOOLEAN DEFAULT 0",
            },
        )

    def add_phone_number(self):
        return add_columns_if_missing(
            db=self.db,
            table="staff",
            columns={"phone_number": "phone_number VARCHAR(30) DEFAULT ''"},
        )

    def add_watch_pattern_configuration(self):
        return add_watch_pattern_configuration(db=self.db)

    def add_invitation_target(self):
        return add_invitation_target(db=self.db)

    def add_toil_and_leave_fields(self):
        return add_toil_and_leave_fields(db=self.db)

    def ensure_shift(
        self,
        code,
        name,
        start=None,
        end=None,
        is_working=False,
        is_training=False,
    ):
        return ensure_bootstrap_shift(
            code,
            name,
            db=self.db,
            ShiftType=self.ShiftType,
            start=start,
            end=end,
            is_working=is_working,
            is_training=is_training,
        )

    def ensure_watch(self, name: str, order_index: int):
        return ensure_bootstrap_watch(name, order_index, db=self.db, Watch=self.Watch)

    def seed_once(self):
        return seed_legacy_operational_data(
            db=self.db,
            Unit=self.Unit,
            Watch=self.Watch,
            ShiftType=self.ShiftType,
            Staff=self.Staff,
        )
