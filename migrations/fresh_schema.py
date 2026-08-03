"""Explicit fresh-install schema used by the first Alembic revision."""
import os

import sqlalchemy as sa
from alembic import op as _alembic_op

CONTROL_TABLES = frozenset({
    "unit", "platform_identity", "unit_membership", "secure_invitation",
    "database_routing_metadata", "feature_flag", "plan_history",
    "aggregate_usage_event", "super_admin_audit",
    "platform_mfa_credential", "signup_workflow", "central_security_audit",
    "provisioning_job", "worker_heartbeat",
})


class _RoleFilteredOperations:
    """Filter explicit schema operations for the configured database role."""

    def _allowed(self, table_name):
        role = os.environ.get(
            "ATCROSTER_SCHEMA_ROLE", "combined"
        ).lower()
        if role == "control":
            return table_name in CONTROL_TABLES
        if role == "operational":
            return table_name not in CONTROL_TABLES
        return True

    def create_table(self, table_name, *elements, **kwargs):
        if not self._allowed(table_name):
            return None
        filtered = []
        for element in elements:
            if isinstance(element, sa.Column) and any(
                foreign_key.target_fullname.split(".", 1)[0]
                in CONTROL_TABLES
                for foreign_key in element.foreign_keys
            ):
                element = sa.Column(
                    element.name, element.type,
                    primary_key=element.primary_key,
                    nullable=element.nullable,
                    server_default=element.server_default,
                )
            filtered.append(element)
        return _alembic_op.create_table(table_name, *filtered, **kwargs)

    def create_index(self, name, table_name, *args, **kwargs):
        if not self._allowed(table_name):
            return None
        return _alembic_op.create_index(
            name, table_name, *args, **kwargs
        )


op = _RoleFilteredOperations()


def create_fresh_schema():
    op.create_table('platform_identity',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('public_id', sa.String(64), nullable=False),
        sa.Column('username', sa.String(120), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('mfa_secret_encrypted', sa.String()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_active_at', sa.DateTime()),
        sa.UniqueConstraint('public_id'),
        sa.UniqueConstraint('username'),
    )
    op.create_table('unit',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('code', sa.String(12), nullable=False),
        sa.Column('name', sa.String(120), nullable=False),
        sa.Column('timezone', sa.String(64), nullable=False),
        sa.Column('locale', sa.String(20), nullable=False),
        sa.Column('date_format', sa.String(30), nullable=False),
        sa.Column('branding_json', sa.String(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('plan', sa.String(40), nullable=False),
        sa.Column('request_months_ahead', sa.Integer(), nullable=False),
        sa.Column('request_lock_day', sa.Integer(), nullable=False),
        sa.Column(
            'protected_roster_months_ahead', sa.Integer(), nullable=False,
            server_default='2',
        ),
        sa.Column('active_user_limit', sa.Integer(), nullable=False),
        sa.Column('onboarding_step', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('trial_ends_at', sa.DateTime()),
        sa.Column('renews_at', sa.DateTime()),
        sa.Column('suspended_at', sa.DateTime()),
        sa.Column('last_active_at', sa.DateTime()),
        sa.UniqueConstraint('code'),
    )
    op.create_table('aggregate_usage_event',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('event_type', sa.String(60), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_aggregate_usage_event_unit_id', 'aggregate_usage_event', ['unit_id'], unique=False)
    op.create_table('ai_rule_set',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('month', sa.Integer(), nullable=False),
        sa.Column('rules_json', sa.String(), nullable=False),
        sa.UniqueConstraint('unit_id', 'year', 'month', name='uniq_ai_ruleset_unit_month'),
    )
    op.create_index('ix_ai_rule_set_unit_id', 'ai_rule_set', ['unit_id'], unique=False)
    op.create_table('annotation_type',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('code', sa.String(10), nullable=False),
        sa.Column('label', sa.String(80), nullable=False),
        sa.Column('category', sa.String(40), nullable=False),
        sa.Column('colour', sa.String(20), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('allow_suffix', sa.Boolean()),
        sa.Column('suffixes', sa.String(20)),
        sa.Column('toil_half_days', sa.Integer()),
        sa.Column('tags', sa.String(200)),
        sa.Column('note_required', sa.Boolean()),
        sa.Column('admin_only', sa.Boolean()),
        sa.Column('has_been_used', sa.Boolean()),
        sa.Column('is_active', sa.Boolean()),
        sa.Column('sort_order', sa.Integer()),
        sa.UniqueConstraint('unit_id', 'code', name='uq_annotation_unit_code'),
    )
    op.create_index('ix_annotation_type_unit_id', 'annotation_type', ['unit_id'], unique=False)
    op.create_table('change_log',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('when', sa.DateTime(), nullable=False),
        sa.Column('who_user_id', sa.Integer()),
        sa.Column('entity_type', sa.String(40)),
        sa.Column('entity_id', sa.Integer()),
        sa.Column('field', sa.String(40)),
        sa.Column('old_value', sa.String()),
        sa.Column('new_value', sa.String()),
        sa.Column('context_month', sa.String(7)),
        sa.Column('note', sa.String()),
    )
    op.create_index('ix_change_log_context_month', 'change_log', ['context_month'], unique=False)
    op.create_index('ix_change_log_entity_id', 'change_log', ['entity_id'], unique=False)
    op.create_index('ix_change_log_entity_type', 'change_log', ['entity_type'], unique=False)
    op.create_index('ix_change_log_unit_id', 'change_log', ['unit_id'], unique=False)
    op.create_index('ix_change_log_when', 'change_log', ['when'], unique=False)
    op.create_index('ix_change_log_who_user_id', 'change_log', ['who_user_id'], unique=False)
    op.create_table('database_routing_metadata',
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), primary_key=True),
        sa.Column('secret_name', sa.String(120), nullable=False),
        sa.Column('health', sa.String(20), nullable=False),
        sa.Column('migration_version', sa.String(64), nullable=False),
        sa.Column('storage_bytes', sa.BigInteger(), nullable=False),
    )
    op.create_table('feature_flag',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('key', sa.String(80), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.UniqueConstraint('unit_id', 'key', name='uq_feature_unit_key'),
    )
    op.create_index('ix_feature_flag_unit_id', 'feature_flag', ['unit_id'], unique=False)
    op.create_table('operational_position',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('code', sa.String(30), nullable=False),
        sa.Column('label', sa.String(120), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('is_safety_critical', sa.Boolean(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.UniqueConstraint('unit_id', 'code', name='uq_position_unit_code'),
    )
    op.create_index('ix_operational_position_unit_id', 'operational_position', ['unit_id'], unique=False)
    op.create_table('plan_history',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('plan', sa.String(40), nullable=False),
        sa.Column('active_user_limit', sa.Integer(), nullable=False),
        sa.Column('effective_at', sa.DateTime(), nullable=False),
        sa.Column('changed_by_identity_id', sa.Integer(), sa.ForeignKey('platform_identity.id')),
    )
    op.create_index('ix_plan_history_unit_id', 'plan_history', ['unit_id'], unique=False)
    op.create_table('qualification_type',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('code', sa.String(30), nullable=False),
        sa.Column('label', sa.String(100), nullable=False),
        sa.Column('warning_days_csv', sa.String(100), nullable=False),
        sa.Column('expiry_required', sa.Boolean(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.UniqueConstraint('unit_id', 'code', name='uq_qualification_unit_code'),
    )
    op.create_index('ix_qualification_type_unit_id', 'qualification_type', ['unit_id'], unique=False)
    op.create_table('requirement',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('month', sa.Integer(), nullable=False),
        sa.Column('req_m', sa.Integer()),
        sa.Column('req_d', sa.Integer()),
        sa.Column('req_a', sa.Integer()),
        sa.Column('req_n', sa.Integer()),
        sa.Column('req_sat_m', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sat_d', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sat_a', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sat_n', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sun_m', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sun_d', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sun_a', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('req_sun_n', sa.Integer(), nullable=False, server_default='0'),
        sa.UniqueConstraint('unit_id', 'year', 'month', name='uniq_unit_year_month'),
    )
    op.create_index('ix_requirement_unit_id', 'requirement', ['unit_id'], unique=False)
    op.create_table('special_requirement',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('label', sa.String(80), nullable=False),
        sa.Column('req_m', sa.Integer(), nullable=False),
        sa.Column('req_d', sa.Integer(), nullable=False),
        sa.Column('req_a', sa.Integer(), nullable=False),
        sa.Column('req_n', sa.Integer(), nullable=False),
        sa.UniqueConstraint('unit_id', 'day', name='uniq_unit_special_requirement_day'),
    )
    op.create_index('ix_special_requirement_unit_id', 'special_requirement', ['unit_id'], unique=False)
    op.create_index('ix_special_requirement_day', 'special_requirement', ['day'], unique=False)
    op.create_table('sms_audit',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('sent_at', sa.DateTime(), nullable=False),
        sa.Column('sent_by_staff_id', sa.Integer(), nullable=False),
        sa.Column('sent_by_name', sa.String(80), nullable=False),
        sa.Column('sender_number', sa.String(20), nullable=False),
        sa.Column('recipient_number', sa.String(20), nullable=False),
        sa.Column('recipient_label', sa.String(120), nullable=False),
        sa.Column('message_type', sa.String(30), nullable=False),
        sa.Column('message_content', sa.Text(), nullable=False),
        sa.Column('provider_message_id', sa.String(64), nullable=False),
    )
    op.create_index('ix_sms_audit_unit_id', 'sms_audit', ['unit_id'], unique=False)
    op.create_index('ix_sms_audit_sent_at', 'sms_audit', ['sent_at'], unique=False)
    op.create_index('ix_sms_audit_sent_by_staff_id', 'sms_audit', ['sent_by_staff_id'], unique=False)
    op.create_table('roster_publication',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('month', sa.Integer(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(20), nullable=False),
        sa.Column('snapshot_json', sa.String(), nullable=False),
        sa.Column('published_at', sa.DateTime()),
        sa.Column('superseded_at', sa.DateTime()),
        sa.UniqueConstraint('unit_id', 'year', 'month', 'version', name='uq_roster_publication_version'),
    )
    op.create_index('ix_roster_publication_unit_id', 'roster_publication', ['unit_id'], unique=False)
    op.create_table('roster_rule_version',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(120), nullable=False),
        sa.Column('rules_json', sa.String(), nullable=False),
        sa.Column('state', sa.String(20), nullable=False),
        sa.Column('effective_from', sa.Date()),
        sa.Column('change_reference', sa.String(120), nullable=False),
        sa.Column('consultation_summary', sa.String(), nullable=False),
        sa.Column('approved_by_id', sa.Integer()),
        sa.Column('approved_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('unit_id', 'version', name='uq_roster_rule_unit_version'),
    )
    op.create_index('ix_roster_rule_version_unit_id', 'roster_rule_version', ['unit_id'], unique=False)
    op.create_table('roster_setting',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('key', sa.String(50), nullable=False),
        sa.Column('value', sa.String(), nullable=False),
        sa.UniqueConstraint('unit_id', 'key', name='uq_roster_setting_unit_key'),
    )
    op.create_index('ix_roster_setting_unit_id', 'roster_setting', ['unit_id'], unique=False)
    op.create_table('scenario',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('name', sa.String(120), nullable=False),
        sa.Column('changes_json', sa.String(), nullable=False),
        sa.Column('created_by_id', sa.Integer(), nullable=False),
        sa.Column('approved_by_id', sa.Integer()),
        sa.Column('applied_at', sa.DateTime()),
    )
    op.create_index('ix_scenario_unit_id', 'scenario', ['unit_id'], unique=False)
    op.create_table('secure_invitation',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('token_digest', sa.String(128), nullable=False),
        sa.Column('role', sa.String(30), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('accepted_at', sa.DateTime()),
        sa.Column('disabled_at', sa.DateTime()),
        sa.UniqueConstraint('token_digest'),
    )
    op.create_index('ix_secure_invitation_unit_id', 'secure_invitation', ['unit_id'], unique=False)
    op.create_table('shift_type',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('code', sa.String(10), nullable=False),
        sa.Column('name', sa.String(40), nullable=False),
        sa.Column('start_time', sa.Time()),
        sa.Column('end_time', sa.Time()),
        sa.Column('is_working', sa.Boolean()),
        sa.Column('is_training', sa.Boolean()),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_requestable', sa.Boolean(), nullable=False),
        sa.Column('required_qualification', sa.String(40), nullable=False),
        sa.UniqueConstraint('unit_id', 'code', name='uq_shift_unit_code'),
        sa.UniqueConstraint('unit_id', 'id', name='uq_shift_unit_id'),
    )
    op.create_index('ix_shift_type_unit_id', 'shift_type', ['unit_id'], unique=False)
    op.create_table('super_admin_audit',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('actor_identity_id', sa.Integer(), sa.ForeignKey('platform_identity.id'), nullable=False),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('action', sa.String(80), nullable=False),
        sa.Column('safe_summary', sa.String(500), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
    )
    op.create_table('unit_membership',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('identity_id', sa.Integer(), sa.ForeignKey('platform_identity.id'), nullable=False),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer()),
        sa.Column('role', sa.String(30), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('permissions_json', sa.String(), nullable=False),
        sa.Column('invited_at', sa.DateTime()),
        sa.Column('activated_at', sa.DateTime()),
        sa.Column('suspended_at', sa.DateTime()),
        sa.UniqueConstraint('identity_id', 'unit_id', name='uq_membership_identity_unit'),
    )
    op.create_index('ix_unit_membership_unit_id', 'unit_membership', ['unit_id'], unique=False)
    op.create_table('watch',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('name', sa.String(32), nullable=False),
        sa.Column('order_index', sa.Integer(), nullable=False),
        sa.UniqueConstraint('unit_id', 'name', name='uq_watch_unit_name'),
    )
    op.create_index('ix_watch_unit_id', 'watch', ['unit_id'], unique=False)
    op.create_table('position_requirement',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('shift_code', sa.String(10), nullable=False),
        sa.Column('position_id', sa.Integer(), sa.ForeignKey('operational_position.id'), nullable=False),
        sa.Column('required_count', sa.Integer(), nullable=False),
        sa.Column('contingency_count', sa.Integer(), nullable=False),
        sa.UniqueConstraint('unit_id', 'day', 'shift_code', 'position_id', name='uq_position_requirement_day_shift'),
    )
    op.create_index('ix_position_requirement_day', 'position_requirement', ['day'], unique=False)
    op.create_index('ix_position_requirement_unit_id', 'position_requirement', ['unit_id'], unique=False)
    op.create_table('staff',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('username', sa.String(80), nullable=False),
        sa.Column('password_hash', sa.String(200), nullable=False),
        sa.Column('role', sa.String(10), nullable=False),
        sa.Column('membership_status', sa.String(20), nullable=False),
        sa.Column('permissions_json', sa.String(), nullable=False),
        sa.Column('phone_number', sa.String(30)),
        sa.Column('is_admin', sa.Boolean()),
        sa.Column('calendar_token', sa.String(64)),
        sa.Column('name', sa.String(80), nullable=False),
        sa.Column('staff_no', sa.String(20), nullable=False),
        sa.Column('watch_id', sa.Integer(), sa.ForeignKey('watch.id')),
        sa.Column('medical_expiry', sa.Date()),
        sa.Column('tower_ue_expiry', sa.Date()),
        sa.Column('radar_ue_expiry', sa.Date()),
        sa.Column('tower_ut', sa.Boolean()),
        sa.Column('radar_ut', sa.Boolean()),
        sa.Column('met_ue_expiry', sa.Date()),
        sa.Column('met_ut', sa.Boolean()),
        sa.Column('has_assessor', sa.Boolean()),
        sa.Column('is_operational', sa.Boolean()),
        sa.Column('is_trainee', sa.Boolean()),
        sa.Column('has_ojti', sa.Boolean()),
        sa.Column('is_wm', sa.Boolean()),
        sa.Column('is_dwm', sa.Boolean()),
        sa.Column('exclude_from_ot', sa.Boolean()),
        sa.Column('pattern_csv', sa.String()),
        sa.Column('pattern_anchor', sa.Date()),
        sa.Column('toil_half_days', sa.Integer()),
        sa.Column('leave_year_start_month', sa.Integer()),
        sa.Column('leave_entitlement_days', sa.Integer()),
        sa.Column('leave_public_holidays', sa.Integer()),
        sa.Column('leave_carryover_days', sa.Integer()),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('unit_id', 'staff_no', name='uq_staff_unit_number'),
        sa.UniqueConstraint('unit_id', 'id', name='uq_staff_unit_id'),
        sa.UniqueConstraint('calendar_token'),
    )
    op.create_index('ix_staff_unit_id', 'staff', ['unit_id'], unique=False)
    op.create_table('assignment',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id')),
        sa.Column('day', sa.Date()),
        sa.Column('code', sa.String(10), nullable=False),
        sa.Column('generated_code', sa.String(10)),
        sa.Column('override_code', sa.String(10)),
        sa.Column('generated_from_pattern_id', sa.Integer()),
        sa.Column('generated_from_pattern_day_index', sa.Integer()),
        sa.Column('generated_at', sa.DateTime()),
        sa.Column('generation_event_id', sa.Integer()),
        sa.Column('generation_version', sa.String(40)),
        sa.Column('override_type', sa.String(40)),
        sa.Column('override_reason', sa.String(500), nullable=False, server_default=''),
        sa.Column('override_by_user_id', sa.Integer()),
        sa.Column('override_at', sa.DateTime()),
        sa.Column('source', sa.String(10)),
        sa.Column('note', sa.String(140)),
        sa.Column('annotation', sa.String(20)),
        sa.UniqueConstraint('staff_id', 'day', name='uniq_staff_day'),
    )
    op.create_index('ix_assignment_day', 'assignment', ['day'], unique=False)
    op.create_index('ix_assignment_staff_id', 'assignment', ['staff_id'], unique=False)
    op.create_index('ix_assignment_unit_id', 'assignment', ['unit_id'], unique=False)
    op.create_table('break_plan',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('position_id', sa.Integer(), sa.ForeignKey('operational_position.id')),
        sa.Column('start_time', sa.Time(), nullable=False),
        sa.Column('end_time', sa.Time(), nullable=False),
        sa.Column('kind', sa.String(20), nullable=False),
        sa.Column('state', sa.String(20), nullable=False),
        sa.Column('recorded_by_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_break_plan_day', 'break_plan', ['day'], unique=False)
    op.create_index('ix_break_plan_person_id', 'break_plan', ['person_id'], unique=False)
    op.create_index('ix_break_plan_unit_id', 'break_plan', ['unit_id'], unique=False)
    op.create_table('fatigue_report',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('duty_day', sa.Date(), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('summary', sa.String(500), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('reported_at', sa.DateTime(), nullable=False),
        sa.Column('manager_response', sa.String(1000), nullable=False),
        sa.Column('reviewed_by_id', sa.Integer()),
        sa.Column('reviewed_at', sa.DateTime()),
        sa.Column('closed_at', sa.DateTime()),
    )
    op.create_index('ix_fatigue_report_duty_day', 'fatigue_report', ['duty_day'], unique=False)
    op.create_index('ix_fatigue_report_person_id', 'fatigue_report', ['person_id'], unique=False)
    op.create_index('ix_fatigue_report_unit_id', 'fatigue_report', ['unit_id'], unique=False)
    op.create_table('leave',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('leave_type', sa.String(10), nullable=False),
        sa.Column('start', sa.Date(), nullable=False),
        sa.Column('end', sa.Date(), nullable=False),
    )
    op.create_index('ix_leave_unit_id', 'leave', ['unit_id'], unique=False)
    op.create_table('mfa_credential',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('encrypted_secret', sa.String(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('enrolled_at', sa.DateTime()),
        sa.Column('last_used_step', sa.BigInteger()),
        sa.Column('recovery_codes_digest', sa.String(), nullable=False),
        sa.UniqueConstraint('person_id'),
    )
    op.create_index('ix_mfa_credential_unit_id', 'mfa_credential', ['unit_id'], unique=False)
    op.create_table('notification',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('recipient_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('kind', sa.String(40), nullable=False),
        sa.Column('message', sa.String(500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('read_at', sa.DateTime()),
    )
    op.create_index('ix_notification_recipient_id', 'notification', ['recipient_id'], unique=False)
    op.create_index('ix_notification_unit_id', 'notification', ['unit_id'], unique=False)
    op.create_table('person_qualification',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('qualification_type_id', sa.Integer(), sa.ForeignKey('qualification_type.id'), nullable=False),
        sa.Column('issued_on', sa.Date()),
        sa.Column('valid_from', sa.Date()),
        sa.Column('expires_on', sa.Date()),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('unit_id', 'person_id', 'qualification_type_id', name='uq_person_qualification_type'),
    )
    op.create_index('ix_person_qualification_person_id', 'person_qualification', ['person_id'], unique=False)
    op.create_index('ix_person_qualification_unit_id', 'person_qualification', ['unit_id'], unique=False)
    op.create_table('position_endorsement',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('position_id', sa.Integer(), sa.ForeignKey('operational_position.id'), nullable=False),
        sa.Column('valid_from', sa.Date(), nullable=False),
        sa.Column('valid_until', sa.Date()),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('restrictions', sa.String(), nullable=False),
        sa.UniqueConstraint('unit_id', 'person_id', 'position_id', name='uq_position_endorsement_person'),
    )
    op.create_index('ix_position_endorsement_person_id', 'position_endorsement', ['person_id'], unique=False)
    op.create_index('ix_position_endorsement_position_id', 'position_endorsement', ['position_id'], unique=False)
    op.create_index('ix_position_endorsement_unit_id', 'position_endorsement', ['unit_id'], unique=False)
    op.create_table('roster_acknowledgement',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('publication_id', sa.Integer(), sa.ForeignKey('roster_publication.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_roster_acknowledgement_unit_id', 'roster_acknowledgement', ['unit_id'], unique=False)
    op.create_table('sickness',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('start', sa.Date(), nullable=False),
        sa.Column('end', sa.Date(), nullable=False),
        sa.Column('code', sa.String(10), nullable=False),
    )
    op.create_index('ix_sickness_unit_id', 'sickness', ['unit_id'], unique=False)
    op.create_table('staff_watch_history',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('watch_id', sa.Integer(), sa.ForeignKey('watch.id'), nullable=False),
        sa.Column('effective_date', sa.Date(), nullable=False),
    )
    op.create_index('ix_staff_watch_history_effective_date', 'staff_watch_history', ['effective_date'], unique=False)
    op.create_index('ix_staff_watch_history_staff_id', 'staff_watch_history', ['staff_id'], unique=False)
    op.create_index('ix_staff_watch_history_unit_id', 'staff_watch_history', ['unit_id'], unique=False)
    op.create_table('achieved_duty',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('planned_assignment_id', sa.Integer(), sa.ForeignKey('assignment.id')),
        sa.Column('actual_start', sa.DateTime(), nullable=False),
        sa.Column('actual_end', sa.DateTime(), nullable=False),
        sa.Column('duty_type', sa.String(30), nullable=False),
        sa.Column('variance_reason', sa.String(500), nullable=False),
        sa.Column('recorded_by_id', sa.Integer(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('unit_id', 'person_id', 'day', name='uq_achieved_duty_person_day'),
    )
    op.create_index('ix_achieved_duty_day', 'achieved_duty', ['day'], unique=False)
    op.create_index('ix_achieved_duty_person_id', 'achieved_duty', ['person_id'], unique=False)
    op.create_index('ix_achieved_duty_unit_id', 'achieved_duty', ['unit_id'], unique=False)
    op.create_table('annotation_audit',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('annotation_type_id', sa.Integer(), sa.ForeignKey('annotation_type.id')),
        sa.Column('assignment_id', sa.Integer(), sa.ForeignKey('assignment.id')),
        sa.Column('actor_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(30), nullable=False),
        sa.Column('old_value', sa.String(), nullable=False),
        sa.Column('new_value', sa.String(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('transaction_key', sa.String(64)),
        sa.UniqueConstraint('transaction_key'),
    )
    op.create_index('ix_annotation_audit_annotation_type_id', 'annotation_audit', ['annotation_type_id'], unique=False)
    op.create_index('ix_annotation_audit_assignment_id', 'annotation_audit', ['assignment_id'], unique=False)
    op.create_index('ix_annotation_audit_unit_id', 'annotation_audit', ['unit_id'], unique=False)
    op.create_table('person_qualification_history',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('person_qualification_id', sa.Integer(), sa.ForeignKey('person_qualification.id'), nullable=False),
        sa.Column('actor_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(30), nullable=False),
        sa.Column('snapshot_json', sa.String(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_person_qualification_history_person_qualification_id', 'person_qualification_history', ['person_qualification_id'], unique=False)
    op.create_index('ix_person_qualification_history_unit_id', 'person_qualification_history', ['unit_id'], unique=False)
    op.create_table('shift_request',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('staff_id', sa.Integer(), sa.ForeignKey('staff.id'), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('code', sa.String(10), nullable=False),
        sa.Column('requester_comment', sa.String(500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('submitted_at', sa.DateTime(), nullable=False),
        sa.Column('fulfilled_at', sa.DateTime()),
        sa.Column('cancelled_at', sa.DateTime()),
        sa.Column('resulting_assignment_id', sa.Integer(), sa.ForeignKey('assignment.id')),
        sa.Column('admin_response', sa.String()),
        sa.Column('responded_by_id', sa.Integer()),
        sa.Column('responded_at', sa.DateTime()),
        sa.Column('dismissed_by_requester_at', sa.DateTime()),
        sa.Column('status', sa.String(20)),
        sa.UniqueConstraint('staff_id', 'day', name='uniq_shift_request_staff_day'),
    )
    op.create_index('ix_shift_request_day', 'shift_request', ['day'], unique=False)
    op.create_index('ix_shift_request_staff_id', 'shift_request', ['staff_id'], unique=False)
    op.create_index('ix_shift_request_unit_id', 'shift_request', ['unit_id'], unique=False)
    op.create_table('request_audit',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), sa.ForeignKey('unit.id'), nullable=False),
        sa.Column('request_id', sa.Integer(), sa.ForeignKey('shift_request.id'), nullable=False),
        sa.Column('actor_id', sa.Integer(), nullable=False),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.Column('transition', sa.String(30), nullable=False),
        sa.Column('old_value', sa.String(), nullable=False),
        sa.Column('new_value', sa.String(), nullable=False),
        sa.Column('reason', sa.String(500), nullable=False),
    )
    op.create_index('ix_request_audit_request_id', 'request_audit', ['request_id'], unique=False)
    op.create_index('ix_request_audit_unit_id', 'request_audit', ['unit_id'], unique=False)
    op.create_table('work_pattern',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(120), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('cycle_length_days', sa.Integer(), nullable=False),
        sa.Column('contracted_minutes_per_cycle', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('unit_id', 'name', name='uq_work_pattern_unit_name'),
        sa.UniqueConstraint('unit_id', 'id', name='uq_work_pattern_unit_id'),
        sa.CheckConstraint('cycle_length_days > 0', name='ck_work_pattern_cycle_positive'),
        sa.CheckConstraint('contracted_minutes_per_cycle >= 0', name='ck_work_pattern_minutes_nonnegative'),
    )
    op.create_index('ix_work_pattern_unit_id', 'work_pattern', ['unit_id'], unique=False)
    op.create_table('work_pattern_day',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('work_pattern_id', sa.Integer(), nullable=False),
        sa.Column('day_index', sa.Integer(), nullable=False),
        sa.Column('day_type', sa.String(32), nullable=False),
        sa.Column('fixed_shift_type_id', sa.Integer()),
        sa.Column('required_work', sa.Boolean(), nullable=False),
        sa.Column('notes', sa.String(500), nullable=False),
        sa.ForeignKeyConstraint(['unit_id', 'work_pattern_id'], ['work_pattern.unit_id', 'work_pattern.id'], name='fk_work_pattern_day_pattern_unit', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unit_id', 'fixed_shift_type_id'], ['shift_type.unit_id', 'shift_type.id'], name='fk_work_pattern_day_shift_unit'),
        sa.UniqueConstraint('unit_id', 'work_pattern_id', 'day_index', name='uq_work_pattern_day_index'),
        sa.UniqueConstraint('unit_id', 'id', name='uq_work_pattern_day_unit_id'),
        sa.CheckConstraint('day_index >= 0', name='ck_work_pattern_day_index_nonnegative'),
        sa.CheckConstraint("day_type IN ('FIXED_SHIFT','WORK_ANY','WORK_ALLOWED_SET','OFF','OPTIONAL_WORK','PROTECTED_NON_OPERATIONAL')", name='ck_work_pattern_day_type'),
        sa.CheckConstraint("(day_type = 'FIXED_SHIFT' AND fixed_shift_type_id IS NOT NULL) OR (day_type <> 'FIXED_SHIFT' AND fixed_shift_type_id IS NULL)", name='ck_work_pattern_day_fixed_shift'),
    )
    op.create_index('ix_work_pattern_day_unit_id', 'work_pattern_day', ['unit_id'], unique=False)
    op.create_index('ix_work_pattern_day_work_pattern_id', 'work_pattern_day', ['work_pattern_id'], unique=False)
    op.create_table('work_pattern_day_allowed_shift',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('work_pattern_day_id', sa.Integer(), nullable=False),
        sa.Column('shift_type_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['unit_id', 'work_pattern_day_id'], ['work_pattern_day.unit_id', 'work_pattern_day.id'], name='fk_pattern_allowed_day_unit', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unit_id', 'shift_type_id'], ['shift_type.unit_id', 'shift_type.id'], name='fk_pattern_allowed_shift_unit'),
        sa.UniqueConstraint('unit_id', 'work_pattern_day_id', 'shift_type_id', name='uq_pattern_day_allowed_shift'),
    )
    op.create_index('ix_pattern_allowed_unit_id', 'work_pattern_day_allowed_shift', ['unit_id'], unique=False)
    op.create_index('ix_pattern_allowed_day_id', 'work_pattern_day_allowed_shift', ['work_pattern_day_id'], unique=False)
    op.create_index('ix_pattern_allowed_shift_id', 'work_pattern_day_allowed_shift', ['shift_type_id'], unique=False)
    op.create_table('staff_pattern_assignment',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('staff_id', sa.Integer(), nullable=False),
        sa.Column('work_pattern_id', sa.Integer(), nullable=False),
        sa.Column('effective_from', sa.Date(), nullable=False),
        sa.Column('effective_to', sa.Date()),
        sa.Column('anchor_date', sa.Date(), nullable=False),
        sa.Column('anchor_day_index', sa.Integer(), nullable=False),
        sa.Column('contracted_minutes_override', sa.Integer()),
        sa.Column('notes', sa.String(500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['unit_id', 'staff_id'], ['staff.unit_id', 'staff.id'], name='fk_staff_pattern_person_unit', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unit_id', 'work_pattern_id'], ['work_pattern.unit_id', 'work_pattern.id'], name='fk_staff_pattern_pattern_unit'),
        sa.CheckConstraint('effective_to IS NULL OR effective_to >= effective_from', name='ck_staff_pattern_effective_range'),
        sa.CheckConstraint('anchor_day_index >= 0', name='ck_staff_pattern_anchor_index'),
        sa.CheckConstraint('contracted_minutes_override IS NULL OR contracted_minutes_override >= 0', name='ck_staff_pattern_minutes_nonnegative'),
    )
    op.create_index('ix_staff_pattern_unit_id', 'staff_pattern_assignment', ['unit_id'], unique=False)
    op.create_index('ix_staff_pattern_staff_id', 'staff_pattern_assignment', ['staff_id'], unique=False)
    op.create_index('ix_staff_pattern_pattern_id', 'staff_pattern_assignment', ['work_pattern_id'], unique=False)
    op.create_index('ix_staff_pattern_effective_from', 'staff_pattern_assignment', ['effective_from'], unique=False)
    op.create_index('ix_staff_pattern_effective_to', 'staff_pattern_assignment', ['effective_to'], unique=False)
    op.create_table('staff_rule',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('unit_id', sa.Integer(), nullable=False),
        sa.Column('staff_id', sa.Integer(), nullable=False),
        sa.Column('rule_type', sa.String(40), nullable=False),
        sa.Column('hardness', sa.String(8), nullable=False),
        sa.Column('effective_from', sa.Date(), nullable=False),
        sa.Column('effective_to', sa.Date()),
        sa.Column('shift_type_id', sa.Integer()),
        sa.Column('shift_group', sa.String(20)),
        sa.Column('maximum_count', sa.Integer()),
        sa.Column('rolling_period_days', sa.Integer()),
        sa.Column('weekdays_mask', sa.Integer()),
        sa.Column('penalty_weight', sa.Integer(), nullable=False),
        sa.Column('reason', sa.String(500), nullable=False),
        sa.Column('authorised_by_user_id', sa.Integer()),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['unit_id', 'staff_id'], ['staff.unit_id', 'staff.id'], name='fk_staff_rule_person_unit', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unit_id', 'shift_type_id'], ['shift_type.unit_id', 'shift_type.id'], name='fk_staff_rule_shift_unit'),
        sa.ForeignKeyConstraint(['unit_id', 'authorised_by_user_id'], ['staff.unit_id', 'staff.id'], name='fk_staff_rule_authoriser_unit'),
        sa.CheckConstraint("hardness IN ('HARD','SOFT')", name='ck_staff_rule_hardness'),
        sa.CheckConstraint("rule_type IN ('NO_NIGHT','AVOID_NIGHT','NO_EARLY','AVOID_EARLY','ALLOWED_SHIFT','DISALLOWED_SHIFT','MAX_NIGHTS_PER_CYCLE','MAX_SHIFTS_PER_CYCLE','AVAILABLE_WEEKDAYS','UNAVAILABLE_WEEKDAYS','MAX_CONTRACTED_MINUTES','PREFERRED_SHIFT','PREFERRED_DAY_OFF')", name='ck_staff_rule_type'),
        sa.CheckConstraint('effective_to IS NULL OR effective_to >= effective_from', name='ck_staff_rule_effective_range'),
        sa.CheckConstraint('maximum_count IS NULL OR maximum_count >= 0', name='ck_staff_rule_maximum_nonnegative'),
        sa.CheckConstraint('rolling_period_days IS NULL OR rolling_period_days > 0', name='ck_staff_rule_period_positive'),
        sa.CheckConstraint('weekdays_mask IS NULL OR (weekdays_mask >= 0 AND weekdays_mask <= 127)', name='ck_staff_rule_weekdays_mask'),
        sa.CheckConstraint('penalty_weight >= 0', name='ck_staff_rule_penalty'),
    )
    op.create_index('ix_staff_rule_unit_id', 'staff_rule', ['unit_id'], unique=False)
    op.create_index('ix_staff_rule_staff_id', 'staff_rule', ['staff_id'], unique=False)
    op.create_index('ix_staff_rule_rule_type', 'staff_rule', ['rule_type'], unique=False)
    op.create_index('ix_staff_rule_effective_from', 'staff_rule', ['effective_from'], unique=False)
    op.create_index('ix_staff_rule_effective_to', 'staff_rule', ['effective_to'], unique=False)
