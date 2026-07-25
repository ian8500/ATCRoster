"""authoritative qualification lifecycle and legacy reconciliation

Revision ID: 20260725_06
Revises: 20260725_05
"""
import sqlalchemy as sa
from alembic import op

revision = "20260725_06"
down_revision = "20260725_05"
branch_labels = None
depends_on = None


def _columns(inspector, table):
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "qualification_type" not in tables:
        op.create_table(
            "qualification_type",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column("code", sa.String(30), nullable=False),
            sa.Column("label", sa.String(100), nullable=False),
            sa.Column("warning_days_csv", sa.String(100), nullable=False, server_default="180,90,60,30"),
            sa.Column("expiry_required", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.ForeignKeyConstraint(["unit_id"], ["unit.id"]),
            sa.UniqueConstraint("unit_id", "code", name="uq_qualification_unit_code"),
        )
    else:
        columns = _columns(inspector, "qualification_type")
        with op.batch_alter_table("qualification_type") as batch:
            if "expiry_required" not in columns:
                batch.add_column(sa.Column(
                    "expiry_required", sa.Boolean(), nullable=False,
                    server_default=sa.true(),
                ))
            if "is_active" not in columns:
                batch.add_column(sa.Column(
                    "is_active", sa.Boolean(), nullable=False,
                    server_default=sa.true(),
                ))

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "person_qualification" not in tables:
        op.create_table(
            "person_qualification",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column("person_id", sa.Integer(), nullable=False, index=True),
            sa.Column("qualification_type_id", sa.Integer(), nullable=False),
            sa.Column("issued_on", sa.Date()),
            sa.Column("valid_from", sa.Date()),
            sa.Column("expires_on", sa.Date()),
            sa.Column("status", sa.String(20), nullable=False, server_default="valid"),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["unit_id"], ["unit.id"]),
            sa.ForeignKeyConstraint(["person_id"], ["staff.id"]),
            sa.ForeignKeyConstraint(["qualification_type_id"], ["qualification_type.id"]),
            sa.UniqueConstraint(
                "unit_id", "person_id", "qualification_type_id",
                name="uq_person_qualification_type",
            ),
        )
    else:
        columns = _columns(inspector, "person_qualification")
        with op.batch_alter_table("person_qualification") as batch:
            if "issued_on" not in columns:
                batch.add_column(sa.Column("issued_on", sa.Date()))
            if "valid_from" not in columns:
                batch.add_column(sa.Column("valid_from", sa.Date()))
            if "updated_at" not in columns:
                batch.add_column(sa.Column(
                    "updated_at", sa.DateTime(), nullable=False,
                    server_default=sa.func.now(),
                ))
        constraints = {
            item.get("name")
            for item in sa.inspect(bind).get_unique_constraints(
                "person_qualification"
            )
        }
        if "uq_person_qualification_type" not in constraints:
            with op.batch_alter_table("person_qualification") as batch:
                batch.create_unique_constraint(
                    "uq_person_qualification_type",
                    ["unit_id", "person_id", "qualification_type_id"],
                )

    if "person_qualification_history" not in set(sa.inspect(bind).get_table_names()):
        op.create_table(
            "person_qualification_history",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("unit_id", sa.Integer(), nullable=False, index=True),
            sa.Column("person_qualification_id", sa.Integer(), nullable=False, index=True),
            sa.Column("actor_id", sa.Integer(), nullable=False),
            sa.Column("action", sa.String(30), nullable=False),
            sa.Column("snapshot_json", sa.Text(), nullable=False),
            sa.Column("occurred_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["unit_id"], ["unit.id"]),
            sa.ForeignKeyConstraint(
                ["person_qualification_id"], ["person_qualification.id"]
            ),
        )

    # Deterministic legacy reconciliation. Inserts are guarded by unit/code and
    # person/type pairs so a resumed deployment cannot duplicate records.
    staff_columns = _columns(sa.inspect(bind), "staff")
    if "staff" not in set(sa.inspect(bind).get_table_names()) or "unit_id" not in staff_columns:
        return
    defaults = (
        ("MEDICAL", "Medical", True),
        ("ADI", "Aerodrome Instrument", False),
        ("APP", "Approach Procedural", False),
        ("APS", "Approach Surveillance", False),
        ("MET", "Meteorological", False),
        ("OJTI", "On-the-job Training Instructor", False),
        ("ASSESSOR", "Assessor", False),
        ("UCA", "Unit Competence Assessor", False),
        ("ENGLISH_LANGUAGE", "English Language", True),
    )
    unit_ids = [
        row[0] for row in bind.execute(
            sa.text("SELECT id FROM unit ORDER BY id")
        )
    ]
    for unit_id in unit_ids:
        for code, label, expiry_required in defaults:
            bind.execute(sa.text(
                "INSERT INTO qualification_type "
                "(unit_id, code, label, warning_days_csv, expiry_required, is_active) "
                "SELECT :unit_id, :code, :label, '180,90,60,30', :expiry, 1 "
                "WHERE NOT EXISTS (SELECT 1 FROM qualification_type "
                "WHERE unit_id=:unit_id AND code=:code)"
            ), {
                "unit_id": unit_id, "code": code, "label": label,
                "expiry": expiry_required,
            })
    legacy_map = (
        ("MEDICAL", "medical_expiry", None),
        ("ADI", "tower_ue_expiry", "tower_ut"),
        ("APS", "radar_ue_expiry", "radar_ut"),
        ("MET", "met_ue_expiry", "met_ut"),
        ("OJTI", None, "has_ojti"),
        ("ASSESSOR", None, "has_assessor"),
    )
    for code, expiry_column, flag_column in legacy_map:
        if expiry_column and expiry_column not in staff_columns:
            expiry_column = None
        if flag_column and flag_column not in staff_columns:
            flag_column = None
        if not expiry_column and not flag_column:
            continue
        eligibility = []
        if expiry_column:
            eligibility.append(f"s.{expiry_column} IS NOT NULL")
        if flag_column:
            eligibility.append(f"s.{flag_column} IS TRUE")
        expiry_select = f"s.{expiry_column}" if expiry_column else "NULL"
        bind.execute(sa.text(
            "INSERT INTO person_qualification "
            "(unit_id, person_id, qualification_type_id, expires_on, status, updated_at) "
            f"SELECT s.unit_id, s.id, qt.id, {expiry_select}, 'valid', CURRENT_TIMESTAMP "
            "FROM staff s JOIN qualification_type qt "
            "ON qt.unit_id=s.unit_id AND qt.code=:code "
            f"WHERE ({' OR '.join(eligibility)}) "
            "AND NOT EXISTS (SELECT 1 FROM person_qualification pq "
            "WHERE pq.unit_id=s.unit_id AND pq.person_id=s.id "
            "AND pq.qualification_type_id=qt.id)"
        ), {"code": code})


def downgrade():
    raise RuntimeError(
        "Qualification reconciliation is intentionally irreversible."
    )
