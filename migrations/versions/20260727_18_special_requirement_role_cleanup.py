"""Remove operational special-requirement data structures from control DBs."""

import os

from alembic import op
from sqlalchemy import inspect

revision = "20260727_18"
down_revision = "20260727_17"
branch_labels = None
depends_on = None


def upgrade():
    if os.environ.get("ATCROSTER_SCHEMA_ROLE") != "control":
        return
    if "special_requirement" in inspect(op.get_bind()).get_table_names():
        op.drop_table("special_requirement")


def downgrade():
    # Revision 17 is role-aware for new installations. Recreating an
    # operational table in a control database would violate that boundary.
    pass
