import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
config.set_main_option("sqlalchemy.url", os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url")))
schema_role = os.environ.get("ATCROSTER_SCHEMA_ROLE", "combined")
if schema_role not in {"control", "operational", "combined"}:
    raise RuntimeError("ATCROSTER_SCHEMA_ROLE must be control, operational, or combined")
if (
    os.environ.get("ATCROSTER_ENVIRONMENT") == "production"
    and schema_role == "combined"
):
    raise RuntimeError(
        "ATCROSTER_SCHEMA_ROLE is mandatory and cannot be combined in production"
    )
# Production upgrades use explicit revision operations and never import Flask.
target_metadata = None


def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_offline() if context.is_offline_mode() else run_migrations_online()
