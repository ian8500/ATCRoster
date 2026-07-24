# Database migrations

Alembic owns production schema upgrades. Set `DATABASE_URL`, then run:

```bash
alembic upgrade head
```

The application retains a small idempotent compatibility migration for older
SQLite desktop installations. Production PostgreSQL deployments must use
Alembic and one operational database per airport.
