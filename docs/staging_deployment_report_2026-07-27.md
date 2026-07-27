# Pilot staging deployment report — 27 July 2026

## Outcome

The `pilot-staging` Railway environment is deployed and healthy. It is
isolated from production and contains:

- one web service;
- one provisioning worker;
- one Redis service;
- one control-plane PostgreSQL database; and
- one physically separate operational PostgreSQL database for the test unit.

No local acceptance database or production data was uploaded.

## Verification

| Check | Result |
| --- | --- |
| Web deployment | Success |
| Worker deployment | Success |
| Redis deployment | Success |
| Control PostgreSQL deployment | Success |
| Operational PostgreSQL deployment | Success |
| `/health/live` | HTTP 200 |
| `/health/ready` | HTTP 200 |
| `/login` | HTTP 200 |
| Control migration | `20260727_15` |
| Operational migration | `20260727_15` |
| Automated regression suite | 101 passed, 2 skipped |

## Defects found and corrected during deployment

1. Railway's generic PostgreSQL URLs selected the legacy `psycopg2` dialect.
   Staging now uses explicit `postgresql+psycopg` private connection URLs.
2. Railway health probes use the host `healthcheck.railway.app`. It is now
   included in the staging trusted-host allowlist alongside the staging
   public domain.
3. An untrusted host could make the HTML error handler fail because Flask had
   deliberately refused to create a URL adapter. Untrusted hosts now receive
   a plain HTTP 400 response, covered by a regression test.
4. The readiness endpoint was hard-coded to migration `20260727_14`.
   Readiness now compares the database revision with the repository's actual
   Alembic head.
5. Railway applied a shared start command to web and worker services. The
   deployment now selects the intended process using
   `ATCROSTER_PROCESS_TYPE`; the worker exposes a minimal supervised health
   endpoint and fails when its child process fails.

## Boundaries

- Staging uses unique Flask, field-encryption and token-encryption keys.
- Database and Redis traffic uses Railway private service networking.
- Production services, variables and data were not changed.
- Email, SMS, monitoring, backup/restore and external penetration testing
  remain separate launch gates.
