# ATCRoster Security

## Reporting a vulnerability

Do not open a public issue containing credentials, personal data, tenant data,
or exploit details. Send a private report to the repository owner with:

- the affected version and deployment type;
- reproduction steps using non-production data;
- likely impact;
- any suggested mitigation.

Rotate exposed secrets immediately. Preserve relevant audit records and avoid
copying operational personnel data into the report.

## Current security controls

- Airport context is derived from the authenticated user, never a form or query
  parameter.
- Operational models are tenant-filtered and cross-airport writes are rejected.
- Passwords are stored using Werkzeug password hashing.
- TOTP MFA is supported with secrets encrypted using a deployment Fernet key;
  production forces unenrolled users through MFA setup.
- Privileged and sensitive write routes use CSRF tokens.
- Login rotates session state.
- Cookies are HTTP-only and SameSite=Lax. Production deployments must set
  `ATCROSTER_SECURE_COOKIES=true`.
- Responses set clickjacking, MIME-sniffing, referrer and browser-permission
  protections. HTTPS responses enable HSTS.
- Roster publications are immutable versioned snapshots; acknowledgements are
  tied to a specific version.
- Compliance and platform exports contain only the data necessary for their
  stated purpose.

## Production requirements

Before processing live operational data:

1. Use PostgreSQL and a unique high-entropy `FLASK_SECRET_KEY`.
2. Terminate TLS at a maintained reverse proxy and force HTTPS.
3. Enable secure cookies.
4. Restrict database and backup credentials through a managed secret store.
5. Complete TOTP MFA enrollment for every account before operational use, or
   integrate an organisation identity provider through an independently
   assessed authentication boundary.
6. Configure encrypted backups and complete a witnessed restore test.
7. Run dependency, static-analysis, tenant-isolation and penetration tests.
8. Define log retention, incident response, access review and leaver processes.
9. Complete a DPIA and agree controller/processor responsibilities.
10. Obtain independent aviation operational acceptance. The software provides
    decision support and does not itself certify regulatory compliance.

## Known readiness gaps

The repository does not currently claim Cyber Essentials, ISO 27001, SOC 2,
CAA approval or EASA certification. Passkeys, enterprise SSO, centralised rate
limiting and SIEM integration remain recommended for a broadly available
hosted service.
