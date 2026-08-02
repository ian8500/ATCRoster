# Tenant request lifecycle

The browser never supplies the authoritative airport ID. Flask-Login resolves
an active `UnitMembership`, and only its server-verified `unit_id` is passed to
`atcroster/tenancy_hooks.py`.

For every request the hook:

1. clears all context variables;
2. creates the request ID and CSP nonce;
3. rejects expired or auth-stamp-invalid sessions;
4. binds a normal account to its verified airport and configured database
   route; or
5. binds a SuperAdmin to platform control, which explicitly forbids routine
   operational database access.

Anonymous and public routes stay unbound. In production, a verified airport
without routing metadata fails closed with HTTP 503. The separate principal
boundary in `app.py` continues to restrict SuperAdmin, kiosk and MFA setup
routes after binding.

Teardown resets the airport and platform-control tokens and forcibly clears the
context variables. It runs for normal responses, redirects and exceptions.
Background work must continue to use `operational_unit_context()` and therefore
cannot inherit an HTTP request's stale tenant.

The direct hook tests cover anonymous requests, verified airport binding,
SuperAdmin isolation and cleanup after exceptions. Existing physical database
tests cover airport-to-airport isolation.
