# Application bootstrap

## Current entry points

`wsgi:application` remains the production entry point and is the same Flask
object exported as `app.app`. `atcroster.create_app()` remains the only
application factory. The refactor has not created a second SQLAlchemy,
Flask-Login, cache, or metrics instance.

`app.py` still assembles extensions, models, domain blueprints and legacy route
registrations. Extracted framework concerns are registered explicitly:

1. security headers and response completion;
2. CSRF protection;
3. public/legal routes;
4. error handlers;
5. authenticated session lifecycle; and
6. tenant request binding and teardown.

Compatibility aliases remain in `app.py` for `_security_headers`,
`_validate_csrf`, `_enforce_csrf`, `_field_ciphers`, `_encrypt_field`,
`_decrypt_field`, `_current_auth_stamp`, `_initialize_authenticated_session`,
`_bind_tenant_context`, `_reset_tenant_context`, and the four error handlers.
They prevent existing blueprints, tests and operational tools from importing a
new internal location during this incremental phase.

## Registration order

Request metrics begin before CSRF and identity work. CSRF remains global and
default-deny for unsafe browser methods. Flask-Login resolves the signed
principal before session revocation and tenant binding. Principal-specific
SuperAdmin, kiosk and MFA restrictions run after the verified context is bound.
Tenant context is cleared again during teardown, including after exceptions.

## Next bootstrap step

Do not move extension creation until operational models and the tenant-routed
SQLAlchemy session have a stable package boundary. After the account,
qualification, publication and CLI domains are extracted, introduce
`atcroster/application.py` as assembly code and retain `app.py` as the
compatibility export for `app` and model aliases.
