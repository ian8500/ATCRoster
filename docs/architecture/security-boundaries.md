# Security boundaries

## Extracted controls

| Control | Owner | Preserved behaviour |
|---|---|---|
| CSP and headers | `atcroster/security/headers.py` | Nonces, CSP directives, HSTS, authenticated no-store, request IDs and metrics completion |
| CSRF | `atcroster/security/csrf.py` | Session token, form/header input, constant-time comparison and global unsafe-method enforcement |
| Field encryption | `atcroster/security/encryption.py` | Current-key encryption, version-specific rotation, legacy unversioned decrypt and startup validation |
| Sessions | `atcroster/security/sessions.py` | Idle/absolute expiry, auth stamps, password/role/membership/MFA revocation and secure initialization |
| Errors | `atcroster/errors.py` | Safe 400/403/404/500 responses, request IDs and security-event logging |

No route is exempt from browser CSRF protection. Plaintext field values are not
cached or logged. Versioned encrypted fields use the first configured key for
new ciphertext and matching older keys for rotation. Unversioned legacy
ciphertext tries the configured ring without changing storage.

Session stamps bind password hash, role, membership state and MFA state. A
change invalidates already-issued browser sessions. A user rejected by the
Flask-Login loader has the stale signed session cleared immediately.

## Residual boundary

Role/MFA/kiosk endpoint restrictions and the Flask-Login user loader remain in
`app.py` because they depend on operational and control-plane models. Account
recovery and MFA HTTP workflows also remain there. Their next extraction must
retain central-versus-operational database ordering and session invalidation.
