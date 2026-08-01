# Browser security policy

The default response policy is nonce-based and contains no `unsafe-inline` token.
It restricts the default, script, style, style attributes, images, fonts,
connections, workers, manifests, forms, frames, objects and base URL. Production
also enables `upgrade-insecure-requests` and HSTS.

Inline JavaScript and `<style>` blocks remain only where they carry the
per-request CSP nonce. Inline event handlers and `style=` attributes are forbidden
by repository tests. Dynamic bars use native `<progress>` controls and annotation
colour previews use a native colour control instead of injected CSS.

The remaining exceptions are the integrity-pinned Bootstrap asset on jsDelivr and
Font Awesome CSS/fonts on cdnjs. They are availability and supply-chain
dependencies, not wildcard sources. A later asset-pipeline change should vendor
these exact files and then reduce `script-src`, `style-src` and `font-src` to
`'self'`. Browser regression must cover the kiosk live-event connection, service
worker, MFA QR image, reports and print/PDF flows before removing a source.
