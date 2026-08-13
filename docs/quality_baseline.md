# Python quality baseline

The maintained Python application is checked by Ruff and Bandit in CI. The
initial enforcement deliberately avoids a repository-wide formatting rewrite.

## Ruff

Ruff lint covers the root application modules and maintained operational
scripts. `app.py` retains a narrow, explicit legacy exception list for
late imports, duplicate compatibility definitions, one lambda assignment,
unused imports and one unused local. Two document/import scripts retain
unused-import exceptions. New modules receive the default rule set.

Formatting remains enforced on the already-formatted service modules and the
`atcroster` application-factory package. MyPy also checks the factory package
alongside the existing typed services. Expand those lists one module at a time; format
`app.py` only in a dedicated, behaviour-neutral change after its routes have
been reduced.

## Bandit

Bandit scans the complete maintained application at medium severity or above.
Low-severity findings—primarily deliberate compatibility exception handling
and fixed process-launch commands—are the initial recorded debt. Three
medium false positives carry inline, rule-specific suppressions:

- ClickSend uses a fixed HTTPS API origin;
- the desktop readiness probe uses a generated loopback URL;
- the Railway worker exposes only its health listener on all interfaces.

Review and reduce the low-severity backlog by subsystem; do not add broad
Bandit exclusions.

## Coverage

Coverage starts at 60% for maintained application modules. Tests, migrations,
one-off scripts, the desktop launcher and the WSGI shim are excluded from the
percentage. The threshold must never be lowered to merge a change. Raise it
incrementally as authentication, recovery and operational routes move into
focused modules with direct tests.
