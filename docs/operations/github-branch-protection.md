# GitHub release protection

Protect `main` in the repository settings. Require pull requests, at least one
independent approving review (two for authentication, tenant, roster-publication or
live-position changes), dismissal of stale approvals, conversation resolution and
the current Quality matrix, PostgreSQL/Redis integration, container/Trivy, CodeQL
and gitleaks checks. Require the branch to be current before merge.

Disable force pushes and branch deletion. Restrict bypass and direct push rights to
named emergency release owners and review every use. Enable private vulnerability
reporting, Dependabot alerts/security updates, secret scanning and push protection.
Require signed release tags and signed commits where the team's tooling supports a
reliable verification policy. Retain Actions and deployment-environment audit logs.

Staging and production GitHub environments should require separate approval, limit
their Railway token to the target environment and permit deployment only from
protected `main`. The promotion workflow must deploy the exact accepted SHA; a
green run for a different commit is not release evidence.
