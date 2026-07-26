# GitHub repository protection handoff

Configure the `main` branch ruleset manually in GitHub:

- require pull requests and at least one approval;
- dismiss stale approvals when new commits are pushed;
- require every Quality job and the complete CodeQL workflow;
- require branches to be current before merging;
- require all review conversations to be resolved;
- block force pushes and branch deletion;
- restrict direct pushes to authorised release maintainers;
- do not allow Dependabot to bypass these protections.

Enable Dependabot security updates in repository settings. Automated updates
must not be auto-merged. Python major versions and material database, security,
container or workflow-runtime updates require a dedicated compatibility pull
request with the complete production verification matrix.
