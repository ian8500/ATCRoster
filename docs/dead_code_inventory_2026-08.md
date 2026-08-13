# Dead-code inventory — August 2026

## Method

This audit combines tracked-file inspection, the Flask route contract, Jinja
template references, deployment entry points and Ruff unused-import checks.
Dynamic blueprint registration, Alembic history, SQLAlchemy events and Railway
scripts are treated as runtime references rather than static-analysis evidence
of deletion.

## Findings

| Classification | Finding | Decision |
| --- | --- | --- |
| Definitely unused | The removed roster-impact queue routes still had three behavioural tests. | Replaced with one 404 route-registration contract; the retained roster-impact service is still invoked by roster, qualification and work-pattern mutations. |
| Definitely unused | An unused `os` import was introduced in the web-server startup test. | Removed. |
| Runtime dynamically referenced | Blueprints, Jinja macros/templates, Alembic migrations and SQLAlchemy models/events. | Retained: static reference analysis is not sufficient evidence of dead code. |
| Operations/deployment only | Database migration, backup/recovery, worker and Railway startup scripts. | Retained: they are invoked by CI, Railway manifests, or operational runbooks. |
| Compatibility requirement | `app.py` compatibility exports and legacy model aliases. | Retained pending the documented incremental architecture extraction. |

Pre-existing untracked duplicate files with names such as `"app 2.py"` were not
included in the repository audit and were left untouched because they are user
workspace data, not tracked application code.
