# ATCRoster Architecture Review (2026-03-28)

## Scope
This review compares the current codebase against the stated commercial target architecture:
- React + TypeScript + Tailwind frontend
- FastAPI + SQLAlchemy + Pydantic + Alembic backend
- SQLite (MVP), Tauri desktop shell
- strict multi-unit (tenant) data isolation

## Current state summary
The current implementation is a Flask monolith with server-rendered templates and a single SQLite database binding. Core models such as `Staff`, `Watch`, `Assignment`, `Leave`, and `Requirement` are implemented directly in `app.py` without a tenant boundary column. Authentication is present, but login is staff-centric and does not establish an explicit active unit context.

## Strengths worth keeping
1. **Existing domain depth**: rostering, leave, sickness, watch history, shift requests, annotations, and changelog already exist and can seed migration.
2. **Practical UX features already built**: roster month loading, caching, and overtime SMS hooks.
3. **Test baseline exists**: pytest route/helper coverage is in place and can be expanded while refactoring.

## Critical gaps vs commercial multi-tenant target

### 1) Stack mismatch (High)
- Current app is Flask + Jinja templates in a single file (`app.py`), not React/FastAPI/Tauri.
- No service-layer boundary; request handlers and business logic are tightly coupled.

**Fix direction**
- Introduce a new `backend/` FastAPI app alongside current Flask app.
- Introduce a `frontend/` React + TypeScript + Tailwind app.
- Use a temporary strangler pattern: new APIs are built first while old views remain until parity is reached.

### 2) Tenant isolation missing (Critical)
- Core business tables do not include `unit_id`.
- No backend-enforced tenant scoping primitive.
- A user can conceptually interact with all global data in one DB namespace.

**Fix direction**
- Create `Unit` table and add `unit_id` foreign keys to all unit-scoped tables.
- Enforce tenant filter in every query at repository/service layer.
- Add defensive checks in write paths to block cross-unit foreign key references.
- Add tests proving cross-unit data cannot be read or edited.

### 3) Authentication model mismatch (High)
- `Staff` currently doubles as the auth principal.
- Required model calls for independent `User` entity (`email`, `password_hash`, `role`, `unit_id`).

**Fix direction**
- Introduce separate `User` model.
- Link to `unit_id`; keep `Staff` as operational workforce entity.
- Include active-unit context in session/JWT and enforce server-side authorization.

### 4) Rules engine not isolated (High)
- No dedicated dynamic rules engine module driven by per-unit `RuleSet` + `RuleConfig`.
- Fatigue/business rules are not centralized into a configurable evaluator contract.

**Fix direction**
- Build `rules_engine/` package with:
  - rule registry
  - config adapters
  - evaluation context loader (unit, staff, range)
  - violation output schema
- Store rule values in DB JSON (`RuleConfig.config_json`) and resolve at runtime.

### 5) Migration and schema governance (High)
- Schema changes in current code are partly handled by ad-hoc migration helpers in app runtime.
- No Alembic migration history controlling repeatable upgrades.

**Fix direction**
- Adopt Alembic as the single migration mechanism in new backend.
- Add baseline migration + data migration scripts from legacy schema.

### 6) Desktop packaging path mismatch (Medium)
- Current desktop packaging uses Python launcher/webview approach, not Tauri.

**Fix direction**
- Replace/augment with Tauri shell serving React frontend and local FastAPI backend process.
- Ensure first-run DB initialization and migration are automatic.

## Recommended implementation roadmap (incremental)

1. **Foundation**
   - Scaffold FastAPI backend with SQLAlchemy models and Alembic.
   - Add `Unit` + `User` + tenant context middleware/dependency.
2. **Core data migration**
   - Port `Staff`, `Watch`, `ShiftType`, `Assignment`, `Leave`, `Requirement` with `unit_id`.
   - Add migration scripts and one-time legacy import utility.
3. **Roster API + frontend shell**
   - Build roster grid API and React grid view (editable cells, watch grouping, day columns).
4. **Rules engine**
   - Implement `RuleSet`/`RuleConfig` tables and evaluator pipeline.
   - Implement baseline fatigue rules from config values.
5. **Security + audit hardening**
   - Unit-scoped audit logs, robust authz checks, and cross-unit negative tests.
6. **Tauri packaging**
   - Bundle frontend and backend startup into single desktop installer.

## Immediate tactical fixes done in this pass
- Updated tests to use `db.session.get(...)` instead of legacy `Query.get(...)`, removing SQLAlchemy 2.x deprecation noise and aligning with modern API usage.

## Suggested next coding tasks
1. Add `Unit` and `User` models in a new backend package.
2. Add a tenant-aware query helper/dependency and enforce it in one pilot endpoint (`GET /staff`).
3. Create first Alembic migration with unitized schema for core entities.
4. Add tests:
   - read isolation (`unit A` cannot list `unit B` staff)
   - write isolation (cannot assign `unit B` watch/staff)
5. Start React roster page consuming API (read-only first, edit second).
