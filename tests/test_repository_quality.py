from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_production_health_captures_login_before_grep():
    workflow = (ROOT / ".github/workflows/production-health.yml").read_text()

    assert 'login="$(curl ' in workflow
    assert 'grep -Fq "<title>Login" <<<"$login"' in workflow
    assert not re.search(r"curl[^\\n]*\\|\\s*grep", workflow)


def test_generated_repository_artifacts_are_ignored_and_removed():
    ignored = (ROOT / ".gitignore").read_text().splitlines()

    assert ".DS_Store" in ignored
    assert "*.db" in ignored
    assert "app for website" in ignored
    assert not (ROOT / ".DS_Store").exists()
    assert not (ROOT / "roster.db").exists()
    assert not (ROOT / "app for website").exists()


def test_production_container_excludes_local_quality_and_test_artifacts():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()

    for path in {
        ".coverage",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "instance",
        "tests",
        "docs",
        "deliverables",
    }:
        assert path in ignored


def test_authentication_blueprint_preserves_public_route_contract():
    from app import app

    routes = {
        rule.endpoint: (rule.rule, rule.methods)
        for rule in app.url_map.iter_rules()
        if rule.endpoint in {"login", "logout"}
    }

    assert routes["login"][0] == "/login"
    assert {"GET", "POST"} <= routes["login"][1]
    assert routes["logout"][0] == "/logout"
    assert "POST" in routes["logout"][1]
    assert "GET" not in routes["logout"][1]
    assert app.view_functions["login"].__module__ == "auth_blueprint"
    assert app.view_functions["logout"].__module__ == "auth_blueprint"


def test_templates_do_not_reintroduce_inline_style_attributes():
    templates = ROOT / "templates"
    offenders = []
    for path in templates.rglob("*.html"):
        text = path.read_text()
        if "style=" in text:
            offenders.append(str(path.relative_to(ROOT)))
        for line in text.splitlines():
            if "<style" in line and 'nonce="{{ csp_nonce() }}"' not in line:
                offenders.append(f"{path.relative_to(ROOT)}:unnonced-style")
    assert offenders == []


def test_quality_workflow_discovers_sources_and_builds_supported_pythons_and_sbom():
    workflow = (ROOT / ".github/workflows/quality.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert 'python-version: ["3.12", "3.14"]' in workflow
    assert "git ls-files -z '*.py' | xargs -0 ruff check" in workflow
    assert "format: cyclonedx" in workflow
    assert "verify_release_candidate.py" in workflow


def test_worker_readiness_requires_database_heartbeat_not_only_process_liveness():
    source = (ROOT / "scripts/run_worker_service.py").read_text()
    assert "worker_heartbeat" in source
    assert "last_seen_at" in source
    assert 'self.path == "/health/ready" and worker_ready()' in source
