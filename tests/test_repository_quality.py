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
