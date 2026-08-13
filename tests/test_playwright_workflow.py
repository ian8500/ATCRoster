from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_playwright_job_is_independent_and_uses_isolated_seeded_runtime():
    workflow = yaml.safe_load((ROOT / ".github/workflows/quality.yml").read_text())
    job = workflow["jobs"]["playwright-e2e"]
    rendered = str(job)

    assert "npm ci" in rendered
    assert job["steps"][2]["with"]["node-version"] == "22.14.0"
    assert "playwright install --with-deps chromium" in rendered
    assert "run_e2e_server.py" in rendered
    assert "health/ready" in rendered
    assert "npm run test:e2e" in rendered
    assert "failure()" in rendered
    assert "test-results/" in rendered and "playwright-report/" in rendered
    assert "postgres-multidatabase" in workflow["jobs"]
    assert "test" in workflow["jobs"]
    assert job["env"]["REDIS_URL"] == ""
