"""Pin the SONAR_TOKEN isolation contract of issue #167.

The SonarQube scan job holds the SONAR_TOKEN secret and runs on the same
self-hosted runner pool as the unprivileged CI test job. These tests pin
the workflow shapes that keep the token away from PR-controlled execution:
the scan job never runs repo code, coverage arrives only as an artifact
from the triggering CI run, and the scanner reads config pinned to
origin/main instead of the scanned tree.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

STEP_SPLIT = re.compile(r"^      - ", re.MULTILINE)


def _sonar_workflow() -> str:
    return (WORKFLOWS / "sonarcloud.yml").read_text(encoding="utf-8")


def _ci_workflow() -> str:
    return (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")


def _sonar_steps() -> dict[str, str]:
    text = _sonar_workflow()
    steps = STEP_SPLIT.split(text)
    chunks: dict[str, str] = {}
    for chunk in steps[1:]:
        name = chunk.splitlines()[0].removeprefix("name: ").strip()
        chunks[name] = chunk
    return chunks


def test_required_check_names_are_preserved() -> None:
    assert "name: Test" in _ci_workflow()
    assert "name: SonarQube Analysis" in _sonar_workflow()


def test_scan_job_never_executes_repo_code() -> None:
    sonar = _sonar_workflow()
    assert "setup-uv" not in sonar, (
        "the scan job must not install toolchains for repo code"
    )
    assert "uv sync" not in sonar, "the scan job must not install repo deps"
    assert "pytest" not in sonar, "the scan job must not run repo tests"


def test_ci_uploads_coverage_artifact_for_the_scan() -> None:
    ci = _ci_workflow()
    assert "--cov-report=xml" in ci, "CI must produce coverage.xml"
    assert "upload-artifact@" in ci, "CI must upload the coverage artifact"
    assert "name: coverage" in ci, "artifact name must match the scan download"
    assert "if-no-files-found: error" in ci, (
        "a green CI run without coverage.xml must fail at the source"
    )


def test_scan_pulls_coverage_from_the_triggering_ci_run() -> None:
    steps = _sonar_steps()
    assert any("gh run download" in chunk for chunk in steps.values()), (
        "the scan must download coverage from the CI run, not run tests"
    )
    assert any("github.event.workflow_run.id" in c for c in steps.values())


def test_scanner_config_is_pinned_to_main() -> None:
    sonar = _sonar_workflow()
    assert "rm -rf sonar-project.properties" in sonar, (
        "PR-supplied sonar-project.properties must be removed first, even"
        " when committed as a directory (rm -rf unlinks symlinks, never"
        " follows them)"
    )
    assert "origin/main:sonar-project.properties" in sonar, (
        "the scanner must read config pinned to origin/main"
    )


def test_sonar_token_is_confined_to_the_scan_step() -> None:
    all_steps = _sonar_steps()
    token_steps = [n for n, c in all_steps.items() if "secrets.SONAR_TOKEN" in c]
    assert token_steps == ["SonarQube Scan"], (
        f"secrets.SONAR_TOKEN leaked outside the scan step: {token_steps}"
    )


def test_scan_job_grants_artifact_read_permission() -> None:
    assert "actions: read" in _sonar_workflow(), (
        "the scan job needs actions:read to download the CI artifact"
    )


def test_ci_job_bounds_runner_time() -> None:
    ci = _ci_workflow()
    assert "timeout-minutes: 30" in ci, (
        "a hung Test run on the shared self-hosted pool must not occupy"
        " the runner indefinitely; it also stalls the SonarQube scan that"
        " now depends on this run completing"
    )
