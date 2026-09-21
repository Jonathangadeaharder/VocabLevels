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
    assert "uv run" not in sonar, (
        "the scan job must not execute repo code (incl. validator scripts)"
    )
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


def test_ci_cancels_superseded_runs() -> None:
    ci = _ci_workflow()
    assert "cancel-in-progress: true" in ci, (
        "rapid pushes must cancel superseded CI runs instead of piling"
        " 30-minute suites onto the single self-hosted runner and"
        " firing a scan for each"
    )


def test_scan_rejects_symlinks_escaping_the_workspace() -> None:
    sonar = _sonar_workflow()
    assert "Reject symlinks escaping the workspace" in sonar, (
        "SonarScanner follows symlinks under sonar.sources=.; a committed"
        " symlink resolving outside the workspace would exfiltrate runner"
        " files to the SonarQube server"
    )


def test_scan_discards_artifact_staging_directory() -> None:
    steps = _sonar_steps()
    download = next(c for c in steps.values() if "gh run download" in c)
    mv_at = download.find("mv coverage-report/coverage.xml coverage.xml")
    cleanup_at = download.find("rm -rf coverage-report", mv_at)
    assert mv_at != -1 and cleanup_at != -1 and cleanup_at > mv_at, (
        "extracted artifact files are untracked and unchecked, so the"
        " staging directory must be discarded once coverage.xml is out"
    )


def test_scan_bounds_artifact_size_before_download() -> None:
    steps = _sonar_steps()
    download = next(c for c in steps.values() if "gh run download" in c)
    size_gate_at = download.find("size_in_bytes")
    download_at = download.find('gh run download "$CI_RUN_ID"')
    assert "coverage artifact too large" in download, (
        "gh run download extracts the archive before any local check can"
        " run, so an oversized PR-controlled artifact can exhaust the disk"
        " of the shared self-hosted runner; its API-reported size must be"
        " fetched and refused before downloading"
    )
    assert size_gate_at != -1 and size_gate_at < download_at, (
        "the size gate must run before gh run download touches the artifact"
    )


def test_scan_rejects_coverage_entries_for_missing_files() -> None:
    sonar = _sonar_workflow()
    assert "os.path.isfile(os.path.join(root, name))" in sonar, (
        "every <class> filename must resolve to a real file in the"
        " workspace before the report is handed to SonarQube"
    )
    assert "does not exist in the scanned tree" in sonar, (
        "PR-controlled ci.yml can fabricate coverage entries for paths"
        " that do not exist; the validator must confine fabricated data"
        " to files the scanner would analyze anyway"
    )


def test_scan_validates_untrusted_coverage_report() -> None:
    sonar = _sonar_workflow()
    assert "Validate coverage report" in sonar, (
        "PR CI runs execute the PR's own ci.yml, so coverage.xml is an"
        " untrusted input to the token-bearing scan job and must be"
        " rejected unless it is a regular, well-formed report whose"
        " paths stay inside the workspace"
    )
    assert "must not contain DTD or entity declarations" in sonar, (
        "entity expansion must be rejected independent of the runner's"
        " expat version; genuine coverage.py reports carry no DTD"
    )
    assert "must be UTF-8 encoded" in sonar, (
        "a UTF-16/UTF-32 report hides <!DOCTYPE from a byte-level scan"
        " while expat still expands its entities"
    )
