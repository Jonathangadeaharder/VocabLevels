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
    sonar = _sonar_workflow()
    assert "actions/runs/$CI_RUN_ID/artifacts" in sonar, (
        "the scan must fetch the coverage artifact from the triggering"
        " CI run, not run tests"
    )
    assert "actions/artifacts/$ARTIFACT_ID/zip" in sonar, (
        "the artifact zip must be fetched raw so its extraction can be"
        " bounded; gh run download extracts unbounded inside the"
        " workspace"
    )
    assert 'gh run download "' not in sonar, (
        "gh run download unzips the whole archive before any local check"
        " can run; extraction must happen in the bounded step"
    )
    assert "github.event.workflow_run.id" in sonar
    assert "gh api --paginate" in sonar, (
        "the artifacts API returns at most 30 entries per page, so"
        " without pagination the lookup can miss the coverage artifact"
        " and fail the scan with a misleading not-found error"
    )
    assert 'ARTIFACTS="$(gh api --paginate' in sonar, (
        "read swallows the command substitution status: a failed gh api"
        " would be misreported as artifact-not-found. The capture must"
        " propagate the API failure under set -e"
    )
    assert "expected one 'coverage' artifact" in sonar, (
        "if a run ever holds more than one artifact named coverage, only"
        " the first line would be consumed and the job could proceed"
        " with an arbitrary artifact"
    )


def test_scan_cancels_superseded_scans() -> None:
    sonar = _sonar_workflow()
    group = (
        "sonarqube-${{ github.event.workflow_run.head_repository.full_name }}"
        "-${{ github.event.workflow_run.head_branch }}"
    )
    assert group in sonar, (
        "SHA-keyed concurrency never cancels superseded scans: every"
        " push piles another queued run onto the single self-hosted"
        " runner. The group must be keyed by head repo and branch: fork"
        " PRs report the unqualified fork branch name, and workflow-level"
        " concurrency applies before the fork-excluding job if, so a"
        " bare branch key lets a fork PR named 'main' cancel a"
        " legitimate same-repo scan"
    )


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
    assert "os.path.realpath(path) == root" in sonar, (
        "a tracked symlink resolving to the workspace root itself (e.g."
        " loop -> .) passes the escape check but makes the scanner"
        " re-traverse the whole tree through the link recursively"
    )
    assert ": {escaping!r}" in sonar, (
        "tracked filenames may contain newlines and quotes: raw"
        " interpolation into ::error:: would allow log/annotation"
        " forging"
    )


def test_scan_bounds_artifact_before_and_during_extraction() -> None:
    steps = _sonar_steps()
    download = next(c for c in steps.values() if "ARTIFACT_ID" in c)
    size_gate_at = download.find("size_in_bytes")
    fetch_at = download.find("actions/artifacts/$ARTIFACT_ID/zip")
    assert "coverage artifact too large" in download, (
        "the API's size_in_bytes is the compressed archive size, but it"
        " still bounds the zip this job writes to disk before opening it"
    )
    assert size_gate_at != -1 and size_gate_at < fetch_at, (
        "the compressed size gate must run before the zip is fetched"
    )
    assert "exceeds 256 MiB while extracting" in download, (
        "a small zip can expand to many GiB; the uncompressed payload"
        " must be capped by counting bytes during extraction"
    )
    assert 'archive.open("coverage.xml")' in download, (
        "only the coverage.xml entry may be extracted; sibling entries"
        " are never written to the shared runner's disk"
    )
    assert "trap 'rm -f \"$ZIP_PATH\"' EXIT" in download, (
        "the zip cleanup must be registered before the fetch: under"
        " set -e a failed extraction otherwise leaves up to 256 MiB"
        " behind on the persistent runner for every failed run"
    )


def test_scan_unzips_artifacts_outside_the_workspace() -> None:
    sonar = _sonar_workflow()
    assert "runner.temp" in sonar, (
        "the artifact zip must land outside sonar.sources=. so extracted"
        " content is never traversed by the scanner"
    )
    assert "${{ runner.temp }}/coverage.xml" in sonar and (
        'os.environ["RUNNER_TEMP"]' in sonar
    ), (
        "coverage.xml must stay out of sonar.sources=.: inside the"
        " workspace the scanner would ingest it as a project file and"
        " pollute the analysis"
    )


def test_scan_refuses_empty_coverage_report() -> None:
    sonar = _sonar_workflow()
    assert "refusing to import an empty report" in sonar, (
        "if every <class> is pruned, a structurally valid but empty"
        " report gives an ambiguous outcome: fail loudly instead"
    )


def test_scan_rejects_coverage_entries_for_missing_files() -> None:
    sonar = _sonar_workflow()
    assert "os.path.isfile(os.path.join(root, name))" in sonar, (
        "every <class> filename must resolve to a real file in the"
        " workspace before the report is handed to SonarQube"
    )
    assert "classes.remove(cls)" in sonar and "package.remove(classes)" in sonar, (
        "CI measures the merge commit while the scan checks out the PR"
        " head, so genuine reports can reference files missing here;"
        " unmappable entries must be removed from the written report,"
        " not merely spared the error"
    )
    assert "escapes the workspace" in sonar, (
        "absolute and parent-traversing filenames stay fatal"
    )


def test_scan_anchors_coverage_source_to_the_scanned_tree() -> None:
    sonar = _sonar_workflow()
    assert 'source.text = "."' in sonar, (
        "coverage.py emits the CI runner's absolute workspace path in"
        " <source>; anchoring it to the scanned tree keeps the report"
        " portable across runner machines"
    )
    assert "ET.ElementTree(parsed).write(" in sonar, (
        "the anchored source must be written back for SonarQube to read"
    )


def test_scan_heredocs_run_in_isolated_python() -> None:
    sonar = _sonar_workflow()
    invocations = re.findall(r"^ *python3 .*", sonar, re.MULTILINE)
    assert invocations, "the scan job must keep using trusted stdlib python3"
    assert all(line.strip().startswith("python3 -I -") for line in invocations), (
        "a stdin heredoc runs with the untrusted checkout as sys.path[0]:"
        " a committed zipfile.py, subprocess.py or xml/ package would"
        " shadow the stdlib and execute PR code inside this"
        " token-bearing job. Isolated mode removes cwd from sys.path"
    )


def test_scan_rejects_nul_interleaved_encodings() -> None:
    sonar = _sonar_workflow()
    assert 'if b"\\x00" in data:' in sonar, (
        "a BOM-less UTF-16/32 report is valid UTF-8 (ASCII interleaved"
        " with NULs), so the substring scan misses the hidden <!DOCTYPE"
        " while expat auto-detects the encoding and expands its"
        " entities; NUL bytes must be rejected before the parse"
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
