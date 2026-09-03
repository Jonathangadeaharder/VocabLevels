"""Guard against foreign PYTHONPATH entries breaking test collection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_pytest_collect(pythonpath: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q"],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        check=False,
    )


def test_foreign_pythonpath_fails_collection_fast(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign-site-packages"
    foreign.mkdir()
    proc = run_pytest_collect(str(foreign))
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "PYTHONPATH" in output


def test_in_repo_pythonpath_entry_is_allowed() -> None:
    proc = run_pytest_collect(str(REPO_ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
