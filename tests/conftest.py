"""Fail fast when ambient PYTHONPATH leaks into the test environment.

Machine-wide PYTHONPATH exports (e.g. an unrelated py3.11 venv set up by
the interactive shell) shadow this repo's venv and break pytest collection
with misleading import errors. Refuse to run instead of silently scrubbing
the environment; the operator decides how to launch the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def foreign_entries(raw_pythonpath: str, repo_root: Path) -> list[str]:
    """Return PYTHONPATH entries that point outside the repo root."""
    return [
        entry
        for entry in raw_pythonpath.split(os.pathsep)
        if entry and not Path(os.path.abspath(entry)).is_relative_to(repo_root)
    ]


_foreign = foreign_entries(os.environ.get("PYTHONPATH", ""), _REPO_ROOT)
if _foreign:
    raise RuntimeError(
        "Refusing to collect tests with a polluted PYTHONPATH.\n"
        f"Foreign entries: {os.pathsep.join(_foreign)}\n"
        "Unset it first, e.g.: env -u PYTHONPATH uv run python -m pytest"
    )
