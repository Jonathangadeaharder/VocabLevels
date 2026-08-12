"""Every CEFR CSV row must be upsertable: 4 columns, unique (lemma, POS).

Vidiom seeds these files with ON CONFLICT (lemma, pos); a repeated key raises
PostgreSQL cardinality_violation 21000 and aborts the whole seed round.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from vocab_schema import LANGS, LEVELS

ROOT = Path(__file__).parent.parent
COLUMNS = 4


def _csv_paths() -> list[Path]:
    return [
        path
        for lang in LANGS
        for level in LEVELS
        if (path := ROOT / lang / f"{level}.csv").exists()
    ]


@pytest.mark.parametrize("path", _csv_paths(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_rows_have_four_columns(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        offenders = [
            (idx, row)
            for idx, row in enumerate(csv.reader(handle), start=1)
            if row and len(row) != COLUMNS
        ]
    assert not offenders, f"{path}: rows with != {COLUMNS} columns: {offenders[:5]}"


@pytest.mark.parametrize("path", _csv_paths(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_lemma_pos_keys_are_unique(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        keys = Counter(
            (row[0].strip(), row[3].strip())
            for row in csv.reader(handle)
            if row and len(row) == COLUMNS
        )
    duplicates = {key: count for key, count in keys.items() if count > 1}
    assert not duplicates, f"{path}: duplicate (lemma, POS) keys: {duplicates}"
