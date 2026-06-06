"""Tests for cleanup_inflections.py — remove inflected forms from vocab CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import cleanup_inflections as ci


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a repo with both lemmas and their plural forms."""
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    lang = "english"
    (tmp_path / lang).mkdir(exist_ok=True)
    lemma_col = "English_Lemma"
    trans_cols = ("German_Translation", "Spanish_Translation")
    fields = [lemma_col, *trans_cols]
    for level in ci.LEVELS:
        path = tmp_path / lang / f"{level}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({lemma_col: "cat", trans_cols[0]: "Katze", trans_cols[1]: "gato"})
            writer.writerow({lemma_col: "cats", trans_cols[0]: "Katzen", trans_cols[1]: "gatos"})
            writer.writerow({lemma_col: "dog", trans_cols[0]: "Hund", trans_cols[1]: "perro"})
    # Also create german/spanish dirs (minimal)
    for other_lang in ("german", "spanish"):
        (tmp_path / other_lang).mkdir(exist_ok=True)
        ocol = {"german": "German_Lemma", "spanish": "Spanish_Lemma"}[other_lang]
        otrans = ci.TRANS_COLS[other_lang]
        ofields = [ocol, *otrans]
        for level in ci.LEVELS:
            opath = tmp_path / other_lang / f"{level}.csv"
            with opath.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ofields)
                writer.writeheader()
                writer.writerow({ocol: "wort", otrans[0]: "word", otrans[1]: "palabra"})
    return tmp_path


class TestCleanupLanguage:
    def test_removes_plurals(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        total = ci.cleanup_language("english")
        assert total > 0
        # Verify "cats" was removed
        path = tmp_repo / "english" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        assert "cats" not in lemmas
        assert "cat" in lemmas

    def test_no_removal_for_german(self, tmp_repo: Path) -> None:
        # German cleanup is not implemented yet (only english)
        total = ci.cleanup_language("german")
        assert total == 0

    def test_no_removal_for_spanish(self, tmp_repo: Path) -> None:
        total = ci.cleanup_language("spanish")
        assert total == 0

    def test_ss_words_not_removed(self, tmp_repo: Path) -> None:
        # Add "glass" and "glas" — "glass" ends in "ss" so should NOT be removed
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({"English_Lemma": "glass", "German_Translation": "Glas", "Spanish_Translation": "vidrio"})
        rows.append({"English_Lemma": "glas", "German_Translation": "Glas2", "Spanish_Translation": "vidrio2"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["English_Lemma", "German_Translation", "Spanish_Translation"])
            writer.writeheader()
            writer.writerows(rows)

        ci.cleanup_language("english")
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        assert "glass" in lemmas

    def test_short_singular_not_removed(self, tmp_repo: Path) -> None:
        # Singular "ca" (2 chars) — should not trigger removal because len > 2 check
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({"English_Lemma": "ca", "German_Translation": "x", "Spanish_Translation": "x"})
        rows.append({"English_Lemma": "cas", "German_Translation": "y", "Spanish_Translation": "y"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["English_Lemma", "German_Translation", "Spanish_Translation"])
            writer.writeheader()
            writer.writerows(rows)

        ci.cleanup_language("english")
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        # "cas" should remain because singular "ca" is only 2 chars (len <= 2)
        assert "cas" in lemmas
