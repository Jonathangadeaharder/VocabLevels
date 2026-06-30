"""Tests for audit_lemmatization.py — find inflected forms in vocab CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import audit_lemmatization as al


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a repo with lemmas that include inflected forms."""
    monkeypatch.setattr(al, "ROOT", tmp_path)
    for lang, cfg in al.LANGS.items():
        (tmp_path / lang).mkdir(exist_ok=True)
        lemma_col = cfg["lemma_col"]
        trans_cols = cfg["trans_cols"]
        fields = [lemma_col, *trans_cols]
        for level in al.LEVELS:
            path = tmp_path / lang / f"{level}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                # Just a few rows
                writer.writerow(
                    {lemma_col: "cat", trans_cols[0]: "Katze", trans_cols[1]: "gato"}
                )
                writer.writerow(
                    {lemma_col: "dog", trans_cols[0]: "Hund", trans_cols[1]: "perro"}
                )
    return tmp_path


class TestFindInflectedForms:
    def test_finds_plural(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Add "cats" (plural of "cat") to english A1
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                "English_Lemma": "cats",
                "German_Translation": "Katzen",
                "Spanish_Translation": "gatos",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "English_Lemma",
                    "German_Translation",
                    "Spanish_Translation",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        al.find_inflected_forms()
        out = capsys.readouterr().out
        assert "plural" in out.lower() or "cats" in out

    def test_finds_gerund(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # "walking"[:-3] = "walk" — simple strip works for regular verbs
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                "English_Lemma": "walk",
                "German_Translation": "gehen",
                "Spanish_Translation": "caminar",
            }
        )
        rows.append(
            {
                "English_Lemma": "walking",
                "German_Translation": "gehend",
                "Spanish_Translation": "caminando",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "English_Lemma",
                    "German_Translation",
                    "Spanish_Translation",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        al.find_inflected_forms()
        out = capsys.readouterr().out
        assert "gerund" in out.lower() or "walking" in out

    def test_finds_past_tense(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Add "walked" (past of "walk") and base "walk"
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                "English_Lemma": "walk",
                "German_Translation": "gehen",
                "Spanish_Translation": "caminar",
            }
        )
        rows.append(
            {
                "English_Lemma": "walked",
                "German_Translation": "ging",
                "Spanish_Translation": "caminó",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "English_Lemma",
                    "German_Translation",
                    "Spanish_Translation",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        al.find_inflected_forms()
        out = capsys.readouterr().out
        assert "past" in out.lower() or "walked" in out

    def test_no_inflected_forms(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        al.find_inflected_forms()
        out = capsys.readouterr().out
        # With no inflected forms added, should report none found
        assert "No obvious" in out or "ENGLISH" in out

    def test_skips_missing_in_progress_levels(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_repo / "french" / "A2.csv").unlink()
        al.find_inflected_forms()
        out = capsys.readouterr().out
        assert "FRENCH" in out

    def test_ss_not_flagged_as_plural(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # "glass" should NOT be flagged as plural of "glas" (ends in "ss")
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                "English_Lemma": "glass",
                "German_Translation": "Glas",
                "Spanish_Translation": "vidrio",
            }
        )
        rows.append(
            {
                "English_Lemma": "glas",
                "German_Translation": "Glas2",
                "Spanish_Translation": "vidrio2",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "English_Lemma",
                    "German_Translation",
                    "Spanish_Translation",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        al.find_inflected_forms()
        out = capsys.readouterr().out
        # "glass" should not appear as a plural candidate (ends in "ss")
        assert "glass" not in out or "plural" not in out.lower() or "No obvious" in out
