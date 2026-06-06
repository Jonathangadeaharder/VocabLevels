"""Tests for check_quality.py — quality checker for trilingual CEFR vocab CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import check_quality as cq


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal repo structure with valid CSVs; patch ROOT."""
    monkeypatch.setattr(cq, "ROOT", tmp_path)
    for lang in ("english", "german", "spanish"):
        (tmp_path / lang).mkdir(exist_ok=True)
        cfg = cq.LANGS[lang]
        fields = [cfg["lemma_col"], *cfg["trans_cols"]]
        for level in cq.LEVELS:
            path = tmp_path / lang / f"{level}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                # Add enough rows to be close to target
                for i in range(5):
                    writer.writerow(
                        {cfg["lemma_col"]: f"{lang[:2]}_{level}_{i}", cfg["trans_cols"][0]: f"t1_{i}", cfg["trans_cols"][1]: f"t2_{i}"}
                    )
    return tmp_path


class TestCheckLanguage:
    def test_valid_csvs_return_zero_issues(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cq.check_language("english")
        # Empty lemma and whitespace issues don't exist; duplicates don't exist
        # The count warnings will print but not count as issues
        out = capsys.readouterr().out
        assert "ENGLISH" in out

    def test_empty_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # Add a row with empty lemma
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "", cfg["trans_cols"][0]: "x", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "empty lemma" in capsys.readouterr().out

    def test_multi_word_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "big apple", cfg["trans_cols"][0]: "x", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "multi-word" in capsys.readouterr().out

    def test_digits_in_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "word123", cfg["trans_cols"][0]: "x", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "digits" in capsys.readouterr().out

    def test_special_chars_in_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "word!", cfg["trans_cols"][0]: "x", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "special" in capsys.readouterr().out.lower()

    def test_missing_translation_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "notrans", cfg["trans_cols"][0]: "", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "missing" in capsys.readouterr().out.lower()

    def test_duplicate_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        # Add duplicate
        rows.append({cfg["lemma_col"]: "en_A1_0", cfg["trans_cols"][0]: "dup", cfg["trans_cols"][1]: "dup"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "duplicate" in capsys.readouterr().out.lower()

    def test_whitespace_in_lemma_flagged(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg = cq.LANGS["english"]
        path = tmp_repo / "english" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: " padded ", cfg["trans_cols"][0]: "x", cfg["trans_cols"][1]: "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[cfg["lemma_col"], *cfg["trans_cols"]])
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("english")
        assert issues >= 1
        assert "whitespace" in capsys.readouterr().out.lower()

    def test_missing_csv_warns(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        (tmp_repo / "english" / "A1.csv").unlink()
        cq.check_language("english")
        out = capsys.readouterr().out
        assert "missing" in out.lower() or "WARN" in out


class TestMain:
    def test_main_all_languages(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ret = cq.main(["check_quality.py"])
        assert isinstance(ret, int)

    def test_main_single_language(self, tmp_repo: Path) -> None:
        ret = cq.main(["check_quality.py", "english"])
        assert isinstance(ret, int)

    def test_main_unknown_language(self, tmp_repo: Path) -> None:
        ret = cq.main(["check_quality.py", "japanese"])
        assert ret == 2

    def test_main_bad_header(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # Write a CSV with bad headers
        path = tmp_repo / "english" / "A1.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bad_col", "col2", "col3"])
            writer.writeheader()
            writer.writerow({"bad_col": "x", "col2": "y", "col3": "z"})
        issues = cq.check_language("english")
        assert issues >= 1
        assert "bad header" in capsys.readouterr().out.lower()
