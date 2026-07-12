"""Tests for check_quality.py — quality checker for multilingual CEFR vocab CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import check_quality as cq

LEVEL_NAMES = {"A1": "one", "A2": "two", "B1": "three", "B2": "four", "C1": "five"}
SEED_WORDS = ("alpha", "bravo", "charlie", "delta", "echo")


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal repo structure with valid CSVs; patch ROOT."""
    monkeypatch.setattr(cq, "ROOT", tmp_path)
    for lang, cfg in cq.LANGS.items():
        (tmp_path / lang).mkdir(exist_ok=True)
        # Harmonized dual-pivot header (commit c122a99):
        # <Lang>_Lemma, English_Lemma, Chinese_Lemma, POS
        # Pivot languages use schema trans_cols to avoid self-reference.
        if lang in ("english", "chinese"):
            fields = [cfg["lemma_col"], *cfg["trans_cols"], "POS"]
        else:
            fields = [cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"]
        for level in cq.LEVELS:
            path = tmp_path / lang / f"{level}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                # Add enough rows to be close to target
                for i, seed_word in enumerate(SEED_WORDS):
                    lemma = f"{lang}{LEVEL_NAMES[level]}{seed_word}"
                    writer.writerow(
                        {
                            fields[0]: lemma,
                            fields[1]: f"translationone{i}",
                            fields[2]: f"translationtwo{i}",
                            "POS": "X",
                        }
                    )
    return tmp_path


class TestCheckLanguage:
    def test_valid_csvs_return_zero_issues(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        issues = cq.check_language("german")
        out = capsys.readouterr().out
        assert issues == 0
        assert "GERMAN" in out

    def test_empty_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Add a row with empty lemma
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({cfg["lemma_col"]: "", "English_Lemma": "x", "Chinese_Lemma": "x"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "empty lemma" in capsys.readouterr().out

    def test_multi_word_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: "big apple",
                "English_Lemma": "x",
                "Chinese_Lemma": "x",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "multi-word" in capsys.readouterr().out

    def test_digits_in_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: "word123",
                "English_Lemma": "x",
                "Chinese_Lemma": "x",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "digits" in capsys.readouterr().out

    def test_special_chars_in_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: "word!",
                "English_Lemma": "x",
                "Chinese_Lemma": "x",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "special" in capsys.readouterr().out.lower()

    def test_missing_translation_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: "notrans",
                "English_Lemma": "",
                "Chinese_Lemma": "x",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "missing" in capsys.readouterr().out.lower()

    def test_duplicate_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        # Add duplicate
        rows.append(
            {
                cfg["lemma_col"]: "germanonealpha",
                "English_Lemma": "duplicate",
                "Chinese_Lemma": "duplicate",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "duplicate" in capsys.readouterr().out.lower()

    def test_whitespace_in_lemma_flagged(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: " padded ",
                "English_Lemma": "x",
                "Chinese_Lemma": "x",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)
        issues = cq.check_language("german")
        assert issues >= 1
        assert "whitespace" in capsys.readouterr().out.lower()

    def test_missing_csv_warns(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_repo / "german" / "A1.csv").unlink()
        cq.check_language("german")
        out = capsys.readouterr().out
        assert "missing" in out.lower() or "WARN" in out

    def test_shared_translation_details_are_opt_in(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[1]["English_Lemma"] = rows[0]["English_Lemma"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)

        assert cq.check_language("german") == 0
        default_out = capsys.readouterr().out
        assert "shared translation groups" in default_out
        assert "shared by" not in default_out

        assert cq.check_language("german", show_shared_translations=True) == 0
        detailed_out = capsys.readouterr().out
        assert "shared by" in detailed_out


class TestMain:
    def test_schema_is_shared_with_vocab_manager(self) -> None:
        import vocab_manager

        assert cq.LANGS is vocab_manager.LANGS

    def test_main_all_languages(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = cq.main(["check_quality.py"])
        assert isinstance(ret, int)

    def test_main_single_language(self, tmp_repo: Path) -> None:
        ret = cq.main(["check_quality.py", "german"])
        assert isinstance(ret, int)

    def test_main_includes_new_languages(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = cq.main(["check_quality.py", "french", "swedish"])
        out = capsys.readouterr().out
        assert ret == 0
        assert "FRENCH" in out
        assert "SWEDISH" in out

    def test_main_accepts_shared_translation_detail_flag(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = cq.main(["check_quality.py", "--show-shared-translations", "german"])
        out = capsys.readouterr().out
        assert ret == 0
        assert "GERMAN" in out

    def test_main_unknown_language(self, tmp_repo: Path) -> None:
        ret = cq.main(["check_quality.py", "japanese"])
        assert ret == 2

    def test_main_bad_header(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Write a CSV with bad headers
        path = tmp_repo / "german" / "A1.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bad_col", "col2", "col3"])
            writer.writeheader()
            writer.writerow({"bad_col": "x", "col2": "y", "col3": "z"})
        issues = cq.check_language("german")
        assert issues >= 1
        assert "bad header" in capsys.readouterr().out.lower()
