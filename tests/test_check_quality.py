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
        # Real on-disk header (commit c122a99): <Lang>_Lemma, English_Lemma,
        # Chinese_Lemma, POS. cfg["lemma_col"] already equals "English_Lemma"
        # for english and "Chinese_Lemma" for chinese, so no special-casing
        # is needed — a DictWriter with a duplicate fieldname simply writes
        # the same dict value into both positions, matching the real
        # self-referential pivot-column design.
        fields = [cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"]
        for level in cq.LEVELS:
            path = tmp_path / lang / f"{level}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                # Plain csv.writer (not DictWriter): for english/chinese,
                # fields[0] duplicates fields[1]/fields[2] by name, and a
                # dict literal can't hold two values under the same key, so
                # rows are built positionally to preserve both physical
                # columns' real on-disk values.
                writer = csv.writer(f)
                writer.writerow(fields)
                # Add enough rows to be close to target
                for i, seed_word in enumerate(SEED_WORDS):
                    lemma = f"{lang}{LEVEL_NAMES[level]}{seed_word}"
                    t1 = lemma if lang == "english" else f"translationone{i}"
                    t2 = lemma if lang == "chinese" else f"translationtwo{i}"
                    writer.writerow([lemma, t1, t2, "X"])
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


class TestPivotLanguageHeaders:
    """Regression tests for the English/Chinese dual-pivot header bug.

    The on-disk header for every language is
    ``[<Lang>_Lemma, English_Lemma, Chinese_Lemma, POS]``. For English and
    Chinese, the lemma column name collides with one of the two fixed
    translation columns (``English_Lemma``/``English_Lemma`` and
    ``Chinese_Lemma``/``Chinese_Lemma`` respectively). ``check_quality.py``
    must validate against this real on-disk shape (not the schema's
    ``trans_cols``, which never appear on disk for these two languages) and
    must read cells positionally so the duplicate column name never causes
    one physical column's data to be silently dropped.
    """

    def _write_raw_csv(
        self, path: Path, header: list[str], rows: list[list[str]]
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def test_english_real_ondisk_header_is_accepted(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Real on-disk shape: English_Lemma duplicated (lemma + self-translation).
        header = ["English_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]
        words = {
            "A1": ("one", "two"),
            "A2": ("three", "four"),
            "B1": ("five", "six"),
            "B2": ("seven", "eight"),
            "C1": ("nine", "ten"),
        }
        for level in cq.LEVELS:
            w1, w2 = words[level]
            self._write_raw_csv(
                tmp_repo / "english" / f"{level}.csv",
                header,
                [[w1, w1, "", "NOUN"], [w2, w2, "", "NOUN"]],
            )
        issues = cq.check_language("english")
        out = capsys.readouterr().out
        assert "bad header" not in out.lower()
        assert issues == 0

    def test_chinese_real_ondisk_header_is_accepted(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        header = ["Chinese_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]
        words = {
            "A1": ("一", "二"),
            "A2": ("三", "四"),
            "B1": ("五", "六"),
            "B2": ("七", "八"),
            "C1": ("九", "十"),
        }
        for level in cq.LEVELS:
            w1, w2 = words[level]
            self._write_raw_csv(
                tmp_repo / "chinese" / f"{level}.csv",
                header,
                [[w1, "one", w1, "NUM"], [w2, "two", w2, "NUM"]],
            )
        issues = cq.check_language("chinese")
        out = capsys.readouterr().out
        assert "bad header" not in out.lower()
        assert issues == 0

    def test_chinese_lemma_read_positionally_not_via_last_duplicate_column(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The real lemma is column 0; column 2 is a separate physical cell.

        A dict-keyed reader collapses both ``Chinese_Lemma`` columns into a
        single key and silently keeps only the last column's value. This
        reproduces the actual C1 "confronting" bug: the true lemma cell
        (column 0) carries the defect, while the identically-named
        column 2 is clean. The digit check must fire off column 0's value,
        proving the checker is not reading column 2 by name-collapse.
        """
        header = ["Chinese_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]
        rows = [
            # column 0 (real lemma) has a digit defect; column 2 is clean.
            ["confront1ng", "confront", "confronting", "VERB"],
            # column 0 is clean; column 2 (translation) has the defect instead.
            ["面对", "confront2", "confronting", "VERB"],
        ]
        for level in cq.LEVELS:
            self._write_raw_csv(tmp_repo / "chinese" / f"{level}.csv", header, rows)
        cq.check_language("chinese")
        out = capsys.readouterr().out
        assert "digits in lemma 'confront1ng'" in out
        assert "digits in lemma '面对'" not in out

    def test_missing_chinese_lemma_translation_is_warning_not_error(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = cq.LANGS["german"]
        path = tmp_repo / "german" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                cfg["lemma_col"]: "notranszh",
                "English_Lemma": "x",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[cfg["lemma_col"], "English_Lemma", "Chinese_Lemma", "POS"],
            )
            writer.writeheader()
            writer.writerows(rows)

        issues_before = cq.check_language("german")
        out = capsys.readouterr().out
        assert "WARN" in out
        assert "missing Chinese_Lemma" in out
        # Missing Chinese_Lemma alone must not count as an issue.
        assert issues_before == 0


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
