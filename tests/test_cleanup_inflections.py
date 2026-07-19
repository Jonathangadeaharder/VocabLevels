"""Tests for cleanup_inflections.py — remove duplicate/inflected rows."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import cleanup_inflections as ci


def _write_csv(path: Path, lemma_col: str, rows: list[list[str]]) -> None:
    header = [lemma_col, "English_Lemma", "Chinese_Lemma", "POS"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a repo using the real on-disk dual-pivot CSV shape."""
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    (tmp_path / "english").mkdir(exist_ok=True)
    for level in ci.LEVELS:
        _write_csv(
            tmp_path / "english" / f"{level}.csv",
            "English_Lemma",
            [
                ["cat", "cat", "", "NOUN"],
                ["cats", "cats", "", "NOUN"],
                ["dog", "dog", "", "NOUN"],
            ],
        )
    # "swedish" is deliberately left empty by default: the inflected-dedup
    # tests below build its per-level files themselves and would otherwise
    # have to account for a stray "wort" row bleeding into every level.
    (tmp_path / "swedish").mkdir(exist_ok=True)
    for other_lang in ("german", "spanish", "french", "dutch"):
        (tmp_path / other_lang).mkdir(exist_ok=True)
        cfg = ci.LANGS[other_lang]
        for level in ci.LEVELS:
            _write_csv(
                tmp_path / other_lang / f"{level}.csv",
                cfg["lemma_col"],
                [["wort", "word", "", "NOUN"]],
            )
    return tmp_path


class TestCleanupLanguage:
    def test_removes_plurals(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        total = ci.cleanup_language("english")
        assert total > 0
        path = tmp_repo / "english" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        assert "cats" not in lemmas
        assert "cat" in lemmas
        assert not (tmp_repo / "english" / ".A1.csv.tmp").exists()

    def test_no_removal_for_german_without_duplicates(self, tmp_repo: Path) -> None:
        total = ci.cleanup_language("german")
        assert total == 0

    def test_no_removal_for_spanish_without_duplicates(self, tmp_repo: Path) -> None:
        total = ci.cleanup_language("spanish")
        assert total == 0

    def test_removes_exact_intra_level_duplicate_for_german(
        self, tmp_repo: Path
    ) -> None:
        """Regression test: German had 12 verbatim (lemma, POS) duplicate
        rows within the same level (e.g. Antwort/kaufen/Jagd/Angst each
        appearing twice). cleanup_inflections.py must remove these for
        every language, not just English.
        """
        cfg = ci.LANGS["german"]
        lemma_col = cfg["lemma_col"]
        path = tmp_repo / "german" / "A1.csv"
        _write_csv(
            path,
            lemma_col,
            [
                ["Antwort", "answer", "", "NOUN"],
                ["Frage", "question", "", "NOUN"],
                # Verbatim duplicate of the first row (real bug pattern).
                ["Antwort", "answer", "", "NOUN"],
            ],
        )

        total = ci.cleanup_language("german")
        assert total == 1
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row[lemma_col] for row in csv.DictReader(f)]
        assert lemmas.count("Antwort") == 1
        assert "Frage" in lemmas

    def test_exact_duplicate_with_different_pos_is_kept(self, tmp_repo: Path) -> None:
        """Same lemma with a different POS (e.g. 'run' NOUN vs VERB) is a
        legitimate homonym pair, not a duplicate — must not be removed."""
        cfg = ci.LANGS["spanish"]
        lemma_col = cfg["lemma_col"]
        path = tmp_repo / "spanish" / "A1.csv"
        _write_csv(
            path,
            lemma_col,
            [
                ["gasto", "expense", "", "NOUN"],
                ["gasto", "expense", "", "VERB"],
            ],
        )

        total = ci.cleanup_language("spanish")
        assert total == 0
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_ss_words_not_removed(self, tmp_repo: Path) -> None:
        # Add "glass" and "glas" — "glass" ends in "ss" so should NOT be removed
        path = tmp_repo / "english" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(
            {
                "English_Lemma": "glass",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        rows.append(
            {
                "English_Lemma": "glas",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["English_Lemma", "Chinese_Lemma", "POS"]
            )
            writer.writeheader()
            writer.writerows(rows)

        ci.cleanup_language("english")
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        assert "glass" in lemmas

    def test_short_singular_not_removed(self, tmp_repo: Path) -> None:
        # Singular "ca" (2 chars) — should not trigger removal because len > 2 check
        path = tmp_repo / "english" / "A1.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({"English_Lemma": "ca", "Chinese_Lemma": "", "POS": "NOUN"})
        rows.append({"English_Lemma": "cas", "Chinese_Lemma": "", "POS": "NOUN"})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["English_Lemma", "Chinese_Lemma", "POS"]
            )
            writer.writeheader()
            writer.writerows(rows)

        ci.cleanup_language("english")
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["English_Lemma"] for row in csv.DictReader(f)]
        # "cas" should remain because singular "ca" is only 2 chars (len <= 2)
        assert "cas" in lemmas


class TestFindInflectedRemovals:
    """Pure clustering-logic tests using a fake tag_fn — no Stanza needed."""

    def test_inflected_form_removed_when_citation_form_present(
        self, tmp_repo: Path
    ) -> None:
        path = tmp_repo / "swedish" / "A1.csv"
        _write_csv(
            path,
            "Swedish_Lemma",
            [
                ["använda", "use", "", "VERB"],
                ["använder", "uses", "", "X"],
            ],
        )

        fake_lemmas = {
            "använda": ("använda", "VERB"),
            "använder": ("använda", "VERB"),
        }

        total = ci.remove_inflected_duplicates(
            "swedish", tag_fn=lambda w: fake_lemmas[w]
        )
        assert total == 1
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["Swedish_Lemma"] for row in csv.DictReader(f)]
        assert lemmas == ["använda"]

    def test_cross_level_inflected_form_removed(self, tmp_repo: Path) -> None:
        """Swedish 'beskrev' (C1) is a form of 'beskriva' (A1) — different
        levels, still a duplicate that must be removed."""
        a1 = tmp_repo / "swedish" / "A1.csv"
        c1 = tmp_repo / "swedish" / "C1.csv"
        _write_csv(a1, "Swedish_Lemma", [["beskriva", "describe", "", "VERB"]])
        _write_csv(c1, "Swedish_Lemma", [["beskrev", "described", "", "X"]])

        fake_lemmas = {
            "beskriva": ("beskriva", "VERB"),
            "beskrev": ("beskriva", "VERB"),
        }
        total = ci.remove_inflected_duplicates(
            "swedish", tag_fn=lambda w: fake_lemmas[w]
        )
        assert total == 1
        with c1.open(encoding="utf-8", newline="") as f:
            assert list(csv.DictReader(f)) == []
        with a1.open(encoding="utf-8", newline="") as f:
            lemmas = [row["Swedish_Lemma"] for row in csv.DictReader(f)]
        assert lemmas == ["beskriva"]

    def test_no_removal_when_citation_form_absent(self, tmp_repo: Path) -> None:
        """If the lemmatizer thinks a word is inflected but no citation-form
        row exists anywhere in the corpus, leave it alone (don't guess)."""
        path = tmp_repo / "swedish" / "A1.csv"
        _write_csv(path, "Swedish_Lemma", [["använder", "uses", "", "X"]])

        total = ci.remove_inflected_duplicates(
            "swedish", tag_fn=lambda w: ("använda", "VERB")
        )
        assert total == 0

    def test_ambiguous_multiple_self_rows_left_untouched(self, tmp_repo: Path) -> None:
        """Two rows that both claim to be their own citation form (e.g. two
        distinct homonyms) must not be collapsed — ambiguous, skip."""
        path = tmp_repo / "swedish" / "A1.csv"
        _write_csv(
            path,
            "Swedish_Lemma",
            [
                ["bank", "bank(finance)", "", "NOUN"],
                ["bank", "bank(river)", "", "NOUN"],
            ],
        )
        total = ci.remove_inflected_duplicates(
            "swedish", tag_fn=lambda w: ("bank", "NOUN")
        )
        assert total == 0

    def test_fuzzy_match_catches_near_miss_lemmatizer_typo(
        self, tmp_repo: Path
    ) -> None:
        """Regression for the real Swedish 'fortsätt' case: Stanza's own
        lemmatizer maps it to 'fortsäta' (one edit off the real citation
        form 'fortsätta'). A small edit-distance fallback must still catch
        this without relying on a hardcoded word list.
        """
        path = tmp_repo / "swedish" / "A1.csv"
        _write_csv(
            path,
            "Swedish_Lemma",
            [
                ["fortsätta", "continue", "", "VERB"],
                ["fortsätt", "continue(imperative)", "", "VERB"],
            ],
        )
        fake_lemmas = {
            "fortsätta": ("fortsätta", "VERB"),
            "fortsätt": ("fortsäta", "VERB"),  # lemmatizer typo, 1 edit off
        }
        total = ci.remove_inflected_duplicates(
            "swedish", tag_fn=lambda w: fake_lemmas[w]
        )
        assert total == 1
        with path.open(encoding="utf-8", newline="") as f:
            lemmas = [row["Swedish_Lemma"] for row in csv.DictReader(f)]
        assert lemmas == ["fortsätta"]

    def test_unsupported_language_is_noop(self, tmp_repo: Path) -> None:
        assert ci.remove_inflected_duplicates("arabic") == 0
        assert ci.remove_inflected_duplicates("chinese") == 0
