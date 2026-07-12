"""Tests for fix_pos_and_overflow.py.

Stanza models are heavy (~1-2 GB). Tests that need a real tagger are
marked with @pytest.mark.slow and skipped unless --run-stanza is passed.
Pure-logic tests (gloss fallback, CSV I/O, dedup, overflow detection)
run without stanza.
"""

from __future__ import annotations

import pytest

import sys

from fix_pos_and_overflow import (
    ChangeSet,
    analyze_redundant_overflow,
    dedup_after_pos_fix,
    gloss_fallback,
    load_csv,
    save_csv,
)


# --- Gloss fallback (no stanza needed) ------------------------------------


class TestGlossFallback:
    def test_to_prefix_returns_verb(self):
        assert gloss_fallback("to run") == "VERB"
        assert gloss_fallback("to announce") == "VERB"

    def test_hyphenated_adj_with_dash_suffix_returns_adj(self):
        # Patterns ending in "-ed" (with leading hyphen) match.
        assert gloss_fallback("self-ed") == "ADJ"
        assert gloss_fallback("cross-ing") == "ADJ"

    def test_plain_words_return_empty(self):
        # Conservative: plain words without "to " prefix or explicit
        # "-ed/-ing/-ous/-ish" (hyphenated) suffix return empty.
        assert gloss_fallback("hidden") == ""
        assert gloss_fallback("flashing") == ""
        assert gloss_fallback("abundant") == ""
        assert gloss_fallback("yellowish") == ""
        assert gloss_fallback("journalist") == ""
        assert gloss_fallback("saddle") == ""
        assert gloss_fallback("gray-haired") == ""

    def test_empty_gloss_returns_empty(self):
        assert gloss_fallback("") == ""


# --- CSV round-trip (no stanza needed) ------------------------------------


class TestCsvIO:
    def test_load_and_save_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fix_pos_and_overflow.ROOT", tmp_path)
        lang_dir = tmp_path / "german"
        lang_dir.mkdir()
        rows = [
            {
                "German_Lemma": "Haus",
                "English_Lemma": "house",
                "Chinese_Lemma": "房子",
                "POS": "NOUN",
            },
            {
                "German_Lemma": "laufen",
                "English_Lemma": "to run",
                "Chinese_Lemma": "跑",
                "POS": "VERB",
            },
        ]
        save_csv("german", "A1", rows)
        loaded = load_csv("german", "A1")
        assert len(loaded) == 2
        assert loaded[0]["German_Lemma"] == "Haus"
        assert loaded[1]["POS"] == "VERB"


# --- Redundant overflow detection (no stanza needed) ----------------------


class TestRedundantOverflow:
    def test_redundant_overflow_staged(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fix_pos_and_overflow.ROOT", tmp_path)
        # Use a small target so we don't need 600 rows.
        monkeypatch.setattr(
            "fix_pos_and_overflow.TARGETS",
            {"A1": 4, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000},
        )
        lang_dir = tmp_path / "french"
        lang_dir.mkdir()
        # 4 target rows + 2 overflow where lemma is already in target.
        rows = [
            {
                "French_Lemma": "forme",
                "English_Lemma": "form",
                "Chinese_Lemma": "",
                "POS": "X",
            },
            {
                "French_Lemma": "garde",
                "English_Lemma": "guard",
                "Chinese_Lemma": "",
                "POS": "X",
            },
            {
                "French_Lemma": "autre",
                "English_Lemma": "other",
                "Chinese_Lemma": "",
                "POS": "ADJ",
            },
            {
                "French_Lemma": "mot",
                "English_Lemma": "word",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            },
            # overflow (lemma already in target zone):
            {
                "French_Lemma": "forme",
                "English_Lemma": "form",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            },
            {
                "French_Lemma": "garde",
                "English_Lemma": "guard",
                "Chinese_Lemma": "",
                "POS": "VERB",
            },
        ]
        save_csv("french", "A1", rows)

        cs = ChangeSet()
        analyze_redundant_overflow("french", cs)
        assert len(cs.trims) == 2
        assert all(t.reason.startswith("redundant") for t in cs.trims)

    def test_unique_overflow_not_staged(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fix_pos_and_overflow.ROOT", tmp_path)
        lang_dir = tmp_path / "french"
        lang_dir.mkdir()
        rows = [
            {
                "French_Lemma": f"mot{i}",
                "English_Lemma": f"word{i}",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
            for i in range(4)
        ]
        # overflow with unique lemma (not in target zone):
        rows.append(
            {
                "French_Lemma": "unique",
                "English_Lemma": "unique",
                "Chinese_Lemma": "",
                "POS": "ADJ",
            }
        )
        save_csv("french", "A1", rows)

        cs = ChangeSet()
        analyze_redundant_overflow("french", cs)
        assert len(cs.trims) == 0


# --- Dedup after POS fix (no stanza needed) -------------------------------


class TestDedupAfterPosFix:
    def test_dedup_skips_files_at_target(self, tmp_path, monkeypatch):
        """Files at/under target keep dupes (no trim staged)."""
        monkeypatch.setattr("fix_pos_and_overflow.ROOT", tmp_path)
        lang_dir = tmp_path / "german"
        lang_dir.mkdir()
        rows = [
            {
                "German_Lemma": f"wort{i}",
                "English_Lemma": f"word{i}",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
            for i in range(598)
        ]
        # 2 duplicate lemma+POS at end (file exactly at target 600):
        rows.append(
            {
                "German_Lemma": "wort0",
                "English_Lemma": "word0",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        rows.append(
            {
                "German_Lemma": "wort1",
                "English_Lemma": "word1",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        save_csv("german", "A1", rows)

        cs = ChangeSet()
        dedup_after_pos_fix(["german"], cs)
        assert len(cs.trims) == 0  # at target, keep dupes

    def test_dedup_trims_over_target(self, tmp_path, monkeypatch):
        """Files over target: trim lemma+POS duplicates in overflow."""
        monkeypatch.setattr("fix_pos_and_overflow.ROOT", tmp_path)
        lang_dir = tmp_path / "german"
        lang_dir.mkdir()
        rows = [
            {
                "German_Lemma": f"wort{i}",
                "English_Lemma": f"word{i}",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
            for i in range(598)
        ]
        # 4 overflow rows, 2 are lemma+POS dupes of target zone:
        rows.append(
            {
                "German_Lemma": "wort0",
                "English_Lemma": "word0",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        rows.append(
            {
                "German_Lemma": "wort1",
                "English_Lemma": "word1",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        rows.append(
            {
                "German_Lemma": "extra1",
                "English_Lemma": "extra1",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        rows.append(
            {
                "German_Lemma": "extra2",
                "English_Lemma": "extra2",
                "Chinese_Lemma": "",
                "POS": "NOUN",
            }
        )
        save_csv("german", "A1", rows)

        cs = ChangeSet()
        dedup_after_pos_fix(["german"], cs)
        # wort0 and wort1 are dupes (first occurrence at L2/L3, second
        # at L600/L601). Should stage 2 trims.
        assert len(cs.trims) == 2
        assert all("duplicate" in t.reason for t in cs.trims)


# --- Stanza integration (slow, needs models) ------------------------------


def run_stanza():
    return pytest.mark.skipif(
        "--run-stanza" not in sys.argv,
        reason="needs --run-stanza (downloads stanza models)",
    )


@pytest.mark.slow
@run_stanza()
class TestStanzaTagging:
    def test_german_noun(self):
        from fix_pos_and_overflow import tag_lemma

        assert tag_lemma("german", "Frage") == "NOUN"
        assert tag_lemma("german", "Spiel") == "NOUN"

    def test_german_verb(self):
        from fix_pos_and_overflow import tag_lemma

        assert tag_lemma("german", "arbeiten") == "VERB"
        assert tag_lemma("german", "leben") == "VERB"

    def test_german_aux(self):
        from fix_pos_and_overflow import tag_lemma

        assert tag_lemma("german", "bin") == "AUX"

    def test_arabic_noun(self):
        from fix_pos_and_overflow import tag_lemma

        assert tag_lemma("arabic", "محامي") == "NOUN"
        assert tag_lemma("arabic", "مواطن") == "NOUN"

    def test_arabic_dialectal_returns_empty(self):
        """Darija words Stanza can't classify return empty (X)."""
        from fix_pos_and_overflow import tag_lemma

        assert tag_lemma("arabic", "نتي") == ""
