"""Structural integrity on committed CEFR lists + inventory-driven dialect policy."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANG_DIRS = (
    "english",
    "spanish",
    "french",
    "german",
    "dutch",
    "swedish",
    "arabic",
    "chinese",
)
LEVELS = ("A1", "A2", "B1", "B2", "C1")
INVENTORY = ROOT / "manual_reviews" / "arabic" / "dialect-residual-inventory.csv"
ALLOWED_VERDICTS = frozenset({"keep", "fix", "drop", "review"})


def _rows(lang_dir: str, level: str) -> list[dict[str, str]]:
    path = ROOT / lang_dir / f"{level}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lemma_key(row: dict[str, str]) -> str:
    return next(iter(row.keys()))


def _strip_ar_diacritics(value: str) -> str:
    return "".join(
        c
        for c in value
        if not ("\u064b" <= c <= "\u065f") and c not in "\u0670"
    )


def _load_inventory() -> list[dict[str, str]]:
    assert INVENTORY.exists(), INVENTORY
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_all_scale_tasks_succeeded() -> None:
    import sqlite3

    db = ROOT / ".gemma_qa" / "scale.sqlite3"
    assert db.exists()
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT language, level, status FROM scale_tasks WHERE phase='cefr'"
    ).fetchall()
    con.close()
    assert len(rows) == 40
    assert all(status == "succeeded" for _, _, status in rows)


def test_dialect_inventory_drops_absent_from_arabic_lists() -> None:
    inv = _load_inventory()
    drops = {row["lemma"] for row in inv if row.get("action") == "drop"}
    assert len(drops) >= 20, "inventory must be a non-trivial closed set"
    drop_stripped = {_strip_ar_diacritics(x) for x in drops}
    for level in LEVELS:
        rows = _rows("arabic", level)
        lk = _lemma_key(rows[0])
        for row in rows:
            lem = (row.get(lk) or "").strip()
            assert lem not in drops, (level, lem)
            assert _strip_ar_diacritics(lem) not in drop_stripped or lem in {
                "فم",
                "كلّي",
                "كلي",
            }, (level, lem)


def test_skeptic_maghrebi_residuals_absent() -> None:
    """Named skeptic hits must be covered by inventory application."""
    must_absent = {
        "راه",
        "بلاتي",
        "مبروك",
        "بنة",
        "طَاحَ",
        "طاح",
        "واش",
        "هادشي",
        "نيت",
        "يالله",
        "دراري",
    }
    for level in LEVELS:
        rows = _rows("arabic", level)
        lk = _lemma_key(rows[0])
        present = {(r.get(lk) or "").strip() for r in rows}
        present_s = {_strip_ar_diacritics(x) for x in present}
        for lem in must_absent:
            assert lem not in present, (level, lem)
            assert _strip_ar_diacritics(lem) not in present_s, (level, lem)


def test_german_wart_and_dutch_contracten_fixed() -> None:
    rows = _rows("german", "C1")
    lk = _lemma_key(rows[0])
    lemmas = {(r.get(lk) or "").strip() for r in rows}
    assert "wart" not in lemmas
    assert "sein" in lemmas

    rows = _rows("dutch", "C1")
    lk = _lemma_key(rows[0])
    lemmas = {(r.get(lk) or "").strip() for r in rows}
    assert "contracten" not in lemmas
    assert "contract" in lemmas


def test_post_fix_samples_valid_verdicts_and_no_unapplied_defects() -> None:
    """No forced all-keep: every row has a real verdict; fix/drop must not remain."""
    inv = _load_inventory()
    drops = {row["lemma"] for row in inv if row.get("action") == "drop"}
    total = 0
    for lang in LANG_DIRS:
        base = ROOT / "manual_reviews" / lang / "post-fix-p20"
        for level in LEVELS:
            path = base / f"{level}.sample-p20.csv"
            assert path.exists(), path
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert 13 <= len(rows) <= 30, (path, len(rows))
            for row in rows:
                total += 1
                verdict = (row.get("verdict") or "").strip()
                notes = (row.get("notes") or "").strip()
                assert verdict in ALLOWED_VERDICTS, (path, row)
                assert notes, (path, row)
                assert verdict not in {"fix", "drop"}, (
                    f"unapplied correctness defect in sample: {path} {row}"
                )
                if row.get("lang") == "ar" or lang == "arabic":
                    assert row.get("lemma") not in drops
                    if notes.startswith("policy:"):
                        assert verdict == "keep"
    assert total >= 800


def test_skeptic_non_citation_forms_absent() -> None:
    banned = {
        ("german", "C1"): {
            "meinten",
            "bräuchten",
            "Besonderes",
            "Heiliger",
            "Krachen",
            "Schwarzer",
            "nich",
            "wart",
        },
        ("german", "A1"): {"ander"},
        ("dutch", "C1"): {"uitdagingen", "contracten"},
        ("dutch", "B1"): {"honderden"},
        ("swedish", "B2"): {"los"},
        ("swedish", "B1"): {"unga"},
        ("french", "B1"): {"autoentrepreneur"},
    }
    for (lang, level), lemmas in banned.items():
        rows = _rows(lang, level)
        lk = _lemma_key(rows[0])
        present = {(r.get(lk) or "").strip() for r in rows}
        still = lemmas & present
        assert not still, f"{lang}/{level} still has {still}"


def test_skeptic_gloss_fixes_applied() -> None:
    rows = _rows("spanish", "C1")
    lk = _lemma_key(rows[0])
    emulo = next(r for r in rows if (r.get(lk) or "").strip() == "émulo")
    assert emulo["English_Lemma"] == "rival"

    rows = _rows("dutch", "A1")
    lk = _lemma_key(rows[0])
    zullen = next(r for r in rows if (r.get(lk) or "").strip() == "zullen")
    assert "shall" in zullen["English_Lemma"] or "will" in zullen["English_Lemma"]

    rows = _rows("german", "C1")
    lk = _lemma_key(rows[0])
    marge = next(r for r in rows if (r.get(lk) or "").strip() == "Marge")
    assert "利润率" in marge["Chinese_Lemma"] or "边际" in marge["Chinese_Lemma"]

    rows = _rows("arabic", "A2")
    lk = _lemma_key(rows[0])
    qad = next(r for r in rows if (r.get(lk) or "").strip() == "قد")
    assert "much" not in qad["English_Lemma"]

    rows = _rows("arabic", "C1")
    lk = _lemma_key(rows[0])
    kami = next(r for r in rows if (r.get(lk) or "").strip() == "كمي")
    assert kami["English_Lemma"] == "quantitative"


def test_no_latin_chinese_lemma_outside_chinese() -> None:
    for lang in LANG_DIRS:
        if lang == "chinese":
            continue
        for level in LEVELS:
            for row in _rows(lang, level):
                zh = (row.get("Chinese_Lemma") or "").strip()
                if not zh:
                    continue
                if re.search(r"[A-Za-z]", zh) and not re.search(
                    r"[\u4e00-\u9fff]", zh
                ):
                    pytest.fail(f"{lang}/{level} latin ZH: {row}")


def test_lemmas_are_nfc() -> None:
    for lang in LANG_DIRS:
        for level in LEVELS:
            rows = _rows(lang, level)
            lk = _lemma_key(rows[0])
            for row in rows:
                for key in (lk, "English_Lemma", "Chinese_Lemma"):
                    val = row.get(key) or ""
                    if not val:
                        continue
                    assert val == unicodedata.normalize("NFC", val), (
                        lang,
                        level,
                        key,
                        val,
                    )


def test_aggregate_has_no_inventory_drop_lemmas() -> None:
    inv = _load_inventory()
    drops = {row["lemma"] for row in inv if row.get("action") == "drop"}
    agg = ROOT / "manual_reviews" / "ALL-langs-p20-scored.csv"
    assert agg.exists()
    with agg.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row.get("lemma") not in drops
