"""Structural integrity checks on committed CEFR lists (real files on disk)."""

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


def _rows(lang_dir: str, level: str) -> list[dict[str, str]]:
    path = ROOT / lang_dir / f"{level}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lemma_key(row: dict[str, str]) -> str:
    return next(iter(row.keys()))


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


def test_skeptic_non_citation_forms_absent() -> None:
    """Documented fix/drop items must not remain as bad lemmas."""
    banned = {
        ("german", "C1"): {"meinten", "bräuchten", "Besonderes", "Heiliger", "Krachen", "Schwarzer"},
        ("german", "A1"): {"ander"},
        ("dutch", "C1"): {"uitdagingen"},
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
    assert zullen["English_Lemma"] != "would"

    rows = _rows("german", "C1")
    lk = _lemma_key(rows[0])
    marge = next(r for r in rows if (r.get(lk) or "").strip() == "Marge")
    assert "利润率" in marge["Chinese_Lemma"] or "边际" in marge["Chinese_Lemma"]

    rows = _rows("french", "B1")
    lk = _lemma_key(rows[0])
    assert any((r.get(lk) or "").strip() == "auto-entrepreneur" for r in rows)
    intit = next(r for r in rows if (r.get(lk) or "").strip() == "intituler")
    assert "title" in intit["English_Lemma"]


def test_no_latin_chinese_lemma_outside_chinese() -> None:
    for lang in LANG_DIRS:
        if lang == "chinese":
            continue
        for level in LEVELS:
            rows = _rows(lang, level)
            for row in rows:
                zh = (row.get("Chinese_Lemma") or "").strip()
                if not zh:
                    continue
                if re.search(r"[A-Za-z]", zh) and not re.search(r"[\u4e00-\u9fff]", zh):
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
                    assert val == unicodedata.normalize("NFC", val), (lang, level, key, val)


def test_post_fix_samples_exist_and_all_keep() -> None:
    for lang in LANG_DIRS:
        base = ROOT / "manual_reviews" / lang / "post-fix-p20"
        for level in LEVELS:
            path = base / f"{level}.sample-p20.csv"
            assert path.exists(), path
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert 20 <= len(rows) <= 30, (path, len(rows))
            for row in rows:
                assert row.get("verdict") == "keep", (path, row)
                assert row.get("notes"), row
