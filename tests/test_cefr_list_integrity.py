"""Structural integrity checks on committed CEFR lists and sample packs."""

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
    assert drops, "inventory must list drop lemmas"
    for level in LEVELS:
        rows = _rows("arabic", level)
        lk = _lemma_key(rows[0])
        present = {(r.get(lk) or "").strip() for r in rows}
        still = drops & present
        assert not still, f"arabic/{level} still has inventory drops: {still}"


def test_dialect_inventory_policy_labeled_in_lists() -> None:
    inv = _load_inventory()
    policies = [row for row in inv if row.get("action") == "policy"]
    for item in policies:
        level = item["level"]
        lemma = item["lemma"]
        rows = _rows("arabic", level)
        lk = _lemma_key(rows[0])
        match = [r for r in rows if (r.get(lk) or "").strip() == lemma]
        if not match:
            continue  # optional if level-specific
        en = (match[0].get("English_Lemma") or "").lower()
        zh = match[0].get("Chinese_Lemma") or ""
        assert (
            "colloquial" in en or "dialect" in en or "口语" in zh or "方言" in zh
        ), (level, lemma, match[0])


def test_post_fix_samples_have_valid_verdicts_not_forced_all_keep() -> None:
    """Every sample row scored; policy keeps use policy: notes; no rubber stamp rule."""
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
                if verdict == "keep" and notes.startswith("policy:"):
                    continue
                if verdict == "keep":
                    # clean keep must not be an inventory drop lemma
                    inv = _load_inventory()
                    drops = {r["lemma"] for r in inv if r.get("action") == "drop"}
                    assert row.get("lemma") not in drops, (path, row)
                # correctness defects must not remain as keep/clean
                if verdict in {"fix", "drop"}:
                    pytest.fail(
                        f"unapplied correctness defect still in sample: {path} {row}"
                    )
    assert total >= 800


def test_post_fix_policy_lemmas_use_policy_notes() -> None:
    inv = _load_inventory()
    policies = {row["lemma"] for row in inv if row.get("action") == "policy"}
    if not policies:
        return
    for level in LEVELS:
        path = (
            ROOT
            / "manual_reviews"
            / "arabic"
            / "post-fix-p20"
            / f"{level}.sample-p20.csv"
        )
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("lemma") in policies:
                assert (row.get("notes") or "").startswith("policy:"), row
                assert row.get("verdict") == "keep"


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
        },
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
    lemmas = {(r.get(lk) or "").strip() for r in rows}
    assert "nicht" in lemmas

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
            rows = _rows(lang, level)
            for row in rows:
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


def test_aggregate_p20_has_no_inventory_drop_lemmas() -> None:
    inv = _load_inventory()
    drops = {row["lemma"] for row in inv if row.get("action") == "drop"}
    policies = {row["lemma"] for row in inv if row.get("action") == "policy"}
    agg = ROOT / "manual_reviews" / "ALL-langs-p20-scored.csv"
    assert agg.exists()
    with agg.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row.get("lemma") not in drops
        if row.get("lang") == "ar" and row.get("lemma") in policies:
            notes = row.get("notes") or ""
            assert notes.startswith("policy:") or "colloquial" in (
                row.get("english_lemma") or ""
            ).lower()
