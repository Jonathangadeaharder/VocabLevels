"""Integrity: scale status, classifier inventory, pure sample scorer."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pytest

from scripts.gemma_qa.arabic_dialect import (
    classify_ar_lemma,
    load_inventory,
    scan_arabic_lists,
    score_sample_row,
)

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
LEVELS = ("A1", "A2", "B1", "B2", "Advanced")
INVENTORY = ROOT / "manual_reviews" / "arabic" / "dialect-residual-inventory.csv"
ALLOWED_VERDICTS = frozenset({"keep", "fix", "drop", "review"})


def _rows(lang_dir: str, level: str) -> list[dict[str, str]]:
    path = ROOT / lang_dir / f"{level}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lemma_key(row: dict[str, str]) -> str:
    return next(iter(row.keys()))


def test_all_scale_tasks_succeeded() -> None:
    import sqlite3

    db = ROOT / ".gemma_qa" / "scale.sqlite3"
    if not db.exists():
        pytest.skip("scale.sqlite3 not present (gitignored runner-local state)")
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT language, level, status FROM scale_tasks WHERE phase='cefr'"
    ).fetchall()
    con.close()
    assert len(rows) == 40
    assert all(status == "succeeded" for _, _, status in rows)


def test_inventory_matches_full_scan_policy_and_covers_classifier_drops() -> None:
    """Inventory is frozen classifier output: drops are closed lexicon;
    live scan is policy-only."""
    assert INVENTORY.exists()
    inv = load_inventory(INVENTORY)
    assert len(inv) >= 50
    drops = {
        r["lemma"] for r in inv if r.get("action") == "drop" and "#" not in r["lemma"]
    }
    assert len(drops) >= 40
    # Live full-population scan must not find drop residuals still in lists
    live = scan_arabic_lists(ROOT)
    leftover_drops = [r for r in live if r.action == "drop"]
    assert not leftover_drops, leftover_drops[:20]


def test_inventory_drop_lemmas_absent_from_arabic_lists() -> None:
    """No list row may classify as drop; MSA exceptions (روح NOUN) may stay."""
    for level in LEVELS:
        rows = _rows("arabic", level)
        lk = _lemma_key(rows[0])
        for row in rows:
            lem = (row.get(lk) or "").strip()
            en = (row.get("English_Lemma") or "").strip()
            up = (row.get("POS") or "").strip()
            result = classify_ar_lemma(lem, upos=up, english=en)
            assert result.action != "drop", (level, lem, up, en, result)


def test_classify_ar_lemma_hits_skeptic_class() -> None:
    """Classifier itself flags residual Maghrebi/loan class (not a name list test)."""
    samples = [
        ("بشوية", "ADV", "a little"),
        ("واقيلا", "ADV", "probably"),
        ("جاب", "VERB", "bring"),
        ("ناعس", "ADJ", "sleepy"),
        ("كمل", "VERB", "complete"),
        ("تغدى", "VERB", "lunch"),
        ("ميدة", "NOUN", "table"),
        ("بروفيل", "NOUN", "profile"),
        ("تداريب", "NOUN", "training"),
        ("كرعين", "NOUN", "trotters"),
        ("كساب", "NOUN", "breeder"),
        ("توحش", "VERB", "to miss someone"),
        ("بانيو", "NOUN", "bathtub"),
        ("زعما", "ADV", "supposedly"),
        ("زعفان", "ADJ", "angry"),
        ("خايف", "ADJ", "afraid"),
        ("عاود", "VERB", "to repeat"),
        ("جلبانة", "NOUN", "peas"),
        ("قمرون", "NOUN", "shrimp"),
        ("خدوم", "ADJ", "helpful"),
        ("دبلوم", "NOUN", "diploma"),
        ("شوفاج", "NOUN", "heater"),
        ("زعف", "VERB", "get angry"),
        ("حدش", "NUM", "eleven"),
        ("ضو", "NOUN", "light"),
        ("كونيكسيون", "NOUN", "connection"),
        ("مايكة", "NOUN", "plastic bag"),
        ("تقشاب", "NOUN", "joking"),
        ("جمركة", "NOUN", "customs clearance"),
        ("شباكية", "NOUN", "chebakia"),
        ("خاوي", "ADJ", "empty"),
        ("حيط", "NOUN", "wall"),
        ("مقرقب", "ADJ", "crunchy"),
        ("بوز", "NOUN", "buzz"),
        ("مخط", "VERB", "blow the nose"),
        ("إمتا", "ADV", "when"),
        ("تمانية", "NUM", "eight"),
        ("سنتيم", "NOUN", "centime"),
        ("فرملة", "NOUN", "brake"),
        ("ستوري", "NOUN", "story"),
        ("دغيا", "ADV", "quickly"),
        ("بزربة", "ADV", "in a hurry"),
        ("خربق", "VERB", "mess up"),
        ("تقاشر", "NOUN", "socks"),
        ("تلفزة", "NOUN", "television"),
        ("جاوب", "VERB", "answer"),
        ("وضب", "VERB", "to tidy"),
        ("ستاج", "NOUN", "internship"),
        ("ترام", "NOUN", "tram"),
    ]
    for lemma, upos, en in samples:
        result = classify_ar_lemma(lemma, upos=upos, english=en)
        assert result.action == "drop", (lemma, result)
    # MSA soul stays; Egyptian imperative روح as VERB drops
    assert classify_ar_lemma("روح", upos="NOUN", english="soul").action == "ok"
    assert classify_ar_lemma("روح", upos="VERB", english="go").action == "drop"
    assert classify_ar_lemma("بخير", upos="ADJ", english="fine").action == "ok"
    # eng-conditioned Maghrebi "or" (MSA أو); MSA "and not" stays
    assert classify_ar_lemma("ولا", upos="CCONJ", english="or").action == "drop"
    assert classify_ar_lemma("ولا", upos="PART", english="and not").action == "ok"


def test_score_sample_row_matches_sample_files() -> None:
    """Every post-fix sample row equals pure score_sample_row (no hand stamp)."""
    inv = load_inventory(INVENTORY)
    drops = {
        r["lemma"]
        for r in inv
        if r.get("action") == "drop" and "#" not in (r.get("lemma") or "")
    }
    policies = {r["lemma"] for r in inv if r.get("action") == "policy"}
    total = 0
    for lang_dir in LANG_DIRS:
        code = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "dutch": "nl",
            "swedish": "sv",
            "arabic": "ar",
            "chinese": "zh",
        }[lang_dir]
        for level in LEVELS:
            path = (
                ROOT
                / "manual_reviews"
                / lang_dir
                / "post-fix-p20"
                / f"{level}.sample-p20.csv"
            )
            assert path.exists(), path
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            assert 13 <= len(rows) <= 30, (path, len(rows))
            for row in rows:
                total += 1
                expected_v, expected_n = score_sample_row(
                    lang=code,
                    lemma=row.get("lemma") or "",
                    english_lemma=row.get("english_lemma") or "",
                    chinese_lemma=row.get("chinese_lemma") or "",
                    upos=row.get("upos") or "",
                    inventory_drops=drops,
                    inventory_policies=policies,
                )
                assert row.get("verdict") == expected_v, (path, row, expected_v)
                assert row.get("notes") == expected_n, (path, row, expected_n)
                assert expected_v in ALLOWED_VERDICTS
                # no bare-clean inventory hits
                if code == "ar":
                    assert (row.get("lemma") or "") not in drops
                    if expected_n.startswith("policy:"):
                        assert expected_v == "keep"
                # no unapplied correctness defects
                assert expected_v not in {"fix", "drop"}, (path, row)
    assert total >= 800


def test_german_wart_and_dutch_contracten_fixed() -> None:
    rows = _rows("german", "Advanced")
    lk = _lemma_key(rows[0])
    lemmas = {(r.get(lk) or "").strip() for r in rows}
    assert "wart" not in lemmas
    assert "sein" in lemmas

    rows = _rows("dutch", "Advanced")
    lk = _lemma_key(rows[0])
    lemmas = {(r.get(lk) or "").strip() for r in rows}
    assert "contracten" not in lemmas
    assert "contract" in lemmas


def test_citation_and_gloss_fixes_still_applied() -> None:
    rows = _rows("spanish", "Advanced")
    lk = _lemma_key(rows[0])
    emulo = next(r for r in rows if (r.get(lk) or "").strip() == "émulo")
    assert emulo["English_Lemma"] == "rival"

    rows = _rows("dutch", "A1")
    lk = _lemma_key(rows[0])
    zullen = next(r for r in rows if (r.get(lk) or "").strip() == "zullen")
    assert "shall" in zullen["English_Lemma"] or "will" in zullen["English_Lemma"]

    rows = _rows("german", "Advanced")
    lk = _lemma_key(rows[0])
    marge = next(r for r in rows if (r.get(lk) or "").strip() == "Marge")
    assert "利润率" in marge["Chinese_Lemma"] or "边际" in marge["Chinese_Lemma"]
    assert "nich" not in {(r.get(lk) or "").strip() for r in rows}

    rows = _rows("arabic", "A2")
    lk = _lemma_key(rows[0])
    qad = next(r for r in rows if (r.get(lk) or "").strip() == "قد")
    assert "much" not in qad["English_Lemma"]

    rows = _rows("arabic", "Advanced")
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
                    assert val == unicodedata.normalize("NFC", val), (
                        lang,
                        level,
                        key,
                        val,
                    )


def test_classify_ar_lemma_rules_and_edge_cases(tmp_path: Path) -> None:
    from scripts.gemma_qa.arabic_dialect import (
        InventoryRow,
        apply_inventory_to_arabic_lists,
        closed_lexicon_inventory,
        write_inventory,
    )

    # Empty
    assert classify_ar_lemma("").action == "ok"

    # UPOS conditional
    assert classify_ar_lemma("يعني", upos="PART").action == "drop"
    assert classify_ar_lemma("كون", upos="SCONJ").action == "drop"
    assert classify_ar_lemma("لازم", upos="AUX").action == "drop"
    assert classify_ar_lemma("سحاب", english="zipper").action == "drop"
    assert classify_ar_lemma("عمر", english="never").action == "drop"
    assert classify_ar_lemma("معمر", english="full").action == "drop"
    assert classify_ar_lemma("ولا", upos="CCONJ", english="or").action == "drop"
    assert classify_ar_lemma("كراج", english="garage").action == "drop"
    assert classify_ar_lemma("كلمة", english="colloquial word").action == "policy"

    # write_inventory / closed_lexicon_inventory
    inv_rows = [
        InventoryRow("ar", "A1", "بشوية", "slowly", "慢", "ADV", "drop", "reason"),
        InventoryRow(
            "ar", "A1", "بشوية", "slowly", "慢", "ADV", "drop", "reason"
        ),  # duplicate
    ]
    inv_file = tmp_path / "inventory.csv"
    write_inventory(inv_file, inv_rows)
    assert inv_file.exists()

    loaded = closed_lexicon_inventory()
    assert any(r.lemma == "بشوية" for r in loaded)

    # score_sample_row edge cases
    assert (
        score_sample_row(
            lang="ar",
            lemma="",
            english_lemma="en",
            chinese_lemma="zh",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="lemma\u0301",
            english_lemma="en",
            chinese_lemma="zh",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="lemma",
            english_lemma="en",
            chinese_lemma="latin_zh",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="°",
            english_lemma="en",
            chinese_lemma="度",
            upos="NOUN",
        )[0]
        == "drop"
    )
    assert (
        score_sample_row(
            lang="sv",
            lemma="los",
            english_lemma="loose",
            chinese_lemma="松",
            upos="ADJ",
        )[0]
        == "drop"
    )
    assert (
        score_sample_row(
            lang="sv",
            lemma="träffades",
            english_lemma="met",
            chinese_lemma="见",
            upos="VERB",
        )[0]
        == "drop"
    )
    assert (
        score_sample_row(
            lang="sv",
            lemma="mötas",
            english_lemma="meet",
            chinese_lemma="见",
            upos="VERB",
        )[0]
        == "drop"
    )
    assert (
        score_sample_row(
            lang="de",
            lemma="meinten",
            english_lemma="meant",
            chinese_lemma="以为",
            upos="VERB",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="nl",
            lemma="uitdagingen",
            english_lemma="challenges",
            chinese_lemma="挑战",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="es",
            lemma="émulo",
            english_lemma="emulator",
            chinese_lemma="仿真",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="nl",
            lemma="zullen",
            english_lemma="would",
            chinese_lemma="将",
            upos="VERB",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="قد",
            english_lemma="much",
            chinese_lemma="很多",
            upos="PART",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="كمي",
            english_lemma="quantum",
            chinese_lemma="量子",
            upos="ADJ",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="إلا",
            english_lemma="if",
            chinese_lemma="如果",
            upos="SCONJ",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="تكييف",
            english_lemma="qualification",
            chinese_lemma="资格",
            upos="NOUN",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="قضى",
            english_lemma="errand",
            chinese_lemma="差事",
            upos="VERB",
        )[0]
        == "fix"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="يعني",
            english_lemma="means",
            chinese_lemma="意思",
            upos="PART",
        )[0]
        == "drop"
    )
    assert (
        score_sample_row(
            lang="ar",
            lemma="حاجة",
            english_lemma="thing",
            chinese_lemma="东西",
            upos="NOUN",
            inventory_drops=["حاجة"],
        )[0]
        == "drop"
    )

    # apply_inventory_to_arabic_lists
    ar_dir = tmp_path / "arabic"
    ar_dir.mkdir(parents=True, exist_ok=True)
    (ar_dir / "A1.csv").write_text(
        "Arabic_Lemma,English_Lemma,Chinese_Lemma,POS\nبشوية,slowly,慢,ADV\nبيت,house,房,NOUN\n",
        encoding="utf-8",
    )
    apply_inventory_to_arabic_lists(
        tmp_path,
        [InventoryRow("ar", "A1", "بشوية", "slowly", "慢", "ADV", "drop", "test")],
    )
    a1_lines = (ar_dir / "A1.csv").read_text(encoding="utf-8").splitlines()
    assert len(a1_lines) == 2
    assert "بيت" in a1_lines[1]
