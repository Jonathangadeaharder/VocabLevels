"""Tests for build_contract_delivery, the harmonization pipeline.

The pipeline turns the authored CEFR CSVs into contract TSV deliveries:
concept alignment across languages, canonical English gloss per concept,
dedup on (lang, lemma, gloss_norm), one rank-1 per (lang, gloss, pos),
and a stable concept_key across languages.
"""

from __future__ import annotations

from pathlib import Path

from build_contract_delivery import (
    DeliveryRow,
    align_concepts,
    build_delivery_rows,
    canonical_gloss,
    canonical_pos,
    load_csv_records,
    normalize_cn,
    write_tsv,
)


def make_csv(root: Path, lang: str, level: str, rows: list[list[str]]) -> None:
    level_dir = root / lang
    level_dir.mkdir(parents=True, exist_ok=True)
    path = level_dir / f"{level}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{lang.capitalize()}_Lemma,English_Lemma,Chinese_Lemma,POS\n")
        for row in rows:
            handle.write(",".join(row) + "\n")


def test_normalize_cn_strips_spaces_and_slashes() -> None:
    assert normalize_cn("上 / 上面") == "上/上面"
    assert normalize_cn("从……起") == "从……起"
    assert normalize_cn("") == ""


def test_load_csv_records_reads_all_columns(tmp_path: Path) -> None:
    make_csv(
        tmp_path,
        "german",
        "A1",
        [["Haus", "house", "房子", "NOUN"], ["gehen", "to go", "去", "VERB"]],
    )
    rows = load_csv_records(tmp_path)
    assert rows == [
        ("de", "A1", "Haus", "house", "房子", "NOUN"),
        ("de", "A1", "gehen", "to go", "去", "VERB"),
    ]


def test_align_concepts_links_same_english_gloss_across_languages() -> None:
    records = [
        ("de", "A1", "Haus", "house", "房子", "NOUN"),
        ("es", "A1", "casa", "house", "房子", "NOUN"),
        ("de", "A1", "gehen", "to go", "去", "VERB"),
    ]
    concepts = align_concepts(records)
    ids = {r: c for c, rows in concepts.items() for r in rows}
    assert ids[records[0]] == ids[records[1]]
    assert ids[records[0]] != ids[records[2]]


def test_align_concepts_links_via_chinese_when_english_is_blank() -> None:
    records = [
        ("de", "A1", "Haus", "house", "房子", "NOUN"),
        ("zh", "A1", "房子", "", "房子", "NOUN"),
    ]
    concepts = align_concepts(records)
    ids = {r: c for c, rows in concepts.items() for r in rows}
    assert ids[records[0]] == ids[records[1]]


def test_canonical_gloss_prefers_multiword_and_majority() -> None:
    assert canonical_gloss(["to be called", "to be called", "called"]) == "to be called"
    assert canonical_gloss(["house", "house", "house"]) == "house"
    assert canonical_gloss(["", "", ""]) == ""


def test_canonical_pos_returns_majority() -> None:
    assert canonical_pos(["NOUN", "NOUN", "VERB"]) == "NOUN"


def test_build_delivery_rows_dedups_and_ranks_and_keys(tmp_path: Path) -> None:
    make_csv(
        tmp_path,
        "german",
        "A1",
        [["Haus", "house", "房子", "NOUN"]],
    )
    make_csv(
        tmp_path,
        "german",
        "A2",
        [["Haus", "house", "房子", "NOUN"]],  # duplicate lemma+gloss across levels
    )
    make_csv(
        tmp_path,
        "spanish",
        "A1",
        [["casa", "house", "房子", "NOUN"], ["morada", "house", "房子", "NOUN"]],
    )
    rows = build_delivery_rows(load_csv_records(tmp_path))
    de = sorted(rows["de"], key=lambda r: r.cefr)
    es = sorted(rows["es"], key=lambda r: r.lemma)
    assert len(de) == 1  # duplicate merged, lowest level kept
    assert de[0].cefr == "A1"
    assert de[0].rank == 1
    assert es[0].rank == 1
    assert es[1].rank == 2
    assert es[0].concept_key == es[1].concept_key == de[0].concept_key
    assert de[0].concept_key  # non-empty, stable


def test_write_tsv_emits_contract_header(tmp_path: Path) -> None:
    rows = [
        DeliveryRow(
            lemma="Haus",
            pos="NOUN",
            gloss="house",
            english_pos="NOUN",
            cefr="A1",
            rank=1,
            concept_key="house-NOUN",
        )
    ]
    out = tmp_path / "delivery"
    write_tsv(out, "de", rows)
    lines = (out / "de.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "lemma\tpos\tenglish_gloss\tenglish_pos\tcefr\trank\tconcept_key"
    assert lines[1] == "Haus\tNOUN\thouse\tNOUN\tA1\t1\thouse-NOUN"


def test_load_expansion_records_reads_expansion_csvs(tmp_path: Path) -> None:
    exp_dir = tmp_path / "german"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_file = exp_dir / "expansion.csv"
    exp_file.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS,CEFR\n"
        "Hund,dog,狗,NOUN,B2\n"
        '"Katze",cat,猫,NOUN,\n'
        ",,,,\n",
        encoding="utf-8",
    )
    from build_contract_delivery import load_expansion_records

    records = load_expansion_records(tmp_path)
    assert ("de", "B2", "Hund", "dog", "狗", "NOUN") in records
    assert ("de", "B1", "Katze", "cat", "猫", "NOUN") in records


def test_build_all_and_main(tmp_path: Path, capsys) -> None:
    make_csv(
        tmp_path,
        "german",
        "A1",
        [["Haus", "house", "房子", "NOUN"]],
    )
    from build_contract_delivery import build_all, main

    out_dir = tmp_path / "output"
    counts = build_all(out_dir, root=tmp_path)
    assert counts.get("de") == 1
    assert (out_dir / "de.tsv").exists()

    code = main([str(out_dir / "sub")])
    assert code == 0
    captured = capsys.readouterr()
    assert "de:" in captured.out


def test_clean_gloss_handles_accents_and_punctuation() -> None:
    from build_contract_delivery import _unquote, clean_gloss

    assert clean_gloss("fiancé (male)") == "fiance male"
    assert clean_gloss("apple/pear") == "apple pear"
    assert clean_gloss("") == ""
    assert _unquote('""word""') == '"word"'
    assert _unquote('"simple"') == "simple"
