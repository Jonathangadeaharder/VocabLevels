"""Tests for check_data_contract, the acceptance gate of the data contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from check_data_contract import (
    TSV_HEADER,
    check_ascii,
    check_duplicates,
    check_rank_gaps,
    check_script_and_substance,
    check_shrunken_glosses,
    load_csv_rows,
    load_delivery_rows,
    main,
    normalize_gloss,
    pair_coverage,
    run_baseline,
    run_delivery,
    Row,
)


def test_normalize_gloss_matches_the_app_backfill() -> None:
    assert normalize_gloss("to be called") == "be called"
    assert normalize_gloss("  House  ") == "house"
    assert normalize_gloss("tomorrow") == "tomorrow"
    assert normalize_gloss(" to be called ") == "to be called"
    # btrim strips spaces only; tabs survive on both sides.
    assert normalize_gloss("\thouse\t") == "\thouse\t"


def make_row(
    lang: str,
    lemma: str,
    gloss: str,
    pos: str = "VERB",
    rank: int | None = None,
) -> Row:
    return Row(lang, lemma, pos, gloss, rank)


def test_check_shrunken_glosses_flags_head_reduction() -> None:
    source = [make_row("de", "heißen", "to be called")]
    delivery = [make_row("de", "heißen", "be")]
    assert check_shrunken_glosses(delivery, source) == [
        "de:heißen: ['to be called'] -> 'be'"
    ]


def test_check_shrunken_glosses_accepts_full_gloss_and_new_lemmas() -> None:
    source = [make_row("de", "heißen", "to be called")]
    delivery = [
        make_row("de", "heißen", "to be called"),
        make_row("de", "neu", "new"),
    ]
    assert check_shrunken_glosses(delivery, source) == []


def test_check_duplicates() -> None:
    rows = [
        make_row("de", "Bank", "bank"),
        make_row("de", "Bank", "Bank"),
        make_row("de", "Bank", "river bank"),
    ]
    assert check_duplicates(rows) == ["de:Bank:bank"]


def test_check_rank_gaps_requires_exactly_one_rank_one() -> None:
    rows = [
        make_row("es", "llamar", "to call", rank=1),
        make_row("es", "telefonear", "to call", rank=2),
        make_row("es", "ser", "to be", rank=1),
        make_row("es", "estar", "to be", rank=1),
        make_row("es", "ir", "to go"),
    ]
    gaps = check_rank_gaps(rows)
    assert "es:be:VERB" in gaps
    assert "es:go:VERB" in gaps
    assert "es:call:VERB" not in gaps


def test_pair_coverage_counts_distinct_target_lemmas() -> None:
    rows = [
        make_row("de", "heißen", "to be called"),
        make_row("de", "Haus", "house", pos="NOUN"),
        make_row("de", "allein", "alone"),
        make_row("es", "llamarse", "to be called"),
        make_row("es", "casa", "house", pos="NOUN"),
        make_row("es", "solo", "alone"),
        make_row("es", "solo_mismo", "alone"),
    ]
    ein, mehr, ohne = pair_coverage(rows, "de", "es")
    # heißen->llamarse and Haus->casa eindeutig, allein has two es rows.
    assert (ein, mehr, ohne) == (2, 1, 0)


def test_check_ascii_rejects_foreign_and_blank_glosses() -> None:
    # Criterion 5 is a charset check: pure-ASCII foreign words pass it;
    # catching them is the data author's job (rule 6), not the charset's.
    rows = [
        make_row("de", "a", "llamárse"),
        make_row("de", "b", ""),
        make_row("de", "c", "to be called"),
        make_row("de", "d", "house (home)"),
    ]
    assert check_ascii(rows) == [
        "de:a: 'llamárse'",
        "de:b: ''",
        "de:d: 'house (home)'",
    ]


def write_delivery(directory: Path, name: str, lines: list[list[str]]) -> None:
    path = directory / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(TSV_HEADER) + "\n")
        for line in lines:
            handle.write("\t".join(line) + "\n")


def make_source_csv(root: Path, lang_dir: str, rows: list[list[str]]) -> None:
    level_dir = root / lang_dir
    level_dir.mkdir(parents=True, exist_ok=True)
    path = level_dir / "A1.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("X_Lemma,English_Lemma,Chinese_Lemma,POS\n")
        for row in rows:
            handle.write(",".join(row) + "\n")


def test_load_csv_rows_reads_lemma_gloss_and_pos(tmp_path: Path) -> None:
    make_source_csv(tmp_path, "german", [["heißen", "to be called", "叫", "VERB"]])
    rows = load_csv_rows(tmp_path)
    assert rows == [Row("de", "heißen", "VERB", "to be called", None)]


def test_load_delivery_rows_rejects_an_unknown_header(tmp_path: Path) -> None:
    (tmp_path / "de.tsv").write_text("lemma\twrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected header"):
        load_delivery_rows(tmp_path)


def test_load_delivery_rows_rejects_short_rows(tmp_path: Path) -> None:
    (tmp_path / "de.tsv").write_text(
        "\t".join(TSV_HEADER) + "\nBank\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="columns"):
        load_delivery_rows(tmp_path)


def test_run_delivery_passes_on_a_clean_pair(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    make_source_csv(source_root, "german", [["Haus", "house", "房", "NOUN"]])
    make_source_csv(source_root, "spanish", [["casa", "house", "房", "NOUN"]])
    write_delivery(
        delivery,
        "de.tsv",
        [["Haus", "NOUN", "house", "NOUN", "A1", "1", ""]],
    )
    write_delivery(
        delivery,
        "es.tsv",
        [["casa", "NOUN", "house", "NOUN", "A1", "1", ""]],
    )
    assert run_delivery(delivery, source_root) == 0


def test_run_delivery_fails_on_coverage_and_foreign_gloss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_root = tmp_path / "source"
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    make_source_csv(source_root, "german", [["einsam", "lonely", "孤", "ADJ"]])
    write_delivery(
        delivery,
        "de.tsv",
        [["einsam", "ADJ", "lonely", "ADJ", "B2", "", ""]],
    )
    write_delivery(
        delivery, "es.tsv", [["solo", "ADJ", "solo_es", "ADJ", "B2", "1", ""]]
    )
    assert run_delivery(delivery, source_root) == 1
    out = capsys.readouterr().out
    assert "RESULT: FAIL" in out
    assert "[LOW] de->es" in out
    assert "criterion 3: 1 group(s)" in out


def test_run_baseline_reports_and_returns_zero(tmp_path: Path) -> None:
    make_source_csv(tmp_path, "german", [["Haus", "house", "房", "NOUN"]])
    make_source_csv(tmp_path, "spanish", [["casa", "house", "房", "NOUN"]])
    assert run_baseline(tmp_path) == 0


def test_main_dispatches_between_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_source_csv(tmp_path, "german", [["Haus", "house", "房", "NOUN"]])
    monkeypatch.setattr("check_data_contract.ROOT", tmp_path)
    assert main([]) == 0
    assert main([str(tmp_path / "missing")]) == 1


def test_load_csv_rows_with_expansion(tmp_path: Path) -> None:
    make_source_csv(tmp_path, "german", [["Haus", "house", "房", "NOUN"]])
    exp_dir = tmp_path / "german"
    (exp_dir / "expansion.csv").write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS,CEFR\n"
        '"Katze","cat","猫","NOUN","B1"\n'
        ",,,,\n",
        encoding="utf-8",
    )
    rows = load_csv_rows(tmp_path)
    assert len(rows) == 2
    assert rows[1].lemma == "Katze"


def test_check_script_and_substance_flags_wrong_scripts_and_junk() -> None:
    valid_rows = [
        Row("ar", "كتاب", "NOUN", "book", 1),
        Row("zh", "书", "NOUN", "book", 1),
        Row("de", "Buch", "NOUN", "book", 1),
        Row("en", "book", "NOUN", "book", 1),
        Row("es", "casa de campo", "NOUN", "country house", 1),
    ]
    assert check_script_and_substance(valid_rows) == []

    invalid_rows = [
        Row("ar", "book", "NOUN", "book", 1),  # Latin in Arabic
        Row("zh", "book", "NOUN", "book", 1),  # Latin in Chinese
        Row("zh", "复合词", "NOUN", "word", 1),  # Junk placeholder
        Row("zh", "词", "NOUN", "word", 1),  # Junk placeholder
        Row("de", "", "NOUN", "book", 1),  # Empty lemma
        Row("de", "his departure", "NOUN", "his departure", 1),  # English copy
        Row(
            "nl", "to defrost", "VERB", "to defrost", 1
        ),  # English function prefix copy
        Row("sv", "barge", "NOUN", "barge", 1),  # Single-word ungrounded copy
    ]
    violations = check_script_and_substance(invalid_rows)
    assert len(violations) == 8
    assert any("ar:non_arabic_script:'book'" in v for v in violations)
    assert any("zh:non_chinese_script:'book'" in v for v in violations)
    assert any("zh:junk_lemma:'复合词'" in v for v in violations)
    assert any("zh:junk_lemma:'词'" in v for v in violations)
    assert any("de:empty_lemma:'book'" in v for v in violations)
    assert any("de:ungrounded_gloss_copy:'his departure'" in v for v in violations)
    assert any(
        "nl:english_function_prefix_copy:'to defrost'" in v
        or "nl:ungrounded_gloss_copy:'to defrost'" in v
        for v in violations
    )
    assert any("sv:ungrounded_gloss_copy:'barge'" in v for v in violations)
