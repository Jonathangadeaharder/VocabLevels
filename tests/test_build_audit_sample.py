"""Characterization tests for build_audit_sample classification logic."""

from __future__ import annotations

from pathlib import Path

from scripts.gemma_qa.build_audit_sample import classify_rows, load_level_csv


def write_level_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Lemma,English_Lemma,Chinese_Lemma,POS"]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_level_csv_parses_keys_and_fp(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    write_level_csv(
        path,
        [["Haus", "house", "房子", "NOUN"], ["laufen", "to run", "跑", "VERB"]],
    )
    rows = load_level_csv(path)
    assert [row["csv_line"] for row in rows] == ["2", "3"]
    assert rows[0]["fp"] == "haus\0noun"
    assert rows[0]["lemma_key"] == "haus"
    assert rows[0]["english_lemma"] == "house"


def test_load_level_csv_empty_file_yields_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    path.write_text("Lemma,English_Lemma,Chinese_Lemma,POS\n", encoding="utf-8")
    assert load_level_csv(path) == []


def _row(lemma: str, english: str, chinese: str, pos: str) -> dict[str, str]:
    return {
        "csv_line": "2",
        "lemma": lemma,
        "english_lemma": english,
        "chinese_lemma": chinese,
        "upos": pos,
        "fp": f"{lemma.casefold()}\0{pos.casefold()}",
        "lemma_key": lemma.casefold(),
    }


def test_classify_rows_without_committed_marks_all_unchanged() -> None:
    proposed = [_row("Haus", "house", "房子", "NOUN")]
    classified = classify_rows(proposed, None)
    assert len(classified) == 1
    assert classified[0].change == "unchanged"
    assert classified[0].committed_before == ""


def test_classify_rows_same_fp_same_glosses_is_unchanged() -> None:
    committed = [_row("Haus", "house", "房子", "NOUN")]
    classified = classify_rows([_row("Haus", "house", "房子", "NOUN")], committed)
    assert classified[0].change == "unchanged"
    assert classified[0].committed_before == "Haus|house|房子|NOUN"


def test_classify_rows_same_fp_changed_gloss_is_edited() -> None:
    committed = [_row("Haus", "house", "房子", "NOUN")]
    classified = classify_rows([_row("Haus", "building", "房子", "NOUN")], committed)
    assert classified[0].change == "edited"
    assert classified[0].committed_before == "Haus|house|房子|NOUN"


def test_classify_rows_lemma_match_pos_changed_is_pos_or_key_changed() -> None:
    committed = [_row("Haus", "house", "房子", "NOUN")]
    proposed = _row("Haus", "house", "房子", "VERB")
    proposed["fp"] = "haus\0verb"
    classified = classify_rows([proposed], committed)
    assert classified[0].change == "pos_or_key_changed"
    assert classified[0].committed_before == "Haus|house|房子|NOUN"


def test_classify_rows_unknown_lemma_is_new_or_renamed() -> None:
    committed = [_row("Haus", "house", "房子", "NOUN")]
    classified = classify_rows([_row("Baum", "tree", "树", "NOUN")], committed)
    assert classified[0].change == "new_or_renamed"
    assert classified[0].committed_before == ""


def test_classify_rows_first_committed_match_wins() -> None:
    first = _row("Haus", "house", "房子", "NOUN")
    second = _row("Haus", "building", "房子", "NOUN")
    classified = classify_rows([_row("Haus", "house", "房子", "NOUN")], [first, second])
    assert classified[0].change == "unchanged"
    assert classified[0].committed_before == "Haus|house|房子|NOUN"
