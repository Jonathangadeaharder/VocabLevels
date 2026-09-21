"""Characterization tests for generate_dutch_vocab translate_only."""

from __future__ import annotations

import csv
from pathlib import Path

import generate_dutch_vocab as gdv
from generate_dutch_vocab import translate_only


def write_dutch_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(["Dutch_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        writer.writerows(rows)


def read_dutch_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[1:]


def test_translate_only_fills_blank_and_echo_cells(tmp_path: Path, monkeypatch) -> None:
    for level in gdv.LEVELS:
        write_dutch_csv(
            tmp_path / "dutch" / f"{level}.csv",
            [
                ["huis", "house", "房子", "NOUN"],
                ["boom", "", "树", "NOUN"],
                ["boom", "boom", "树", "NOUN"],
            ],
        )
    monkeypatch.setattr(gdv, "ROOT", tmp_path)
    monkeypatch.setattr(gdv, "translate_words", lambda words: {"boom": "tree"})
    assert translate_only() == 0
    for level in gdv.LEVELS:
        rows = read_dutch_csv(tmp_path / "dutch" / f"{level}.csv")
        assert rows[0] == ["huis", "house", "房子", "NOUN"]
        assert rows[1] == ["boom", "tree", "树", "NOUN"]
        assert rows[2] == ["boom", "tree", "树", "NOUN"]


def test_translate_only_all_present_is_noop(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    for level in gdv.LEVELS:
        write_dutch_csv(
            tmp_path / "dutch" / f"{level}.csv",
            [["huis", "house", "房子", "NOUN"]],
        )
    monkeypatch.setattr(gdv, "ROOT", tmp_path)

    def fail_translate(words):
        raise AssertionError("must not translate")

    monkeypatch.setattr(gdv, "translate_words", fail_translate)
    assert translate_only() == 0
    assert "All translations already present." in capsys.readouterr().out
    for level in gdv.LEVELS:
        assert read_dutch_csv(tmp_path / "dutch" / f"{level}.csv") == [
            ["huis", "house", "房子", "NOUN"]
        ]


def test_translate_only_missing_translation_falls_back_to_lemma(
    tmp_path: Path, monkeypatch
) -> None:
    for level in gdv.LEVELS:
        write_dutch_csv(
            tmp_path / "dutch" / f"{level}.csv",
            [["boom", "", "树", "NOUN"]],
        )
    monkeypatch.setattr(gdv, "ROOT", tmp_path)
    monkeypatch.setattr(gdv, "translate_words", lambda words: {})
    assert translate_only() == 0
    for level in gdv.LEVELS:
        assert read_dutch_csv(tmp_path / "dutch" / f"{level}.csv") == [
            ["boom", "boom", "树", "NOUN"]
        ]
