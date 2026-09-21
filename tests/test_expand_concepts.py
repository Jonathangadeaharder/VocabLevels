"""Characterization tests for expand_concepts blank-gloss and language passes."""

from __future__ import annotations

import csv
from pathlib import Path

import httpx

import scripts.expand_concepts as expand_concepts
from scripts.expand_concepts import (
    BATCH_SIZE,
    Checkpoint,
    Concept,
    _apply_blank_fills,
    _collect_blank_rows,
    _process_language,
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["Lang_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        writer.writerows(rows)


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[1:]


def test_collect_blank_rows_skips_header_and_non_blanks(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    write_csv(
        path,
        [
            ["Haus", "house", "房子", "NOUN"],
            ["Baum", "", "树", "NOUN"],
            ["", "no lemma", "", ""],
            ["Blatt", "leaf", "", "NOUN"],
        ],
    )
    blanks = _collect_blank_rows([path], None)
    assert blanks == [("A1", "Baum", 3, "NOUN")]


def test_collect_blank_rows_limit_slices(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    write_csv(path, [[f"w{i}", "", "", "NOUN"] for i in range(5)])
    assert _collect_blank_rows([path], 2) == [
        ("A1", "w0", 2, "NOUN"),
        ("A1", "w1", 3, "NOUN"),
    ]


def test_collect_blank_rows_missing_file_skipped(tmp_path: Path) -> None:
    assert _collect_blank_rows([tmp_path / "A1.csv"], None) == []


def test_apply_blank_fills_fills_only_blank_cells(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    write_csv(
        path,
        [
            ["Haus", "house", "房子", "NOUN"],
            ["Baum", "", "树", "NOUN"],
            ["Blatt", "leaf", "", "NOUN"],
        ],
    )
    _apply_blank_fills([path], [("A1", "Baum", 3, "NOUN", "tree")])
    rows = read_csv(path)
    assert rows[0] == ["Haus", "house", "房子", "NOUN"]
    assert rows[1] == ["Baum", "tree", "树", "NOUN"]
    assert rows[2] == ["Blatt", "leaf", "", "NOUN"]


def test_apply_blank_fills_no_updates_leaves_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "A1.csv"
    write_csv(path, [["Haus", "house", "房子", "NOUN"]])
    before = path.read_bytes()
    _apply_blank_fills([path], [])
    assert path.read_bytes() == before


def _stub_generate(monkeypatch, results: dict[int, tuple[str, str]]):
    def fake_generate_batch(concepts, lang, *, client):
        return results

    monkeypatch.setattr(expand_concepts, "_generate_batch", fake_generate_batch)


def _stub_review(monkeypatch, approved_ids: set[int]):
    def fake_review_batch(items, *, client):
        return [item for index, item in enumerate(items) if index in approved_ids]

    monkeypatch.setattr(expand_concepts, "_review_batch", fake_review_batch)


def test_process_language_approved_from_checkpoint_without_generation(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")
    checkpoint.record_approved("de", Concept("house", "NOUN"), "Haus")
    monkeypatch.setattr(
        expand_concepts,
        "_generate_batch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not generate")),
    )
    results = _process_language(
        "de", [Concept("house", "NOUN")], [], checkpoint, httpx.Client()
    )
    assert [(r.concept.gloss, r.lemma) for r in results] == [("house", "Haus")]


def test_process_language_rejected_concepts_skipped(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")
    checkpoint.record_rejected("de", Concept("house", "NOUN"), "Haus")
    _stub_generate(monkeypatch, {})
    results = _process_language(
        "de", [Concept("house", "NOUN")], [], checkpoint, httpx.Client()
    )
    assert results == []


def test_process_language_generates_reviews_and_records(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")
    _stub_generate(monkeypatch, {0: ("Haus", "房子"), 1: ("Baum", "树")})
    _stub_review(monkeypatch, {0})
    results = _process_language(
        "de",
        [Concept("house", "NOUN"), Concept("tree", "NOUN")],
        [],
        checkpoint,
        httpx.Client(),
    )
    assert [(r.concept.gloss, r.lemma) for r in results] == [("house", "Haus")]
    assert checkpoint.approved[("de", Concept("house", "NOUN"))] == "Haus"
    assert checkpoint.generated[("de", Concept("tree", "NOUN"))] == "Baum"


def test_process_language_generation_failure_skips_batch(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")

    def failing_generate(concepts, lang, *, client):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(expand_concepts, "_generate_batch", failing_generate)
    results = _process_language(
        "de", [Concept("house", "NOUN")], [], checkpoint, httpx.Client()
    )
    assert results == []
    assert checkpoint.approved == {}


def test_process_language_review_failure_falls_back_to_generated(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")
    _stub_generate(monkeypatch, {0: ("Haus", "房子")})

    def failing_review(items, *, client):
        raise ValueError("bad reply")

    monkeypatch.setattr(expand_concepts, "_review_batch", failing_review)
    results = _process_language(
        "de", [Concept("house", "NOUN")], [], checkpoint, httpx.Client()
    )
    assert [(r.concept.gloss, r.lemma) for r in results] == [("house", "Haus")]


def test_process_language_batches_respect_batch_size(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = Checkpoint(tmp_path / "cp.jsonl")
    seen: list[int] = []

    def fake_generate(concepts, lang, *, client):
        seen.append(len(concepts))
        return {i: (f"lemma{i}", "") for i in range(len(concepts))}

    monkeypatch.setattr(expand_concepts, "_generate_batch", fake_generate)
    _stub_review(monkeypatch, set())
    concepts = [Concept(f"gloss{i}", "NOUN") for i in range(BATCH_SIZE + 1)]
    _process_language("de", concepts, [], checkpoint, httpx.Client())
    assert seen == [BATCH_SIZE, 1]
