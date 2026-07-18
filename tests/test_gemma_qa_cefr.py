from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.gemma_qa.cefr import (
    read_cefr_csv,
    validate_review_batch,
    write_reviewed_csv,
)
from scripts.gemma_qa.schemas import (
    CefrReviewBatch,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def review_for(row_id: str, lemma: str) -> CefrReviewRow:
    return CefrReviewRow(
        id=row_id,
        lemma=lemma,
        english_lemma=lemma,
        chinese_lemma="词",
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )


def test_duplicate_physical_header_is_read_positionally(tmp_path: Path) -> None:
    source = tmp_path / "A1.csv"
    write_csv(
        source,
        ["English_Lemma", "English_Lemma", "Chinese_Lemma", "POS"],
        [["colour", "color", "颜色", "NOUN"]],
    )
    document = read_cefr_csv(source, lang="english", level="A1")
    assert document.rows[0].lemma == "colour"
    assert document.rows[0].english_lemma == "color"
    assert document.rows[0].id == "english:A1:1"


def test_validation_requires_exact_id_order_and_cardinality(tmp_path: Path) -> None:
    source = tmp_path / "A1.csv"
    write_csv(
        source,
        ["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"],
        [["Abend", "evening", "", "NOUN"], ["Haus", "house", "", "NOUN"]],
    )
    document = read_cefr_csv(source, lang="german", level="A1")
    reversed_batch = CefrReviewBatch(
        rows=[
            review_for(document.rows[1].id, "Haus"),
            review_for(document.rows[0].id, "Abend"),
        ]
    )
    with pytest.raises(ValueError, match="IDs"):
        validate_review_batch(document.rows, reversed_batch)


def test_dry_run_writes_proposed_only(tmp_path: Path) -> None:
    source = tmp_path / "A1.csv"
    write_csv(
        source,
        ["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"],
        [["Abend", "evening", "", "NOUN"]],
    )
    original = source.read_bytes()
    document = read_cefr_csv(source, lang="german", level="A1")
    reviews = CefrReviewBatch(rows=[review_for(document.rows[0].id, "Abend")])
    output = write_reviewed_csv(document, reviews, apply=False)
    assert output == tmp_path / "A1.proposed.csv"
    assert output.exists()
    assert b"\r\n" not in output.read_bytes()
    assert source.read_bytes() == original


def test_is_retriable_batch_error_covers_hang_paths() -> None:
    import httpx
    from scripts.gemma_qa.cefr import _is_retriable_batch_error

    assert _is_retriable_batch_error(TimeoutError("dual generate wait ceiling"))
    assert _is_retriable_batch_error(
        httpx.ReadTimeout(
            "request wall clock exceeded after 180s",
            request=httpx.Request("POST", "https://example.test"),
        )
    )
    assert _is_retriable_batch_error(ValueError("422 Unprocessable Entity"))
    assert not _is_retriable_batch_error(ValueError("schema mismatch"))


def test_dual_wait_ceiling_tracks_wall_clock(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA_QA_REQUEST_WALL_S", "100")
    from scripts.gemma_qa.cefr import _dual_wait_ceiling_s

    assert _dual_wait_ceiling_s() == 260.0
