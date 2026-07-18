from __future__ import annotations

import unicodedata

import pytest
from pydantic import ValidationError

from scripts.gemma_qa.schemas import (
    CefrReviewBatch,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)


def valid_row() -> dict[str, str]:
    return {
        "id": "german:A1:1",
        "lemma": "Abend",
        "english_lemma": "evening",
        "chinese_lemma": "晚上",
        "upos": "NOUN",
        "action": "keep",
    }


def test_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CefrReviewRow.model_validate({**valid_row(), "reason": "unrequested"})


def test_schema_rejects_bad_upos() -> None:
    with pytest.raises(ValidationError):
        CefrReviewRow.model_validate({**valid_row(), "upos": "NN"})


@pytest.mark.parametrize("field", ["id", "lemma", "english_lemma", "chinese_lemma"])
def test_schema_rejects_newlines_and_tabs(field: str) -> None:
    with pytest.raises(ValidationError):
        CefrReviewRow.model_validate({**valid_row(), field: "bad\nvalue"})
    with pytest.raises(ValidationError):
        CefrReviewRow.model_validate({**valid_row(), field: "bad\tvalue"})


def test_schema_coerces_non_nfc_text_to_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Café")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    row = CefrReviewRow.model_validate({**valid_row(), "lemma": decomposed})
    assert row.lemma == unicodedata.normalize("NFC", "Café")


def test_review_batch_uses_exact_enums() -> None:
    batch = CefrReviewBatch.model_validate({"rows": [valid_row()]})
    assert batch.rows[0].upos is UPOS.NOUN
    assert batch.rows[0].action is ReviewAction.KEEP
