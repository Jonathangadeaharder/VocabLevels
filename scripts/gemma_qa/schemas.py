from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
TokenId = Annotated[str, StringConstraints(min_length=1, pattern=r"^\d+(?:[-.]\d+)?$")]


class UPOS(StrEnum):
    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"


class ReviewAction(StrEnum):
    KEEP = "keep"
    FIX = "fix"
    DROP = "drop"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _validate_text(value: str) -> str:
    if not value:
        raise ValueError("must not be empty")
    if "\n" in value or "\r" in value or "\t" in value:
        raise ValueError("must not contain newlines or tabs")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("must use NFC normalization")
    return value


class CefrInputRow(StrictModel):
    id: NonEmptyText
    lemma: NonEmptyText
    english_lemma: NonEmptyText
    chinese_lemma: NonEmptyText | None
    upos: UPOS

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    @field_validator("id", "lemma", "english_lemma")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("chinese_lemma")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _validate_text(value)


class CefrReviewRow(StrictModel):
    id: NonEmptyText
    lemma: NonEmptyText
    english_lemma: NonEmptyText
    chinese_lemma: NonEmptyText
    upos: UPOS
    action: ReviewAction

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value: object) -> object:
        return ReviewAction(value) if isinstance(value, str) else value

    @field_validator("id", "lemma", "english_lemma", "chinese_lemma")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class CefrInputBatch(StrictModel):
    batch_id: NonEmptyText
    rows: list[CefrInputRow] = Field(min_length=1)

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, value: str) -> str:
        return _validate_text(value)


class CefrReviewBatch(StrictModel):
    rows: list[CefrReviewRow] = Field(min_length=1)


class CefrLanguageIssue(StrictModel):
    code: NonEmptyText
    message: NonEmptyText

    @field_validator("code", "message")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class CefrLanguageRepairItem(StrictModel):
    row: CefrReviewRow
    issues: list[CefrLanguageIssue] = Field(min_length=1)


class CefrRefillConcept(StrictModel):
    id: NonEmptyText
    english_lemma: NonEmptyText
    chinese_lemma: NonEmptyText | None
    upos: UPOS

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    @field_validator("id", "english_lemma")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("chinese_lemma")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _validate_text(value)


class CefrRefillRow(StrictModel):
    id: NonEmptyText
    lemma: NonEmptyText
    chinese_lemma: NonEmptyText
    action: ReviewAction

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value: object) -> object:
        return ReviewAction(value) if isinstance(value, str) else value

    @field_validator("id", "lemma", "chinese_lemma")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class CefrRefillBatch(StrictModel):
    rows: list[CefrRefillRow] = Field(min_length=1)


class CefrNovelRow(CefrReviewRow):
    pass


class CefrNovelBatch(StrictModel):
    rows: list[CefrNovelRow] = Field(min_length=1)


class ConlluToken(StrictModel):
    id: TokenId
    form: NonEmptyText
    lemma: NonEmptyText
    upos: UPOS
    xpos: NonEmptyText = "_"
    feats: NonEmptyText = "_"
    head: NonEmptyText = "_"
    deprel: NonEmptyText = "_"
    deps: NonEmptyText = "_"
    misc: NonEmptyText = "_"
    action: ReviewAction = ReviewAction.KEEP

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value: object) -> object:
        return ReviewAction(value) if isinstance(value, str) else value

    @field_validator(
        "id",
        "form",
        "lemma",
        "xpos",
        "feats",
        "head",
        "deprel",
        "deps",
        "misc",
    )
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class ConlluSentence(StrictModel):
    id: NonEmptyText
    text: NonEmptyText
    tokens: list[ConlluToken] = Field(min_length=1)

    @field_validator("id", "text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("tokens")
    @classmethod
    def unique_token_ids(cls, value: list[ConlluToken]) -> list[ConlluToken]:
        ids = [token.id for token in value]
        if len(ids) != len(set(ids)):
            raise ValueError("token IDs must be unique")
        return value


class ConlluBatch(StrictModel):
    batch_id: NonEmptyText
    sentences: list[ConlluSentence] = Field(min_length=1)

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("sentences")
    @classmethod
    def unique_sentence_ids(cls, value: list[ConlluSentence]) -> list[ConlluSentence]:
        ids = [sentence.id for sentence in value]
        if len(ids) != len(set(ids)):
            raise ValueError("sentence IDs must be unique")
        return value

    def ensure_ids(self, expected: list[str]) -> Self:
        actual = [sentence.id for sentence in self.sentences]
        if actual != expected:
            raise ValueError("sentence IDs/order/cardinality differ from input")
        return self


class HandcraftToken(StrictModel):
    id: TokenId
    form: NonEmptyText
    lemma: NonEmptyText
    upos: UPOS

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    @field_validator("id", "form", "lemma")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)


class HandcraftSentence(StrictModel):
    sent_id: NonEmptyText
    text: NonEmptyText
    target_ids: list[NonEmptyText] = Field(min_length=1)
    tokens: list[HandcraftToken] = Field(min_length=1)

    @field_validator("sent_id", "text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text(value)

    @field_validator("target_ids")
    @classmethod
    def validate_target_ids(cls, value: list[str]) -> list[str]:
        for target_id in value:
            _validate_text(target_id)
        if len(value) != len(set(value)):
            raise ValueError("target IDs must be unique")
        return value

    @field_validator("tokens")
    @classmethod
    def unique_token_ids(
        cls,
        value: list[HandcraftToken],
    ) -> list[HandcraftToken]:
        ids = [token.id for token in value]
        if len(ids) != len(set(ids)):
            raise ValueError("token IDs must be unique")
        return value


class HandcraftBatch(StrictModel):
    sentences: list[HandcraftSentence] = Field(min_length=1)

    @field_validator("sentences")
    @classmethod
    def unique_sentence_ids(
        cls,
        value: list[HandcraftSentence],
    ) -> list[HandcraftSentence]:
        ids = [sentence.sent_id for sentence in value]
        if len(ids) != len(set(ids)):
            raise ValueError("sentence IDs must be unique")
        return value


CEFRRow = CefrReviewRow
CEFRBatch = CefrReviewBatch
Action = ReviewAction
