"""Shared CEFR vocabulary schema metadata."""

from __future__ import annotations

from typing import TypedDict


class LanguageSchema(TypedDict):
    lemma_col: str
    trans_cols: tuple[str, str]


LEVELS = ("A1", "A2", "B1", "B2", "C1")
TARGETS = {"A1": 600, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000}

LANGS: dict[str, LanguageSchema] = {
    "english": {
        "lemma_col": "English_Lemma",
        "trans_cols": ("German_Translation", "Spanish_Translation"),
    },
    "german": {
        "lemma_col": "German_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
    "spanish": {
        "lemma_col": "Spanish_Lemma",
        "trans_cols": ("English_Translation", "German_Translation"),
    },
    "arabic": {
        "lemma_col": "Arabic_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
    "french": {
        "lemma_col": "French_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
    "swedish": {
        "lemma_col": "Swedish_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
}
