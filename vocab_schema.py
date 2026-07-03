"""Shared CEFR vocabulary schema metadata."""

from __future__ import annotations

from typing import TypedDict


class LanguageSchema(TypedDict):
    lemma_col: str
    trans_cols: tuple[str, str]


LEVELS = ("A1", "A2", "B1", "B2", "C1")
TARGETS = {"A1": 600, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000}

# HSK levels for Chinese (1=beginner .. 6=advanced). Stored alongside
# CEFR levels in the same enum; the Vidiom UI labels them per-language.
HSK_LEVELS = ("HSK1", "HSK2", "HSK3", "HSK4", "HSK5", "HSK6")
HSK_TARGETS = {
    "HSK1": 150,
    "HSK2": 150,   # 300 cumulative, 150 new
    "HSK3": 300,   # 600 cumulative, 300 new
    "HSK4": 600,   # 1200 cumulative, 600 new
    "HSK5": 1300,  # 2500 cumulative, 1300 new
    "HSK6": 2500,  # 5000 cumulative, 2500 new
}

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
    "chinese": {
        "lemma_col": "Chinese_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
}
