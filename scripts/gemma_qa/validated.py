from __future__ import annotations

import csv
import sqlite3
import unicodedata
from collections.abc import Sequence
from pathlib import Path


def validated_store_path(root: Path) -> Path:
    return root / ".gemma_qa" / "validated.sqlite3"


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def fingerprint(
    lang: str,
    level: str,
    lemma: str,
    english: str,
    chinese: str,
    upos: str,
) -> str:
    parts = (
        _nfc(lang),
        _nfc(level),
        _nfc(lemma),
        _nfc(english),
        _nfc(chinese),
        _nfc(upos),
    )
    return "\x1f".join(parts)


class ValidatedStore:
    def __init__(self, path: Path | str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS validated (
                fingerprint TEXT PRIMARY KEY,
                lang TEXT NOT NULL,
                level TEXT NOT NULL,
                lemma TEXT NOT NULL,
                english_lemma TEXT NOT NULL,
                chinese_lemma TEXT NOT NULL,
                upos TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS validated_lang_level "
            "ON validated (lang, level)"
        )
        self._connection.commit()

    def contains(
        self,
        lang: str,
        level: str,
        lemma: str,
        english: str,
        chinese: str,
        upos: str,
    ) -> bool:
        digest = fingerprint(lang, level, lemma, english, chinese, upos)
        row = self._connection.execute(
            "SELECT 1 FROM validated WHERE fingerprint = ?",
            (digest,),
        ).fetchone()
        return row is not None

    def add_many(
        self,
        lang: str,
        level: str,
        rows: Sequence[tuple[str, str, str, str]],
    ) -> None:
        if not rows:
            return
        self._connection.executemany(
            """
            INSERT OR IGNORE INTO validated (
                fingerprint, lang, level, lemma,
                english_lemma, chinese_lemma, upos
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    fingerprint(lang, level, lemma, english, chinese, upos),
                    _nfc(lang),
                    _nfc(level),
                    _nfc(lemma),
                    _nfc(english),
                    _nfc(chinese),
                    _nfc(upos),
                )
                for lemma, english, chinese, upos in rows
            ],
        )
        self._connection.commit()

    def mark_rows(
        self,
        lang: str,
        level: str,
        rows: Sequence[tuple[str, str, str, str]],
    ) -> None:
        normalized_lang = _nfc(lang)
        normalized_level = _nfc(level)
        self._connection.execute(
            "DELETE FROM validated WHERE lang = ? AND level = ?",
            (normalized_lang, normalized_level),
        )
        self.add_many(lang, level, rows)

    def count(self, lang: str, level: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM validated WHERE lang = ? AND level = ?",
            (_nfc(lang), _nfc(level)),
        ).fetchone()
        assert row is not None
        return int(row[0])

    def seed_from_csv(self, root: Path, *, lang: str, level: str) -> None:
        path = root / lang / f"{level}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            physical_rows = list(csv.reader(handle))
        if not physical_rows:
            raise ValueError(f"empty CSV: {path}")
        rows: list[tuple[str, str, str, str]] = []
        for row_number, row in enumerate(physical_rows[1:], start=2):
            if len(row) != 4:
                raise ValueError(f"{path}:{row_number}: expected 4 columns")
            rows.append((row[0], row[1], row[2], row[3]))
        self.mark_rows(lang, level, rows)

    def close(self) -> None:
        self._connection.close()
