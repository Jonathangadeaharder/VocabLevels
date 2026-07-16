from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .client import Usage


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    prompt_hash: str
    model: str
    batch_id: str
    request_json: dict[str, object]
    response_json: dict[str, object]
    usage: Usage


@dataclass(frozen=True)
class LedgerStatus:
    checkpoints: int
    prompt_tokens: int
    candidate_tokens: int
    total_tokens: int


class Ledger:
    def __init__(self, database: Path | str) -> None:
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                candidate_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (prompt_hash, model, batch_id)
            )
            """
        )
        self._connection.commit()
        self._lock = threading.RLock()

    def get(self, digest: str, model: str, batch_id: str) -> Checkpoint | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT request_json, response_json,
                       prompt_tokens, candidate_tokens, total_tokens
                FROM checkpoints
                WHERE prompt_hash = ? AND model = ? AND batch_id = ?
                """,
                (digest, model, batch_id),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            prompt_hash=digest,
            model=model,
            batch_id=batch_id,
            request_json=json.loads(row[0]),
            response_json=json.loads(row[1]),
            usage=Usage(
                prompt_tokens=int(row[2]),
                candidate_tokens=int(row[3]),
                total_tokens=int(row[4]),
            ),
        )

    def store(self, checkpoint: Checkpoint) -> None:
        request_json = json.dumps(
            checkpoint.request_json,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        response_json = json.dumps(
            checkpoint.response_json,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO checkpoints (
                    prompt_hash, model, batch_id, request_json, response_json,
                    prompt_tokens, candidate_tokens, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prompt_hash, model, batch_id) DO NOTHING
                """,
                (
                    checkpoint.prompt_hash,
                    checkpoint.model,
                    checkpoint.batch_id,
                    request_json,
                    response_json,
                    checkpoint.usage.prompt_tokens,
                    checkpoint.usage.candidate_tokens,
                    checkpoint.usage.total_tokens,
                    time.time(),
                ),
            )

    def delete(self, digest: str, model: str, batch_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM checkpoints
                WHERE prompt_hash = ? AND model = ? AND batch_id = ?
                """,
                (digest, model, batch_id),
            )
        return cursor.rowcount == 1

    def status(self) -> LedgerStatus:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0),
                       COALESCE(SUM(candidate_tokens), 0),
                       COALESCE(SUM(total_tokens), 0)
                FROM checkpoints
                """
            ).fetchone()
        if row is None:
            return LedgerStatus(0, 0, 0, 0)
        return LedgerStatus(*(int(value) for value in row))

    def close(self) -> None:
        with self._lock:
            self._connection.close()
