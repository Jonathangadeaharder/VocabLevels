from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import (
    MODEL_CEILINGS,
    ModelCeilings,
    REQUESTS_PER_DAY,
    REQUESTS_PER_MINUTE,
    TOKENS_PER_MINUTE,
    ceilings_for,
)


class DailyQuotaExceeded(RuntimeError):
    """Raised when a model's daily request ceiling would be exceeded."""


@dataclass(frozen=True)
class QuotaStatus:
    requests_last_minute: int
    tokens_last_minute: int
    requests_last_day: int


class QuotaGate:
    def __init__(
        self,
        database: Path | str,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        # Test override: when any of rpm/tpm/rpd is set, apply that ceiling to all
        # models (preserves existing unit-test behavior).
        overrides = (rpm, tpm, rpd)
        if any(value is not None for value in overrides) and any(
            value is None for value in overrides
        ):
            raise ValueError("rpm, tpm, and rpd overrides must be provided together")
        if overrides[0] is not None:
            assert rpm is not None and tpm is not None and rpd is not None
            if min(rpm, tpm, rpd) <= 0:
                raise ValueError("quota ceilings must be positive")
            self._override = ModelCeilings(
                requests_per_minute=rpm,
                tokens_per_minute=tpm,
                requests_per_day=rpd,
                hard_requests_per_minute=max(rpm, REQUESTS_PER_MINUTE),
                hard_tokens_per_minute=max(tpm, TOKENS_PER_MINUTE),
                hard_requests_per_day=max(rpd, REQUESTS_PER_DAY),
                rpd_policy="wait",
            )
        else:
            self._override = None
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            database,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS quota_reservations (
                reservation_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created_at REAL NOT NULL,
                token_count INTEGER NOT NULL CHECK (token_count >= 0)
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS quota_model_created
            ON quota_reservations(model, created_at)
            """
        )
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.RLock()

    def ceilings(self, model: str) -> ModelCeilings:
        if self._override is not None:
            return self._override
        return ceilings_for(model)

    def remaining_daily_requests(self, model: str) -> int:
        ceilings = self.ceilings(model)
        with self._lock:
            now = self._clock()
            day_rows = self._rows_since(model, now - 86_400)
        return max(0, ceilings.requests_per_day - len(day_rows))

    def reserve(
        self,
        model: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> str:
        # TPM bucket is input-only; max_output_tokens is accepted for call-site
        # compatibility but does not consume the per-minute input ceiling.
        if prompt_tokens < 0 or max_output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        ceilings = self.ceilings(model)
        if prompt_tokens > ceilings.tokens_per_minute:
            raise ValueError("single request exceeds per-minute input token ceiling")
        while True:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    now = self._clock()
                    minute_rows = self._rows_since(model, now - 60)
                    day_rows = self._rows_since(model, now - 86_400)
                    if (
                        ceilings.rpd_policy == "fail"
                        and len(day_rows) >= ceilings.requests_per_day
                    ):
                        self._connection.execute("ROLLBACK")
                        raise DailyQuotaExceeded(
                            f"{model}: daily request ceiling "
                            f"{ceilings.requests_per_day} exhausted"
                        )
                    wait = self._wait_required(
                        now,
                        minute_rows,
                        day_rows,
                        prompt_tokens,
                        ceilings,
                    )
                    if wait <= 0:
                        reservation_id = uuid.uuid4().hex
                        self._connection.execute(
                            """
                            INSERT INTO quota_reservations
                                (reservation_id, model, created_at, token_count)
                            VALUES (?, ?, ?, ?)
                            """,
                            (reservation_id, model, now, prompt_tokens),
                        )
                        self._connection.execute("COMMIT")
                        return reservation_id
                    self._connection.execute("ROLLBACK")
                except DailyQuotaExceeded:
                    raise
                except BaseException:
                    if self._connection.in_transaction:
                        self._connection.execute("ROLLBACK")
                    raise
            self._sleeper(wait)

    def reconcile(self, reservation_id: str, *, actual_input_tokens: int) -> None:
        if actual_input_tokens < 0:
            raise ValueError("actual token count must be non-negative")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE quota_reservations
                SET token_count = ?
                WHERE reservation_id = ?
                """,
                (actual_input_tokens, reservation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown reservation: {reservation_id}")

    def status(self, model: str) -> QuotaStatus:
        with self._lock:
            now = self._clock()
            minute_rows = self._rows_since(model, now - 60)
            day_rows = self._rows_since(model, now - 86_400)
        return QuotaStatus(
            requests_last_minute=len(minute_rows),
            tokens_last_minute=sum(tokens for _, tokens in minute_rows),
            requests_last_day=len(day_rows),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _rows_since(self, model: str, threshold: float) -> list[tuple[float, int]]:
        rows = self._connection.execute(
            """
            SELECT created_at, token_count
            FROM quota_reservations
            WHERE model = ? AND created_at > ?
            ORDER BY created_at
            """,
            (model, threshold),
        ).fetchall()
        return [(float(created_at), int(tokens)) for created_at, tokens in rows]

    def _wait_required(
        self,
        now: float,
        minute_rows: list[tuple[float, int]],
        day_rows: list[tuple[float, int]],
        reserved_tokens: int,
        ceilings: ModelCeilings,
    ) -> float:
        waits = [0.0]
        if len(minute_rows) >= ceilings.requests_per_minute:
            waits.append(max(0.0, minute_rows[0][0] + 60 - now))
        minute_tokens = sum(tokens for _, tokens in minute_rows)
        if minute_tokens + reserved_tokens > ceilings.tokens_per_minute:
            remaining = minute_tokens
            for created_at, tokens in minute_rows:
                remaining -= tokens
                if remaining + reserved_tokens <= ceilings.tokens_per_minute:
                    waits.append(max(0.0, created_at + 60 - now))
                    break
        if (
            ceilings.rpd_policy == "wait"
            and len(day_rows) >= ceilings.requests_per_day
        ):
            waits.append(max(0.0, day_rows[0][0] + 86_400 - now))
        return max(waits)


# Re-export for callers that imported MODEL_CEILINGS via quota historically.
__all__ = [
    "DailyQuotaExceeded",
    "QuotaGate",
    "QuotaStatus",
    "MODEL_CEILINGS",
]
