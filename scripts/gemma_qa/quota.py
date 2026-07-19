"""No-op quota gate.

TNG (chat.model.tngtech.com) load-balances and does not expose Gemini-style
RPM/TPM/RPD limits. Call sites may still construct ``QuotaGate`` for API
compatibility; reserve/reconcile never sleep or fail on quota.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import time


class DailyQuotaExceeded(RuntimeError):
    """Retained for import compatibility; never raised by NoOpQuotaGate."""


@dataclass(frozen=True)
class QuotaStatus:
    requests_last_minute: int = 0
    tokens_last_minute: int = 0
    requests_last_day: int = 0


class QuotaGate:
    """No-op stand-in for the retired client-side quota gate."""

    def __init__(
        self,
        database: Path | str | None = None,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        _ = database
        _ = rpm
        _ = tpm
        _ = rpd
        _ = clock
        _ = sleeper

    def remaining_daily_requests(self, model: str) -> int:
        _ = model
        return 10**9

    def reserve(
        self,
        model: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> str:
        _ = model
        _ = prompt_tokens
        _ = max_output_tokens
        return uuid.uuid4().hex

    def reconcile(self, reservation_id: str, *, actual_input_tokens: int) -> None:
        _ = reservation_id
        _ = actual_input_tokens

    def status(self, model: str) -> QuotaStatus:
        _ = model
        return QuotaStatus()

    def close(self) -> None:
        return None


__all__ = [
    "DailyQuotaExceeded",
    "QuotaGate",
    "QuotaStatus",
]
