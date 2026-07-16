from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Protocol, TypeVar

import tiktoken
from pydantic import BaseModel

from .config import INPUT_BATCH_TOKEN_CAP

T = TypeVar("T")


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenEstimator:
    def __init__(self) -> None:
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _jsonable(record: object) -> object:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    return record


def estimate_batch_tokens(
    records: Sequence[object],
    *,
    prompt_overhead: str,
    estimator: TokenEstimator,
) -> int:
    payload = json.dumps(
        [_jsonable(record) for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return estimator.count(prompt_overhead) + estimator.count(payload)


def pack_records(
    records: Sequence[T],
    *,
    prompt_overhead: str,
    cap: int = INPUT_BATCH_TOKEN_CAP,
    max_records: int | None = None,
    estimator: TokenEstimator | None = None,
) -> list[list[T]]:
    if cap <= 0:
        raise ValueError("cap must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    token_estimator = estimator or TiktokenEstimator()
    batches: list[list[T]] = []
    current: list[T] = []
    for record in records:
        candidate = [*current, record]
        if (
            max_records is None or len(candidate) <= max_records
        ) and estimate_batch_tokens(
            candidate,
            prompt_overhead=prompt_overhead,
            estimator=token_estimator,
        ) <= cap:
            current = candidate
            continue
        if not current:
            raise ValueError("single record exceeds input token cap")
        batches.append(current)
        current = [record]
        if (
            estimate_batch_tokens(
                current,
                prompt_overhead=prompt_overhead,
                estimator=token_estimator,
            )
            > cap
        ):
            raise ValueError("single record exceeds input token cap")
    if current:
        batches.append(current)
    return batches
