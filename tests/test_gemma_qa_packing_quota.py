from __future__ import annotations

import math

import pytest
import tiktoken

from scripts.gemma_qa.packing import TiktokenEstimator, pack_records
from scripts.gemma_qa.quota import QuotaGate


class FixedEstimator:
    def count(self, text: str) -> int:
        return len(text)


def test_packer_never_splits_records_or_exceeds_cap() -> None:
    records = [{"id": str(index), "lemma": "x" * 20} for index in range(8)]
    batches = pack_records(
        records,
        prompt_overhead="prompt",
        cap=100,
        estimator=FixedEstimator(),
    )
    assert [record["id"] for batch in batches for record in batch] == [
        str(index) for index in range(8)
    ]
    assert all(
        len("prompt") + len(__import__("json").dumps(batch, ensure_ascii=False)) <= 100
        for batch in batches
    )


def test_packer_honors_max_records_without_losing_order() -> None:
    records = [{"id": str(index)} for index in range(8)]
    batches = pack_records(
        records,
        prompt_overhead="prompt",
        cap=10_000,
        max_records=3,
        estimator=FixedEstimator(),
    )
    assert [len(batch) for batch in batches] == [3, 3, 2]
    assert [record["id"] for batch in batches for record in batch] == [
        str(index) for index in range(8)
    ]


def test_quota_gate_is_noop() -> None:
    gate = QuotaGate(":memory:")
    reservation = gate.reserve("any", prompt_tokens=10**9, max_output_tokens=10**9)
    gate.reconcile(reservation, actual_input_tokens=1)
    assert gate.remaining_daily_requests("any") > 0
    assert gate.status("any").requests_last_minute == 0
    gate.close()


def test_tiktoken_estimator_falls_back_when_encoding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_name: str) -> None:
        raise RuntimeError("no tokenizer on this box")

    monkeypatch.setattr(tiktoken, "get_encoding", boom)
    estimator = TiktokenEstimator()
    assert estimator.count("abc") == max(1, math.ceil(3 / 4))


def test_tiktoken_estimator_falls_back_when_encode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenEncoding:
        def encode(self, _text: str) -> int:
            raise ValueError("tokenizer exploded")

    monkeypatch.setattr(tiktoken, "get_encoding", lambda _n: BrokenEncoding())
    estimator = TiktokenEstimator()
    # Byte heuristic: at least one token, roughly len/4.
    assert estimator.count("abcd" * 10) == 10
