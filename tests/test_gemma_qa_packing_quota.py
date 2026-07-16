from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gemma_qa.config import (
    HARD_REQUESTS_PER_DAY,
    HARD_REQUESTS_PER_MINUTE,
    HARD_TOKENS_PER_MINUTE,
    MODEL_CEILINGS,
    ModelCeilings,
)
from scripts.gemma_qa.packing import pack_records
from scripts.gemma_qa.quota import QuotaGate


class FixedEstimator:
    def count(self, text: str) -> int:
        return len(text)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


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


def test_quota_waits_for_rolling_request_and_token_limits(tmp_path: Path) -> None:
    clock = FakeClock()
    gate = QuotaGate(
        tmp_path / "quota.sqlite3",
        rpm=2,
        tpm=100,
        rpd=10,
        clock=clock,
        sleeper=clock.sleep,
    )
    gate.reserve("gemma-4-31b-it", prompt_tokens=20, max_output_tokens=20)
    gate.reserve("gemma-4-31b-it", prompt_tokens=20, max_output_tokens=20)
    gate.reserve("gemma-4-31b-it", prompt_tokens=20, max_output_tokens=20)
    assert clock.sleeps == [60.0]
    assert gate.status("gemma-4-31b-it").requests_last_minute == 1
    gate.close()


def test_quota_reconciles_actual_usage(tmp_path: Path) -> None:
    clock = FakeClock()
    gate = QuotaGate(
        tmp_path / "quota.sqlite3",
        rpm=30,
        tpm=100,
        rpd=10,
        clock=clock,
        sleeper=clock.sleep,
    )
    reservation = gate.reserve("gemma-4-31b-it", prompt_tokens=20, max_output_tokens=50)
    gate.reconcile(reservation, actual_input_tokens=25)
    assert gate.status("gemma-4-31b-it").tokens_last_minute == 25
    gate.close()


def test_quota_tpm_ignores_output_token_budget(tmp_path: Path) -> None:
    clock = FakeClock()
    gate = QuotaGate(
        tmp_path / "quota.sqlite3",
        rpm=30,
        tpm=100,
        rpd=10,
        clock=clock,
        sleeper=clock.sleep,
    )
    # Output budget alone would exceed TPM if double-counted; input-only must pass.
    gate.reserve("gemma-4-31b-it", prompt_tokens=40, max_output_tokens=10_000)
    assert clock.sleeps == []
    assert gate.status("gemma-4-31b-it").tokens_last_minute == 40
    with pytest.raises(ValueError, match="input token ceiling"):
        gate.reserve("gemma-4-31b-it", prompt_tokens=101, max_output_tokens=0)
    gate.close()


def test_configured_model_ceilings_never_exceed_hard_caps() -> None:
    for ceilings in MODEL_CEILINGS.values():
        assert ceilings.requests_per_minute <= ceilings.hard_requests_per_minute
        assert ceilings.tokens_per_minute <= ceilings.hard_tokens_per_minute
        assert ceilings.requests_per_day <= ceilings.hard_requests_per_day


@pytest.mark.parametrize(
    "ceilings",
    [
        {
            "requests_per_minute": HARD_REQUESTS_PER_MINUTE + 1,
            "tokens_per_minute": HARD_TOKENS_PER_MINUTE,
            "requests_per_day": HARD_REQUESTS_PER_DAY,
        },
        {
            "requests_per_minute": HARD_REQUESTS_PER_MINUTE,
            "tokens_per_minute": HARD_TOKENS_PER_MINUTE + 1,
            "requests_per_day": HARD_REQUESTS_PER_DAY,
        },
        {
            "requests_per_minute": HARD_REQUESTS_PER_MINUTE,
            "tokens_per_minute": HARD_TOKENS_PER_MINUTE,
            "requests_per_day": HARD_REQUESTS_PER_DAY + 1,
        },
    ],
)
def test_model_ceilings_reject_values_above_hard_caps(
    ceilings: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="hard caps"):
        ModelCeilings(**ceilings)
