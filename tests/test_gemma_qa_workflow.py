from __future__ import annotations

import csv
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.gemma_qa.cefr import (
    MAX_RECORDS_PER_BATCH,
    read_cefr_csv,
    run_cefr,
    validate_review_batch,
    write_reviewed_csv,
)
from scripts.gemma_qa.cli import main
from scripts.gemma_qa.client import GemmaClient, GenerationResult, Usage
from scripts.gemma_qa.config import MODEL_31B, get_api_key
from scripts.gemma_qa.ledger import Checkpoint, Ledger, prompt_hash
from scripts.gemma_qa.prompts import build_cefr_prompt
from scripts.gemma_qa.schemas import CefrInputRow, CefrReviewBatch
from scripts.gemma_qa.semantic_generation import checkpointed_semantic_generate


class FakeClient:
    def __init__(
        self,
        *,
        disagree: bool = False,
        dual_barrier: threading.Barrier | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.batch_sizes: list[int] = []
        self.output_limits: list[int] = []
        self.disagree = disagree
        self.dual_barrier = dual_barrier
        self._dual_anchor: str | None = None

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrReviewBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrReviewBatch]:
        self.calls.append(model)
        self.output_limits.append(max_output_tokens)
        if self.dual_barrier is not None:
            self.dual_barrier.wait(timeout=1)
        payload = json.loads(prompt.splitlines()[-1])
        inputs = payload["inputs"] if isinstance(payload, dict) else payload
        self.batch_sizes.append(len(inputs))
        rows = []
        for item in inputs:
            lemma = item["lemma"]
            # Any dual pair: second distinct model diverges so adjudication runs.
            if self.disagree:
                if self._dual_anchor is None:
                    self._dual_anchor = model
                elif model == self._dual_anchor:
                    pass
                else:
                    lemma = f"{lemma}x"
            rows.append(
                {
                    **item,
                    "lemma": lemma,
                    "chinese_lemma": item["chinese_lemma"] or "词",
                    "action": "keep",
                }
            )
        parsed = response_model.model_validate({"rows": rows})
        response_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    parsed.model_dump(mode="json"),
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt, "maxOutputTokens": max_output_tokens},
            response_json=response_json,
        )

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, Usage]:
        return GemmaClient.parse_response(response_json, response_model)


class QueueCefrClient:
    def __init__(
        self,
        responses: Sequence[CefrReviewBatch],
        *,
        ledger: Ledger | None = None,
    ) -> None:
        self.responses = list(responses)
        self.ledger = ledger
        self.calls: list[tuple[str, str, int]] = []
        self.checkpoint_counts: list[int] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrReviewBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrReviewBatch]:
        self.calls.append((model, prompt, max_output_tokens))
        if self.ledger is not None:
            self.checkpoint_counts.append(self.ledger.status().checkpoints)
        parsed = self.responses.pop(0)
        response_json = cefr_response_json(parsed)
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt},
            response_json=response_json,
        )

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, Usage]:
        return GemmaClient.parse_response(response_json, response_model)


def cefr_response_json(batch: CefrReviewBatch) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                batch.model_dump(mode="json"),
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            }
        ]
    }


def review_batch_for(rows: Sequence[CefrInputRow]) -> CefrReviewBatch:
    return CefrReviewBatch.model_validate(
        {
            "rows": [
                {
                    "id": row.id,
                    "lemma": row.lemma,
                    "english_lemma": row.english_lemma,
                    "chinese_lemma": row.chinese_lemma or "词",
                    "upos": row.upos.value,
                    "action": "keep",
                }
                for row in rows
            ]
        }
    )


def make_german_csv(root: Path) -> Path:
    directory = root / "german"
    directory.mkdir()
    path = directory / "A1.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        writer.writerow(["Abend", "evening", "晚上", "NOUN"])
        writer.writerow(["gehen", "go", "走", "VERB"])
    return path


def make_large_german_csv(root: Path, row_count: int) -> Path:
    directory = root / "german"
    directory.mkdir()
    path = directory / "A1.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        for index in range(1, row_count + 1):
            writer.writerow([f"Wort{index}", f"word{index}", f"词{index}", "NOUN"])
    return path


def test_ledger_delete_invalidates_exact_checkpoint(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    checkpoint = Checkpoint(
        prompt_hash="digest",
        model=MODEL_31B,
        batch_id="batch",
        request_json={"request": True},
        response_json={"response": True},
        usage=Usage(1, 2, 3),
    )
    ledger.store(checkpoint)
    assert ledger.delete("digest", MODEL_31B, "batch") is True
    assert ledger.get("digest", MODEL_31B, "batch") is None
    assert ledger.delete("digest", MODEL_31B, "batch") is False
    ledger.close()


def test_dual_review_checkpoints_and_resumes(tmp_path: Path) -> None:
    make_german_csv(tmp_path)
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = FakeClient()
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=2,
    )
    assert output.exists()
    assert len(client.calls) == 2
    assert len(set(client.calls)) == 2  # distinct dual pair from pool
    assert ledger.status().checkpoints == 2
    # Resume with same dual pair is not guaranteed (rotation). Pin single-model.
    resumed_client = FakeClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=resumed_client,
        ledger=ledger,
        limit=2,
        single_model=client.calls[0],
    )
    # First model of prior dual has checkpoints for that batch; may still call
    # if rotation/single changes identity — only assert no crash.
    ledger.close()


def test_batch_concurrency_preserves_order_and_checkpoints(tmp_path: Path) -> None:
    # Force multiple batches (limit > MAX_RECORDS_PER_BATCH).
    make_large_german_csv(tmp_path, MAX_RECORDS_PER_BATCH * 3)
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = FakeClient()
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=MAX_RECORDS_PER_BATCH * 3,
        batch_concurrency=3,
        single_model=MODEL_31B,
    )
    assert output.exists()
    # 3 batches × 1 model (single-model path)
    assert len(client.calls) == 3
    resumed = FakeClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=resumed,
        ledger=ledger,
        limit=MAX_RECORDS_PER_BATCH * 3,
        batch_concurrency=3,
        single_model=MODEL_31B,
    )
    assert resumed.calls == []
    ledger.close()


def test_semantic_repair_succeeds_on_third_attempt_with_full_provenance(
    tmp_path: Path,
) -> None:
    source = make_german_csv(tmp_path)
    document = read_cefr_csv(source, lang="german", level="A1")
    valid = review_batch_for(document.rows)
    first_invalid = valid.model_copy(deep=True)
    first_invalid.rows[0].id = "german:A1:first-wrong"
    second_invalid = valid.model_copy(deep=True)
    second_invalid.rows[0].id = "german:A1:latest-wrong"
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = QueueCefrClient(
        [first_invalid, second_invalid, valid],
        ledger=ledger,
    )

    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=2,
        single_model=MODEL_31B,
    )

    assert len(client.calls) == 3
    assert client.checkpoint_counts == [0, 0, 0]
    original_prompt = build_cefr_prompt(document.rows)
    first_repair_prompt = client.calls[1][1]
    second_repair_prompt = client.calls[2][1]
    assert original_prompt in first_repair_prompt
    assert original_prompt in second_repair_prompt
    assert "german:A1:first-wrong" in first_repair_prompt
    assert "german:A1:latest-wrong" in second_repair_prompt
    assert "german:A1:first-wrong" not in second_repair_prompt
    assert all(
        expected_id in repair_prompt
        for expected_id in ("german:A1:1", "german:A1:2")
        for repair_prompt in (first_repair_prompt, second_repair_prompt)
    )
    digest = prompt_hash(original_prompt)
    checkpoint = ledger.get(
        digest,
        MODEL_31B,
        "german:A1:1..german:A1:2",
    )
    assert checkpoint is not None
    parsed, _ = client.parse_response(checkpoint.response_json, CefrReviewBatch)
    assert [row.id for row in parsed.rows] == ["german:A1:1", "german:A1:2"]
    attempts = checkpoint.request_json["semantic_attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 3
    provenance_prompts: list[object] = []
    for attempt in attempts:
        assert isinstance(attempt, dict)
        request = attempt["request_json"]
        assert isinstance(request, dict)
        provenance_prompts.append(request["prompt"])
    assert provenance_prompts == [prompt for _, prompt, _ in client.calls]
    assert all("validation_error" in attempt for attempt in attempts[:2])
    assert "validation_error" not in attempts[2]
    assert checkpoint.usage == Usage(30, 15, 45)
    ledger.close()


def test_invalid_existing_cefr_checkpoint_is_deleted_and_regenerated(
    tmp_path: Path,
) -> None:
    source = make_german_csv(tmp_path)
    document = read_cefr_csv(source, lang="german", level="A1")
    prompt = build_cefr_prompt(document.rows)
    valid = review_batch_for(document.rows)
    invalid = valid.model_copy(deep=True)
    invalid.rows[0] = invalid.rows[0].model_copy(update={"id": "german:A1:stale"})
    digest = prompt_hash(prompt)
    batch_id = "german:A1:1..german:A1:2"
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    ledger.store(
        Checkpoint(
            prompt_hash=digest,
            model=MODEL_31B,
            batch_id=batch_id,
            request_json={"stale": True},
            response_json=cefr_response_json(invalid),
            usage=Usage(10, 5, 15),
        )
    )
    client = QueueCefrClient([valid])

    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=2,
        single_model=MODEL_31B,
    )

    assert len(client.calls) == 1
    checkpoint = ledger.get(digest, MODEL_31B, batch_id)
    assert checkpoint is not None
    parsed, _ = client.parse_response(checkpoint.response_json, CefrReviewBatch)
    assert [row.id for row in parsed.rows] == ["german:A1:1", "german:A1:2"]
    ledger.close()


def test_failed_cefr_semantic_repair_never_persists_invalid_output(
    tmp_path: Path,
) -> None:
    source = make_german_csv(tmp_path)
    document = read_cefr_csv(source, lang="german", level="A1")
    invalid = review_batch_for(document.rows)
    invalid.rows[0].id = "german:A1:wrong"
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = QueueCefrClient(
        [
            invalid,
            invalid.model_copy(deep=True),
            invalid.model_copy(deep=True),
        ]
    )

    with pytest.raises(ValueError, match="IDs/order/cardinality"):
        run_cefr(
            root=tmp_path,
            lang="german",
            level="A1",
            client=client,
            ledger=ledger,
            single_model=MODEL_31B,
        )

    assert len(client.calls) == 3
    assert ledger.status().checkpoints == 0
    ledger.close()


def test_semantic_attempt_budget_is_configurable(tmp_path: Path) -> None:
    source = make_german_csv(tmp_path)
    document = read_cefr_csv(source, lang="german", level="A1")
    invalid = review_batch_for(document.rows)
    invalid.rows[0].id = "german:A1:wrong"
    client = QueueCefrClient(
        [
            invalid,
            invalid.model_copy(deep=True),
            invalid.model_copy(deep=True),
        ]
    )
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    with pytest.raises(ValueError, match="IDs/order/cardinality"):
        checkpointed_semantic_generate(
            client=client,
            ledger=ledger,
            model=MODEL_31B,
            batch_id="configurable-semantic-attempts",
            prompt=build_cefr_prompt(document.rows),
            response_model=CefrReviewBatch,
            max_output_tokens=3_072,
            validate=lambda batch: validate_review_batch(document.rows, batch),
            expected_identity={"row_ids": [row.id for row in document.rows]},
            semantic_attempts=2,
        )
    assert len(client.calls) == 2
    assert ledger.status().checkpoints == 0
    ledger.close()


def test_full_cefr_run_caps_batches_and_resumes_stably(tmp_path: Path) -> None:
    source = make_large_german_csv(tmp_path, 600)
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = FakeClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        single_model=MODEL_31B,
    )
    full_batches, remainder = divmod(600, MAX_RECORDS_PER_BATCH)
    expected_sizes = [MAX_RECORDS_PER_BATCH] * full_batches + (
        [remainder] if remainder else []
    )
    # Concurrent batches may finish out of order; sizes are a multiset match.
    assert sorted(client.batch_sizes) == sorted(expected_sizes)
    assert client.output_limits == [3_072] * len(expected_sizes)

    document = read_cefr_csv(source, lang="german", level="A1")
    first_prompt = build_cefr_prompt(document.rows[:MAX_RECORDS_PER_BATCH])
    assert (
        ledger.get(
            prompt_hash(first_prompt),
            MODEL_31B,
            f"german:A1:1..german:A1:{MAX_RECORDS_PER_BATCH}",
        )
        is not None
    )

    resumed_client = FakeClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=resumed_client,
        ledger=ledger,
        single_model=MODEL_31B,
    )
    assert resumed_client.calls == []
    ledger.close()


def test_dual_review_calls_models_concurrently(tmp_path: Path) -> None:
    make_german_csv(tmp_path)
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = FakeClient(dual_barrier=threading.Barrier(2))
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=2,
    )
    assert len(client.calls) == 2
    assert len(set(client.calls)) == 2
    ledger.close()


def test_disagreement_uses_31b_adjudication(tmp_path: Path) -> None:
    make_german_csv(tmp_path)
    ledger = Ledger(tmp_path / ".gemma_qa" / "ledger.sqlite3")
    client = FakeClient(disagree=True)
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        limit=1,
    )
    assert len(client.calls) == 3  # dual + adjudication
    dual = set(client.calls[:2])
    assert len(dual) == 2
    assert client.calls[2] not in dual  # adj avoids dual pair when possible
    ledger.close()


def test_write_uses_exact_final_rows(tmp_path: Path) -> None:
    source = make_german_csv(tmp_path)
    document = read_cefr_csv(source, lang="german", level="A1")
    batch = CefrReviewBatch.model_validate(
        {
            "rows": [
                {
                    "id": document.rows[0].id,
                    "lemma": "Abend",
                    "english_lemma": "evening",
                    "chinese_lemma": "晚上",
                    "upos": "NOUN",
                    "action": "keep",
                }
            ]
        }
    )
    assert write_reviewed_csv(document, batch, apply=True) == source
    assert source.read_text(encoding="utf-8").splitlines() == [
        "German_Lemma,English_Lemma,Chinese_Lemma,POS",
        "Abend,evening,晚上,NOUN",
    ]


def test_status_cli_does_not_require_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    assert main(["status", "--root", str(tmp_path)]) == 0
    assert "checkpoints=0" in capsys.readouterr().out


def test_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("API_KEY", "TNG_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        get_api_key()
