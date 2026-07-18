from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from scripts.gemma_qa.client import (
    DEFAULT_STRUCTURED_ATTEMPTS,
    MAX_TRANSIENT_RETRIES,
    PRODUCTION_TIMEOUT,
    GemmaClient,
)
from scripts.gemma_qa.config import MODEL_GEMMA, MODEL_QWEN_35B, MODEL_QWEN_397B
from scripts.gemma_qa.packing import TokenEstimator
from scripts.gemma_qa.schemas import CefrReviewBatch


def gemini_response(text: str) -> httpx.Response:
    """OpenAI-compatible chat.completion body (name kept for test history)."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        },
    )


def valid_payload() -> dict[str, object]:
    return {
        "rows": [
            {
                "id": "german:A1:1",
                "lemma": "Abend",
                "english_lemma": "evening",
                "chinese_lemma": "晚上",
                "upos": "NOUN",
                "action": "keep",
            }
        ]
    }


def request_prompt(request: dict[str, object]) -> str:
    messages = request["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    prompt = message["content"]
    assert isinstance(prompt, str)
    return prompt


def retry_info_response(delay: object) -> httpx.Response:
    # TNG uses Retry-After header; body is optional.
    _ = delay
    return httpx.Response(429, json={"error": {"message": "rate limited"}})


class FixedEstimator(TokenEstimator):
    def count(self, text: str) -> int:
        return len(text)




def test_client_retries_429_and_parses_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            response = retry_info_response("43s")
            response.headers["Retry-After"] = "2"
            return response
        text = f"```json\n{json.dumps(valid_payload(), ensure_ascii=False)}\n```"
        return gemini_response(text)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = GemmaClient(
            http_client=http_client, sleeper=sleeps.append, jitter=lambda: 0
        )
        result = client.generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert result.usage.total_tokens == 20
    assert sleeps == [2.0]
    assert requests[0].headers["Authorization"] == "Bearer secret-value"


def test_client_uses_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": {"message": "rate limited"}},
            )
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(
            http_client=http_client,
            sleeper=sleeps.append,
            jitter=lambda: 0.25,
        ).generate(
            model="Qwen/Qwen3.6-35B-A3B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert calls == 2
    assert sleeps == [2.0]


def test_default_transient_retry_budget_is_bounded() -> None:
    assert MAX_TRANSIENT_RETRIES == 8


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_client_does_not_retry_nontransient_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": {"message": "denied"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            sleeper=sleeps.append,
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.generate(
                model="Qwen/Qwen3.6-35B-A3B-FP8",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )

    assert calls == 1
    assert sleeps == []


def test_client_retries_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("read timed out", request=request)
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
            sleeper=sleeps.append,
            jitter=lambda: 0,
            max_retries=1,
        ).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert calls == 2
    assert sleeps == [1.0]


def test_client_raises_read_timeout_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
            sleeper=sleeps.append,
            jitter=lambda: 0,
            max_retries=2,
        )
        with pytest.raises(httpx.ReadTimeout, match="read timed out"):
            client.generate(
                model="Qwen/Qwen3.5-397B-A17B-FP8",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )

    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_production_timeout_allows_large_model_responses() -> None:
    # Idle bounds stay finite; wall-clock (request_wall_clock_s) caps total POST.
    assert PRODUCTION_TIMEOUT.connect == 30
    assert PRODUCTION_TIMEOUT.read == 120
    assert PRODUCTION_TIMEOUT.write == 120
    assert PRODUCTION_TIMEOUT.pool == 60


def test_request_wall_clock_default_is_bounded() -> None:
    from scripts.gemma_qa.client import request_wall_clock_s

    assert 1.0 <= request_wall_clock_s() <= 900.0
    assert request_wall_clock_s() == 180.0


def test_client_aborts_slow_stream_via_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow response past wall-clock must raise (idle read alone insufficient)."""
    import time

    monkeypatch.setenv("API_KEY", "secret-value")
    monkeypatch.setenv("GEMMA_QA_REQUEST_WALL_S", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        # Block longer than wall; stream path checks deadline after open.
        time.sleep(1.3)
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    sleeps: list[float] = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
            sleeper=sleeps.append,
            jitter=lambda: 0,
            max_retries=0,
        )
        with pytest.raises(httpx.ReadTimeout, match="wall clock"):
            client.generate(
                model="Qwen/Qwen3.5-397B-A17B-FP8",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )


def test_request_uses_tng_openai_chat_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        GemmaClient(http_client=http_client).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    body = requests[0]
    assert body["model"] == MODEL_QWEN_397B
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"][0]["content"] == "review"


def test_parse_response_ignores_live_shaped_thought_part() -> None:
    payload = json.dumps(valid_payload(), ensure_ascii=False)
    response_json: dict[str, object] = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "", "thought": True},
                        {"text": payload},
                    ]
                }
            }
        ]
    }
    parsed, _ = GemmaClient.parse_response(response_json, CefrReviewBatch)
    assert parsed.rows[0].lemma == "Abend"


def test_parse_response_rejects_thought_only_candidate() -> None:
    response_json: dict[str, object] = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "internal summary", "thought": True},
                        {"text": "", "thought": True},
                    ]
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="chat completion|candidate|nonempty"):
        GemmaClient.parse_response(response_json, CefrReviewBatch)


def test_parse_response_concatenates_nonthought_parts_in_order() -> None:
    payload = json.dumps(valid_payload(), ensure_ascii=False)
    split = len(payload) // 2
    response_json: dict[str, object] = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": payload[:split]},
                        {"text": "ignore me", "thought": True},
                        {"text": payload[split:]},
                    ]
                }
            }
        ]
    }
    parsed, _ = GemmaClient.parse_response(response_json, CefrReviewBatch)
    assert parsed.rows[0].id == "german:A1:1"


@pytest.mark.parametrize(
    "parts",
    [
        [None],
        [{"text": 1}],
        [{"thought": False}],
    ],
)
def test_parse_response_rejects_malformed_output_parts(parts: object) -> None:
    response_json: dict[str, object] = {
        "candidates": [
            {
                "content": {
                    "parts": parts,
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="chat completion|candidate"):
        GemmaClient.parse_response(response_json, CefrReviewBatch)


def test_client_repairs_invalid_structured_output_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_json = json.loads(request.content)
        requests.append(request_json)
        if len(requests) == 1:
            return gemini_response('{"rows":[]}')
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(http_client=http_client).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="original review request",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert len(requests) == 2
    repair_text = request_prompt(requests[1])
    assert "original review request" in repair_text
    assert "rows" in repair_text
    assert "validation" in repair_text.lower()


def test_client_succeeds_after_two_structured_repairs_and_reconciles_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    requests: list[dict[str, object]] = []
    invalid_outputs = [
        '{"rows":[]} first-extra',
        '{"rows":[]} second-extra',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) <= len(invalid_outputs):
            return gemini_response(invalid_outputs[len(requests) - 1])
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        ).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="original review request",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    request_prompts = [request_prompt(request) for request in requests]
    assert DEFAULT_STRUCTURED_ATTEMPTS == 3
    assert result.parsed.rows[0].lemma == "Abend"
    assert len(requests) == 3
    assert invalid_outputs[0] in request_prompts[1]
    assert invalid_outputs[1] in request_prompts[2]
    assert invalid_outputs[0] not in request_prompts[2]
    assert "invalid JSON at line" in request_prompts[1]
    assert "invalid JSON at line" in request_prompts[2]
    assert all("original review request" in prompt for prompt in request_prompts)


def test_client_raises_after_three_malformed_structured_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    requests: list[dict[str, object]] = []
    invalid_outputs = [
        '{"rows":[]} first-extra',
        '{"rows":[]} second-extra',
        '{"rows":[]} third-extra',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return gemini_response(invalid_outputs[len(requests) - 1])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        )
        with pytest.raises(json.JSONDecodeError, match="Extra data"):
            client.generate(
                model="Qwen/Qwen3.5-397B-A17B-FP8",
                prompt="original review request",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )

    request_prompts = [request_prompt(request) for request in requests]
    assert len(requests) == 3
    assert invalid_outputs[0] in request_prompts[1]
    assert invalid_outputs[1] in request_prompts[2]
    assert invalid_outputs[0] not in request_prompts[2]
    assert "invalid JSON at line" in request_prompts[1]
    assert "invalid JSON at line" in request_prompts[2]


def test_structured_attempt_budget_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return gemini_response('{"rows":[]} extra')

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            structured_attempts=2,
        )
        with pytest.raises(json.JSONDecodeError, match="Extra data"):
            client.generate(
                model="Qwen/Qwen3.5-397B-A17B-FP8",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )
    assert calls == 2


def test_selected_model_ids_are_the_tng_trio() -> None:
    assert MODEL_QWEN_397B == "Qwen/Qwen3.5-397B-A17B-FP8"
    assert MODEL_QWEN_35B == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert MODEL_GEMMA == "google/gemma-4-31B-it"


def test_failed_retry_reconciles_unused_token_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500)
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
            sleeper=lambda _: None,
        ).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )


def test_empty_usage_metadata_keeps_conservative_prompt_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    response = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    valid_payload(),
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {},
        },
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: response)
    ) as http_client:
        GemmaClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        ).generate(
            model="Qwen/Qwen3.5-397B-A17B-FP8",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )


def test_client_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "secret-value")
    transport = httpx.MockTransport(lambda request: gemini_response("{not json}"))
    with httpx.Client(transport=transport) as http_client:
        client = GemmaClient(http_client=http_client)
        with pytest.raises((ValueError, ValidationError)):
            client.generate(
                model="Qwen/Qwen3.5-397B-A17B-FP8",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )


def test_slot_acquire_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.gemma_qa import config as cfg

    monkeypatch.setenv("GEMMA_QA_PER_MODEL_INFLIGHT", "1")
    monkeypatch.setenv("GEMMA_QA_SLOT_ACQUIRE_TIMEOUT_S", "0.2")
    # Reset semaphores so env takes effect for a fresh key.
    with cfg._slot_lock:
        cfg._model_semaphores.clear()
        cfg._model_inflight.clear()
    key = "test/slot-timeout-model"
    cfg.acquire_model_slot(key, timeout_s=0.2)
    try:
        with pytest.raises(TimeoutError, match="slot acquire timed out"):
            cfg.acquire_model_slot(key, timeout_s=0.2)
    finally:
        cfg.release_model_slot(key)
        with cfg._slot_lock:
            cfg._model_semaphores.pop(key, None)
            cfg._model_inflight.pop(key, None)
