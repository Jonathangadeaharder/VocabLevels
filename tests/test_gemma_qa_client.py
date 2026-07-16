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
from scripts.gemma_qa.config import THINKING_CONFIG
from scripts.gemma_qa.packing import TokenEstimator
from scripts.gemma_qa.schemas import CefrReviewBatch


def gemini_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8,
                "totalTokenCount": 20,
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
    contents = request["contents"]
    assert isinstance(contents, list)
    content = contents[0]
    assert isinstance(content, dict)
    parts = content["parts"]
    assert isinstance(parts, list)
    part = parts[0]
    assert isinstance(part, dict)
    prompt = part["text"]
    assert isinstance(prompt, str)
    return prompt


def retry_info_response(delay: object) -> httpx.Response:
    return httpx.Response(
        429,
        json={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": delay,
                    }
                ]
            }
        },
    )


class FixedEstimator(TokenEstimator):
    def count(self, text: str) -> int:
        return len(text)


class RecordingQuota:
    def __init__(self) -> None:
        self.reservations: list[tuple[str, int, int]] = []
        self.reconciliations: list[tuple[str, int]] = []

    def reserve(
        self,
        model: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> str:
        self.reservations.append((model, prompt_tokens, max_output_tokens))
        return f"reservation-{len(self.reservations)}"

    def reconcile(self, reservation_id: str, *, actual_input_tokens: int) -> None:
        self.reconciliations.append((reservation_id, actual_input_tokens))


def test_client_retries_429_and_parses_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
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
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert result.usage.total_tokens == 20
    assert sleeps == [2.0]
    assert requests[0].headers["X-goog-api-key"] == "secret-value"


@pytest.mark.parametrize(
    ("delay", "expected_sleep"),
    [
        ("43s", 43.0),
        ("1.5s", 1.5),
        ({"seconds": 1, "nanos": 500_000_000}, 1.5),
        ("invalid", 1.25),
    ],
)
def test_client_uses_gemini_retry_info_delay(
    monkeypatch: pytest.MonkeyPatch,
    delay: object,
    expected_sleep: float,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return retry_info_response(delay)
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(
            http_client=http_client,
            sleeper=sleeps.append,
            jitter=lambda: 0.25,
        ).generate(
            model="gemma-4-26b-a4b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert calls == 2
    assert sleeps == [expected_sleep]


def test_default_transient_retry_budget_is_bounded_for_quota_windows() -> None:
    assert MAX_TRANSIENT_RETRIES == 8


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_client_does_not_retry_nontransient_client_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
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
                model="gemma-4-26b-a4b-it",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )

    assert calls == 1
    assert sleeps == []


def test_client_retries_read_timeout_and_reconciles_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
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
            quota=quota,
            estimator=FixedEstimator(),
            sleeper=sleeps.append,
            jitter=lambda: 0,
            max_retries=1,
        ).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert calls == 2
    assert sleeps == [1.0]
    assert quota.reconciliations == [
        ("reservation-1", len("review")),
        ("reservation-2", 12),
    ]


def test_client_raises_read_timeout_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = GemmaClient(
            http_client=http_client,
            quota=quota,
            estimator=FixedEstimator(),
            sleeper=sleeps.append,
            jitter=lambda: 0,
            max_retries=2,
        )
        with pytest.raises(httpx.ReadTimeout, match="read timed out"):
            client.generate(
                model="gemma-4-31b-it",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )

    assert calls == 3
    assert sleeps == [1.0, 2.0]
    assert quota.reconciliations == [
        ("reservation-1", len("review")),
        ("reservation-2", len("review")),
        ("reservation-3", len("review")),
    ]


def test_production_timeout_allows_large_model_responses() -> None:
    assert PRODUCTION_TIMEOUT.connect == 30
    assert PRODUCTION_TIMEOUT.read == 300
    assert PRODUCTION_TIMEOUT.write == 300
    assert PRODUCTION_TIMEOUT.pool == 300


def test_request_uses_minimal_supported_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        GemmaClient(http_client=http_client).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    generation_config = requests[0]["generationConfig"]
    assert isinstance(generation_config, dict)
    assert THINKING_CONFIG == {"thinkingLevel": "minimal"}
    assert generation_config["thinkingConfig"] == THINKING_CONFIG


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
    with pytest.raises(ValueError, match="nonempty candidate text"):
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
    with pytest.raises(ValueError, match="candidate"):
        GemmaClient.parse_response(response_json, CefrReviewBatch)


def test_client_repairs_invalid_structured_output_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_json = json.loads(request.content)
        requests.append(request_json)
        if len(requests) == 1:
            return gemini_response('{"rows":[]}')
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(http_client=http_client).generate(
            model="gemma-4-31b-it",
            prompt="original review request",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert len(requests) == 2
    contents = requests[1]["contents"]
    assert isinstance(contents, list)
    content = contents[0]
    assert isinstance(content, dict)
    parts = content["parts"]
    assert isinstance(parts, list)
    part = parts[0]
    assert isinstance(part, dict)
    repair_text = part["text"]
    assert isinstance(repair_text, str)
    assert "original review request" in repair_text
    assert "rows" in repair_text
    assert "validation" in repair_text.lower()


def test_client_succeeds_after_two_structured_repairs_and_reconciles_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
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
            quota=quota,
            estimator=FixedEstimator(),
        ).generate(
            model="gemma-4-31b-it",
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
    assert quota.reconciliations == [
        ("reservation-1", 12),
        ("reservation-2", 12),
        ("reservation-3", 12),
    ]


def test_client_raises_after_three_malformed_structured_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
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
            quota=quota,
            estimator=FixedEstimator(),
        )
        with pytest.raises(json.JSONDecodeError, match="Extra data"):
            client.generate(
                model="gemma-4-31b-it",
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
    assert quota.reconciliations == [
        ("reservation-1", 12),
        ("reservation-2", 12),
        ("reservation-3", 12),
    ]


def test_structured_attempt_budget_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
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
                model="gemma-4-31b-it",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )
    assert calls == 2


def test_client_falls_back_across_supported_schema_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    generation_configs: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_json = json.loads(request.content)
        generation_config = request_json["generationConfig"]
        generation_configs.append(generation_config)
        if len(generation_configs) < 3:
            return httpx.Response(
                400,
                json={"error": {"message": "response schema field is unsupported"}},
            )
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = GemmaClient(http_client=http_client).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert result.parsed.rows[0].lemma == "Abend"
    assert "responseJsonSchema" in generation_configs[0]
    assert "responseSchema" in generation_configs[1]
    assert "responseJsonSchema" not in generation_configs[2]
    assert "responseSchema" not in generation_configs[2]
    assert generation_configs[2]["responseMimeType"] == "application/json"


def test_client_uses_configured_schema_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    generation_configs: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_json = json.loads(request.content)
        generation_configs.append(request_json["generationConfig"])
        return gemini_response(json.dumps(valid_payload(), ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        GemmaClient(
            http_client=http_client,
            schema_property="responseSchema",
        ).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert "responseSchema" in generation_configs[0]
    assert "responseJsonSchema" not in generation_configs[0]


def test_failed_retry_reconciles_unused_token_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
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
            quota=quota,
            estimator=FixedEstimator(),
            sleeper=lambda _: None,
        ).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert quota.reservations == [
        ("gemma-4-31b-it", len("review"), 100),
        ("gemma-4-31b-it", len("review"), 100),
    ]
    assert quota.reconciliations == [
        ("reservation-1", len("review")),
        ("reservation-2", 12),
    ]


def test_empty_usage_metadata_keeps_conservative_prompt_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    quota = RecordingQuota()
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
            quota=quota,
            estimator=FixedEstimator(),
        ).generate(
            model="gemma-4-31b-it",
            prompt="review",
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )

    assert quota.reconciliations == [("reservation-1", len("review"))]


def test_client_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    transport = httpx.MockTransport(lambda request: gemini_response("{not json}"))
    with httpx.Client(transport=transport) as http_client:
        client = GemmaClient(http_client=http_client)
        with pytest.raises((ValueError, ValidationError)):
            client.generate(
                model="gemma-4-31b-it",
                prompt="review",
                response_model=CefrReviewBatch,
                max_output_tokens=100,
            )
