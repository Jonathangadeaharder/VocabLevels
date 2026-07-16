from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts.gemma_qa.antigravity import AntigravityClient
from scripts.gemma_qa.config import (
    ANTIGRAVITY_HARD_REQUESTS_PER_DAY,
    ANTIGRAVITY_REQUESTS_PER_DAY,
    MODEL_ANTIGRAVITY,
    MODEL_CEILINGS,
    MODEL_31B,
)
from scripts.gemma_qa.quota import DailyQuotaExceeded, QuotaGate
from scripts.gemma_qa.routing import UnifiedQaClient, select_adjudication_model
from scripts.gemma_qa.schemas import CefrReviewBatch, ReviewAction, UPOS


class FixedEstimator:
    def count(self, text: str) -> int:
        return len(text)


def _interaction_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "output_text": text,
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8,
                "totalTokenCount": 20,
            },
        },
    )


def test_antigravity_ceilings_are_exact_ui_quotas() -> None:
    ceilings = MODEL_CEILINGS[MODEL_ANTIGRAVITY]
    assert ceilings.requests_per_minute == 60
    assert ceilings.tokens_per_minute == 100_000
    assert ceilings.requests_per_day == ANTIGRAVITY_REQUESTS_PER_DAY == 100
    assert ceilings.hard_requests_per_day == ANTIGRAVITY_HARD_REQUESTS_PER_DAY == 100
    assert ceilings.rpd_policy == "fail"


def test_antigravity_rpd_hard_fails_at_100(tmp_path: Path) -> None:
    gate = QuotaGate(tmp_path / "quota.sqlite3")
    for _ in range(100):
        gate.reserve(MODEL_ANTIGRAVITY, prompt_tokens=1, max_output_tokens=0)
    with pytest.raises(DailyQuotaExceeded, match="daily request ceiling 100"):
        gate.reserve(MODEL_ANTIGRAVITY, prompt_tokens=1, max_output_tokens=0)
    assert gate.remaining_daily_requests(MODEL_ANTIGRAVITY) == 0
    gate.close()


def test_antigravity_request_has_no_schema_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        assert request.headers.get("Api-Revision") == "2026-05-20"
        assert "generationConfig" not in payload
        assert "responseMimeType" not in payload
        assert "responseSchema" not in payload
        assert "responseJsonSchema" not in payload
        assert payload["tools"] == []
        assert payload["background"] is False
        assert payload["agent"] == MODEL_ANTIGRAVITY
        assert payload["environment"] == "remote"
        body = {
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
        return _interaction_response(json.dumps(body, ensure_ascii=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = AntigravityClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        ).generate(
            model=MODEL_ANTIGRAVITY,
            prompt='Return JSON: {"rows":[...]}',
            response_model=CefrReviewBatch,
            max_output_tokens=100,
        )
    assert result.parsed.rows[0].lemma == "Abend"
    assert len(captured) == 1


def test_antigravity_parses_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    body = {
        "rows": [
            {
                "id": "german:A1:1",
                "lemma": "Haus",
                "english_lemma": "house",
                "chinese_lemma": "房子",
                "upos": "NOUN",
                "action": "keep",
            }
        ]
    }
    fenced = f"```json\n{json.dumps(body)}\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return _interaction_response(fenced)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = AntigravityClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        ).generate(
            model=MODEL_ANTIGRAVITY,
            prompt="return json",
            response_model=CefrReviewBatch,
            max_output_tokens=50,
        )
    assert result.parsed.rows[0].upos is UPOS.NOUN
    assert result.parsed.rows[0].action is ReviewAction.KEEP


def test_antigravity_limits_repair_creates_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _interaction_response("not-json")
        body = {
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
        return _interaction_response(json.dumps(body))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = AntigravityClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        ).generate(
            model=MODEL_ANTIGRAVITY,
            prompt="return json",
            response_model=CefrReviewBatch,
            max_output_tokens=50,
        )
    assert calls == 2
    assert result.parsed.rows[0].lemma == "Abend"


def test_select_adjudication_model_prefers_antigravity_when_rpd_remains(
    tmp_path: Path,
) -> None:
    gate = QuotaGate(tmp_path / "quota.sqlite3")
    assert select_adjudication_model(gate) == MODEL_ANTIGRAVITY
    for _ in range(100):
        gate.reserve(MODEL_ANTIGRAVITY, prompt_tokens=1, max_output_tokens=0)
    assert select_adjudication_model(gate) == MODEL_31B
    gate.close()


def test_unified_client_routes_antigravity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")

    class StubGemma:
        def generate(self, **kwargs: object) -> object:
            raise AssertionError("gemma should not be called")

        def parse_response(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("gemma parse should not be called")

        def close(self) -> None:
            return None

    body = {
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

    def handler(request: httpx.Request) -> httpx.Response:
        return _interaction_response(json.dumps(body))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        antigravity = AntigravityClient(
            http_client=http_client,
            estimator=FixedEstimator(),
        )
        client = UnifiedQaClient(gemma=StubGemma(), antigravity=antigravity)  # type: ignore[arg-type]
        result = client.generate(
            model=MODEL_ANTIGRAVITY,
            prompt="json",
            response_model=CefrReviewBatch,
            max_output_tokens=10,
        )
    assert result.parsed.rows[0].lemma == "Abend"
