from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .antigravity import AntigravityClient
from .client import GemmaClient, GenerationResult
from .config import MODEL_31B, MODEL_ANTIGRAVITY, MODEL_IDS
from .quota import QuotaGate

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def resolve_adjudication_model(client: object) -> str:
    selector = getattr(client, "adjudication_model", None)
    if callable(selector):
        chosen = selector()
        if chosen in {MODEL_31B, MODEL_ANTIGRAVITY}:
            return str(chosen)
    return MODEL_31B


class UnifiedQaClient:
    """Route generate/parse to Gemma or Antigravity by model id."""

    def __init__(
        self,
        *,
        gemma: GemmaClient,
        antigravity: AntigravityClient | None = None,
        quota: QuotaGate | None = None,
    ) -> None:
        self._gemma = gemma
        self._antigravity = antigravity
        self._quota = quota

    def adjudication_model(self) -> str:
        return select_adjudication_model(self._quota)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]:
        if model == MODEL_ANTIGRAVITY:
            if self._antigravity is None:
                raise RuntimeError("Antigravity client is not configured")
            return self._antigravity.generate(
                model=model,
                prompt=prompt,
                response_model=response_model,
                max_output_tokens=max_output_tokens,
            )
        if model not in MODEL_IDS:
            raise ValueError(f"unsupported model: {model}")
        return self._gemma.generate(
            model=model,
            prompt=prompt,
            response_model=response_model,
            max_output_tokens=max_output_tokens,
        )

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, object]:
        # Prefer Antigravity parser when response looks like Interactions output.
        if self._antigravity is not None and (
            "output_text" in response_json or "outputs" in response_json
        ):
            return self._antigravity.parse_response(response_json, response_model)
        return self._gemma.parse_response(response_json, response_model)

    def close(self) -> None:
        self._gemma.close()
        if self._antigravity is not None:
            self._antigravity.close()


def select_adjudication_model(quota: QuotaGate | None) -> str:
    """Prefer Antigravity for adjudication when daily RPD remains; else Gemma 31B."""
    if quota is None:
        return MODEL_31B
    if quota.remaining_daily_requests(MODEL_ANTIGRAVITY) > 0:
        return MODEL_ANTIGRAVITY
    return MODEL_31B
