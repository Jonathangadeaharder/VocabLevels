from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .client import GenerationResult, Usage
from .config import (
    ANTIGRAVITY_API_REVISION,
    ANTIGRAVITY_INPUT_TOKEN_CAP,
    API_BASE,
    INTERACTIONS_PATH,
    MODEL_ANTIGRAVITY,
    get_api_key,
)
from .packing import TiktokenEstimator, TokenEstimator
from .quota import QuotaGate

ResponseT = TypeVar("ResponseT", bound=BaseModel)
PRODUCTION_TIMEOUT = httpx.Timeout(connect=30, read=300, write=300, pool=300)
# Each create() costs 1 RPD. Allow one create + one repair create max.
DEFAULT_ANTIGRAVITY_STRUCTURED_ATTEMPTS = 2
_JSON_FENCE = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.DOTALL | re.IGNORECASE,
)


class AntigravityClient:
    """Interactions API client for Antigravity (no structured-output schema)."""

    def __init__(
        self,
        *,
        quota: QuotaGate | None = None,
        http_client: httpx.Client | None = None,
        estimator: TokenEstimator | None = None,
        sleeper: Callable[[float], None] | None = None,
        max_retries: int = 4,
        structured_attempts: int = DEFAULT_ANTIGRAVITY_STRUCTURED_ATTEMPTS,
        api_base: str = API_BASE,
    ) -> None:
        if structured_attempts <= 0:
            raise ValueError("structured_attempts must be positive")
        if structured_attempts > 2:
            raise ValueError(
                "Antigravity structured_attempts must be <= 2 (1 RPD per create)"
            )
        self._api_key = get_api_key()
        self._http_client = http_client or httpx.Client(timeout=PRODUCTION_TIMEOUT)
        self._owns_client = http_client is None
        self._quota = quota
        self._estimator = estimator or TiktokenEstimator()
        self._sleeper = sleeper or (lambda _: None)
        self._max_retries = max_retries
        self._structured_attempts = structured_attempts
        self._api_base = api_base.rstrip("/")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]:
        if model != MODEL_ANTIGRAVITY:
            raise ValueError(f"AntigravityClient only supports {MODEL_ANTIGRAVITY}")
        # max_output_tokens is ignored by Interactions API; kept for Protocol parity.
        _ = max_output_tokens
        current_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(self._structured_attempts):
            request_json, response_json = self._create_interaction(current_prompt)
            try:
                parsed, usage = self.parse_response(response_json, response_model)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_error = error
                if attempt + 1 >= self._structured_attempts:
                    raise
                current_prompt = self._repair_prompt(
                    original_prompt=prompt,
                    response_json=response_json,
                    error=error,
                )
                continue
            return GenerationResult(
                parsed=parsed,
                usage=usage,
                request_json=request_json,
                response_json=response_json,
            )
        raise RuntimeError(
            f"Antigravity structured attempts exhausted: {last_error}"
        )

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, Usage]:
        text = self._extract_output_text(response_json)
        payload = self._extract_json_value(text)
        parsed = response_model.model_validate(payload)
        usage = self._parse_usage(response_json)
        return parsed, usage

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    def _create_interaction(
        self,
        prompt: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        prompt_tokens = self._estimator.count(prompt)
        if prompt_tokens > ANTIGRAVITY_INPUT_TOKEN_CAP:
            raise ValueError("Antigravity prompt exceeds input token cap")
        request_json: dict[str, object] = {
            "agent": MODEL_ANTIGRAVITY,
            "input": prompt,
            "environment": "remote",
            "background": False,
            # Empty tools list restricts default search/code/url tools.
            "tools": [],
        }
        # Explicitly reject schema-shaped generation configs if ever attached.
        if "generationConfig" in request_json or "responseMimeType" in request_json:
            raise AssertionError("Antigravity must not send structured-output config")
        url = f"{self._api_base}{INTERACTIONS_PATH}"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
            "Api-Revision": ANTIGRAVITY_API_REVISION,
        }
        for attempt in range(self._max_retries + 1):
            reservation = None
            if self._quota is not None:
                reservation = self._quota.reserve(
                    MODEL_ANTIGRAVITY,
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=0,
                )
            try:
                response = self._http_client.post(
                    url,
                    headers=headers,
                    json=request_json,
                )
            except httpx.TransportError:
                self._reconcile(reservation, prompt_tokens)
                if attempt >= self._max_retries:
                    raise
                self._sleeper(min(2**attempt, 30))
                continue
            except Exception:
                self._reconcile(reservation, prompt_tokens)
                raise
            if response.status_code == 429 or 500 <= response.status_code < 600:
                self._reconcile(reservation, prompt_tokens)
                if attempt >= self._max_retries:
                    response.raise_for_status()
                self._sleeper(min(2**attempt, 30))
                continue
            response.raise_for_status()
            response_json = response.json()
            if not isinstance(response_json, dict):
                self._reconcile(reservation, prompt_tokens)
                raise ValueError("Interactions response must be a JSON object")
            usage = self._parse_usage(response_json)
            actual = usage.prompt_tokens if usage.prompt_tokens > 0 else prompt_tokens
            self._reconcile(reservation, actual)
            return request_json, response_json
        raise RuntimeError("Antigravity HTTP retry loop exhausted")

    def _reconcile(self, reservation: str | None, token_count: int) -> None:
        if reservation is not None and self._quota is not None:
            self._quota.reconcile(reservation, actual_input_tokens=token_count)

    @classmethod
    def _extract_output_text(cls, response_json: dict[str, object]) -> str:
        direct = response_json.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        outputs = response_json.get("outputs")
        if isinstance(outputs, list):
            chunks: list[str] = []
            for item in outputs:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            joined = "".join(chunks).strip()
            if joined:
                return joined
        raise ValueError("Interactions response missing output_text")

    @classmethod
    def _extract_json_value(cls, text: str) -> object:
        stripped = text.strip()
        fence = _JSON_FENCE.search(stripped)
        if fence is not None:
            stripped = fence.group(1).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start_obj = stripped.find("{")
            start_arr = stripped.find("[")
            starts = [index for index in (start_obj, start_arr) if index >= 0]
            if not starts:
                raise
            start = min(starts)
            end_obj = stripped.rfind("}")
            end_arr = stripped.rfind("]")
            end = max(end_obj, end_arr)
            if end <= start:
                raise
            return json.loads(stripped[start : end + 1])

    @staticmethod
    def _parse_usage(response_json: dict[str, object]) -> Usage:
        metadata = response_json.get("usageMetadata") or response_json.get("usage")
        if not isinstance(metadata, dict):
            return Usage(0, 0, 0)
        prompt = int(metadata.get("promptTokenCount") or metadata.get("input_tokens") or 0)
        candidate = int(
            metadata.get("candidatesTokenCount") or metadata.get("output_tokens") or 0
        )
        total = int(metadata.get("totalTokenCount") or prompt + candidate)
        return Usage(prompt, candidate, total)

    @classmethod
    def _repair_prompt(
        cls,
        *,
        original_prompt: str,
        response_json: dict[str, object],
        error: Exception,
    ) -> str:
        try:
            invalid_output = cls._extract_output_text(response_json)
        except ValueError:
            invalid_output = json.dumps(response_json, ensure_ascii=False)
        return (
            "Repair the previous answer. Return ONLY one JSON value matching the "
            "requested schema. No markdown, no tools, no explanation.\n"
            f"Validation errors: {error}\n"
            f"Original request:\n{original_prompt}\n"
            f"Invalid response:\n{invalid_output}"
        )
