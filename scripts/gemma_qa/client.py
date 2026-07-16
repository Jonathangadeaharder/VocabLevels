from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Generic, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import API_BASE, RESPONSE_SCHEMA_PROPERTY, THINKING_CONFIG, get_api_key
from .packing import TiktokenEstimator, TokenEstimator

ResponseT = TypeVar("ResponseT", bound=BaseModel)
SchemaProperty = Literal["responseJsonSchema", "responseSchema"]
PRODUCTION_TIMEOUT = httpx.Timeout(connect=30, read=300, write=300, pool=300)
MAX_TRANSIENT_RETRIES = 8
DEFAULT_STRUCTURED_ATTEMPTS = 3


class QuotaLimiter(Protocol):
    def reserve(
        self,
        model: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> str: ...

    def reconcile(
        self,
        reservation_id: str,
        *,
        actual_input_tokens: int,
    ) -> None: ...


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    candidate_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class GenerationResult(Generic[ResponseT]):
    parsed: ResponseT
    usage: Usage
    request_json: dict[str, object]
    response_json: dict[str, object]


class GemmaClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        quota: QuotaLimiter | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        estimator: TokenEstimator | None = None,
        max_retries: int = MAX_TRANSIENT_RETRIES,
        structured_attempts: int = DEFAULT_STRUCTURED_ATTEMPTS,
        api_base: str = API_BASE,
        schema_property: SchemaProperty = RESPONSE_SCHEMA_PROPERTY,
    ) -> None:
        if structured_attempts <= 0:
            raise ValueError("structured_attempts must be positive")
        self._api_key = get_api_key()
        self._http_client = http_client or httpx.Client(timeout=PRODUCTION_TIMEOUT)
        self._owns_client = http_client is None
        self._quota = quota
        self._sleeper = sleeper
        self._jitter = jitter
        self._estimator = estimator or TiktokenEstimator()
        self._max_retries = max_retries
        self._structured_attempts = structured_attempts
        self._api_base = api_base.rstrip("/")
        self._schema_property: SchemaProperty = schema_property

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]:
        url = f"{self._api_base}/v1beta/models/{model}:generateContent"
        headers = {
            "X-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        current_prompt = prompt
        for structured_attempt in range(self._structured_attempts):
            request_json, response_json = self._request_with_schema_fallback(
                url=url,
                headers=headers,
                model=model,
                prompt=current_prompt,
                response_model=response_model,
                max_output_tokens=max_output_tokens,
            )
            try:
                parsed, usage = self.parse_response(response_json, response_model)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                if structured_attempt + 1 >= self._structured_attempts:
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
        raise RuntimeError("structured retry loop exhausted")

    def _request_with_schema_fallback(
        self,
        *,
        url: str,
        headers: dict[str, str],
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        other_property: SchemaProperty = (
            "responseSchema"
            if self._schema_property == "responseJsonSchema"
            else "responseJsonSchema"
        )
        schema_modes: tuple[SchemaProperty | None, ...] = (
            self._schema_property,
            other_property,
            None,
        )
        for schema_property in schema_modes:
            request_json = self._build_request(
                prompt=prompt,
                response_model=response_model,
                max_output_tokens=max_output_tokens,
                schema_property=schema_property,
            )
            response = self._send_with_retries(
                url=url,
                headers=headers,
                model=model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                request_json=request_json,
            )
            if self._is_schema_unsupported(response):
                continue
            response.raise_for_status()
            response_json = response.json()
            if not isinstance(response_json, dict):
                raise ValueError("Gemini response body must be a JSON object")
            return request_json, response_json
        raise RuntimeError("schema fallback loop exhausted")

    def _send_with_retries(
        self,
        *,
        url: str,
        headers: dict[str, str],
        model: str,
        prompt: str,
        max_output_tokens: int,
        request_json: dict[str, object],
    ) -> httpx.Response:
        prompt_tokens = self._estimator.count(prompt)
        for attempt in range(self._max_retries + 1):
            reservation = None
            if self._quota is not None:
                reservation = self._quota.reserve(
                    model,
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=max_output_tokens,
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
                self._sleeper(self._exponential_retry_delay(attempt))
                continue
            except Exception:
                self._reconcile(reservation, prompt_tokens)
                raise
            if response.status_code == 429 or 500 <= response.status_code < 600:
                self._reconcile(reservation, prompt_tokens)
                if attempt >= self._max_retries:
                    response.raise_for_status()
                self._sleeper(self._retry_delay(response, attempt))
                continue
            usage = self._usage_for_reservation(response, prompt_tokens)
            self._reconcile(reservation, usage)
            return response
        raise RuntimeError("HTTP retry loop exhausted")

    def _reconcile(self, reservation: str | None, token_count: int) -> None:
        if reservation is not None and self._quota is not None:
            self._quota.reconcile(reservation, actual_input_tokens=token_count)

    @classmethod
    def _usage_for_reservation(
        cls,
        response: httpx.Response,
        prompt_tokens: int,
    ) -> int:
        try:
            response_json = response.json()
        except ValueError:
            return prompt_tokens
        if not isinstance(response_json, dict) or not isinstance(
            response_json.get("usageMetadata"), dict
        ):
            return prompt_tokens
        # TPM accounting uses prompt/input tokens only, not candidates/total.
        usage = cls._parse_usage(response_json)
        return usage.prompt_tokens if usage.prompt_tokens > 0 else prompt_tokens

    @staticmethod
    def _build_request(
        *,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
        schema_property: SchemaProperty | None,
    ) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": dict(THINKING_CONFIG),
        }
        if schema_property is not None:
            generation_config[schema_property] = response_model.model_json_schema()
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

    @staticmethod
    def _is_schema_unsupported(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        message = response.text.casefold()
        return "schema" in message and any(
            marker in message
            for marker in ("unsupported", "not supported", "unknown", "unrecognized")
        )

    @classmethod
    def _repair_prompt(
        cls,
        *,
        original_prompt: str,
        response_json: dict[str, object],
        error: Exception,
    ) -> str:
        try:
            invalid_output = cls._extract_text(response_json)
        except ValueError:
            invalid_output = json.dumps(response_json, ensure_ascii=False)
        return (
            "Repair the structured JSON response. Return only one JSON value matching "
            "the requested schema.\n"
            f"Validation errors: {cls._concise_error(error)}\n"
            f"Original request:\n{original_prompt}\n"
            f"Invalid response:\n{invalid_output}"
        )

    @staticmethod
    def _concise_error(error: Exception) -> str:
        if isinstance(error, ValidationError):
            messages = [
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors(include_url=False, include_input=False)[:5]
            ]
            return "; ".join(messages)
        if isinstance(error, json.JSONDecodeError):
            return f"invalid JSON at line {error.lineno}, column {error.colno}"
        return str(error).splitlines()[0][:500]

    def close(self) -> None:
        if self._owns_client:
            self._http_client.close()

    @classmethod
    def parse_response(
        cls,
        response_json: dict[str, object],
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, Usage]:
        text = cls._extract_text(response_json)
        parsed_json = json.loads(cls._strip_fences(text))
        return response_model.model_validate(parsed_json), cls._parse_usage(
            response_json
        )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    retry_at = None
                if retry_at is not None:
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(
                        0.0,
                        (retry_at - datetime.now(timezone.utc)).total_seconds(),
                    )
            else:
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
        retry_info_delay = self._gemini_retry_delay(response)
        if retry_info_delay is not None:
            return retry_info_delay
        return self._exponential_retry_delay(attempt)

    @classmethod
    def _gemini_retry_delay(cls, response: httpx.Response) -> float | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        details = error.get("details")
        if not isinstance(details, list):
            return None
        for detail in details:
            if not isinstance(detail, dict):
                continue
            detail_type = detail.get("@type")
            if not isinstance(detail_type, str) or not detail_type.endswith(
                "google.rpc.RetryInfo"
            ):
                continue
            parsed = cls._parse_retry_delay(detail.get("retryDelay"))
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _parse_retry_delay(cls, value: object) -> float | None:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped.endswith("s"):
                return None
            return cls._nonnegative_finite_number(stripped[:-1])
        if not isinstance(value, dict) or not ({"seconds", "nanos"} & value.keys()):
            return None
        seconds = cls._nonnegative_finite_number(value.get("seconds", 0))
        nanos = cls._nonnegative_finite_number(value.get("nanos", 0))
        if seconds is None or nanos is None or nanos >= 1_000_000_000:
            return None
        return seconds + nanos / 1_000_000_000

    @staticmethod
    def _nonnegative_finite_number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0 else None

    def _exponential_retry_delay(self, attempt: int) -> float:
        return min(60.0, 2**attempt + self._jitter())

    @staticmethod
    def _strip_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```") or not stripped.endswith("```"):
            return stripped
        opening, separator, body = stripped.partition("\n")
        if not separator or opening.casefold() not in {"```", "```json"}:
            return stripped
        return body[:-3].strip()

    @staticmethod
    def _extract_text(response_json: dict[str, object]) -> str:
        candidates = response_json.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini response has no candidate text")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ValueError("Gemini response has no candidate text")
        content = candidate.get("content")
        if not isinstance(content, dict):
            raise ValueError("Gemini response has no candidate text")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Gemini response has no candidate text")
        output_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Gemini candidate parts must be objects")
            if part.get("thought") is True:
                continue
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Gemini candidate text must be a string")
            output_parts.append(text)
        output = "".join(output_parts)
        if not output.strip():
            raise ValueError("Gemini response has no nonempty candidate text")
        return output

    @staticmethod
    def _parse_usage(response_json: dict[str, object]) -> Usage:
        metadata = response_json.get("usageMetadata")
        if not isinstance(metadata, dict):
            return Usage(0, 0, 0)
        prompt = int(metadata.get("promptTokenCount", 0))
        candidate = int(metadata.get("candidatesTokenCount", 0))
        total = int(metadata.get("totalTokenCount", prompt + candidate))
        return Usage(prompt, candidate, total)
