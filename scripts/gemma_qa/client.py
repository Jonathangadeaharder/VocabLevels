from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import (
    API_BASE,
    CHAT_COMPLETIONS_PATH,
    REASONING_EFFORT,
    ModelSpec,
    acquire_model_slot,
    get_api_key,
    release_model_slot,
    resolve_model_spec,
)
from .packing import TiktokenEstimator, TokenEstimator
from .trace import event, log_bodies_enabled, summarize_parsed

ResponseT = TypeVar("ResponseT", bound=BaseModel)
# Idle read alone is not enough: slow-streamed hang can reset read timer forever.
# connect stays short; read is an idle bound; wall-clock caps total POST lifetime.
PRODUCTION_TIMEOUT = httpx.Timeout(connect=30, read=120, write=120, pool=60)
MAX_TRANSIENT_RETRIES = 8
DEFAULT_STRUCTURED_ATTEMPTS = 3
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def request_wall_clock_s() -> float:
    """Hard max seconds for one HTTP POST (default 180).

    Env: GEMMA_QA_REQUEST_WALL_S.
    """
    raw = os.environ.get("GEMMA_QA_REQUEST_WALL_S", "180")
    try:
        value = float(raw)
    except ValueError:
        value = 180.0
    return max(0.1, min(value, 900.0))


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
    """OpenAI-compatible multi-endpoint TNG client.

    Routes each model key to internal or external gateway via MODEL_REGISTRY.
    Name kept for package compatibility.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        quota: object | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        estimator: TokenEstimator | None = None,
        max_retries: int = MAX_TRANSIENT_RETRIES,
        structured_attempts: int = DEFAULT_STRUCTURED_ATTEMPTS,
        api_base: str = API_BASE,
        reasoning_effort: str = REASONING_EFFORT,
    ) -> None:
        # quota accepted but ignored (TNG load-balances; no client-side gate).
        _ = quota
        if structured_attempts <= 0:
            raise ValueError("structured_attempts must be positive")
        self._api_key = get_api_key()
        # Wide pool: dual x batch concurrency across internal + external.
        self._http_client = http_client or httpx.Client(
            timeout=PRODUCTION_TIMEOUT,
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=32,
            ),
        )
        self._owns_client = http_client is None
        self._sleeper = sleeper
        self._jitter = jitter
        self._estimator = estimator or TiktokenEstimator()
        self._max_retries = max_retries
        self._structured_attempts = structured_attempts
        # Default base only used if a model has no registry entry (should not happen).
        self._api_base = api_base.rstrip("/")
        self._reasoning_effort = reasoning_effort
        self._request_wall_s = request_wall_clock_s()

    def _send_or_disable(
        self,
        spec: ModelSpec,
        url: str,
        headers: dict[str, str],
        current_prompt: str,
        request_json: dict[str, object],
    ) -> dict[str, object]:
        """Send with retries; disable the model and re-raise on optional-422."""
        try:
            return self._send_with_retries(
                url=url,
                headers=headers,
                model=spec.key,
                prompt=current_prompt,
                request_json=request_json,
                optional=spec.optional,
            )
        except httpx.HTTPStatusError as error:
            if (
                spec.optional
                and error.response is not None
                and error.response.status_code == 422
            ):
                from .config import mark_model_unavailable

                mark_model_unavailable(spec.key)
                event(
                    "generate.model_disabled",
                    level="WARN",
                    model=spec.key,
                    http_status=422,
                    reason="optional model unavailable (422)",
                )
            raise

    def _parse_or_repair(
        self,
        *,
        model: str,
        attempt: int,
        original_prompt: str,
        response_json: dict[str, object],
        error: Exception,
    ) -> str:
        """Log a parse failure; re-raise on the last attempt, else repair prompt."""
        event(
            "generate.parse_error",
            level="WARN",
            model=model,
            attempt=attempt,
            error=self._concise_error(error),
            prompt_tokens=self._parse_usage(response_json).prompt_tokens,
            candidate_tokens=self._parse_usage(response_json).candidate_tokens,
            response_preview=(
                json.dumps(response_json, ensure_ascii=False)[:800]
                if log_bodies_enabled()
                else None
            ),
        )
        if attempt >= self._structured_attempts:
            raise error
        return self._repair_prompt(
            original_prompt=original_prompt,
            response_json=response_json,
            error=error,
        )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]:
        from .config import is_model_available

        spec = resolve_model_spec(model)
        if not is_model_available(spec.key):
            raise ValueError(f"model unavailable: {spec.key}")
        url = f"{spec.api_base.rstrip('/')}{CHAT_COMPLETIONS_PATH}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        current_prompt = prompt
        started = time.time()
        # Cap concurrent HTTP per model (default 2) so independent batches can
        # fill every model without overloading a single backend.
        acquire_model_slot(spec.key)
        try:
            event(
                "generate.start",
                model=spec.key,
                wire_id=spec.wire_id,
                api_base=spec.api_base,
                response_model=response_model.__name__,
                max_output_tokens=max_output_tokens,
                prompt_chars=len(prompt),
                provider="tng",
                prompt_preview=prompt[:400] if log_bodies_enabled() else None,
            )
            for structured_attempt in range(self._structured_attempts):
                request_json = self._build_request(
                    model_key=spec.key,
                    prompt=current_prompt,
                    response_model=response_model,
                    max_output_tokens=max_output_tokens,
                )
                response_json = self._send_or_disable(
                    spec, url, headers, current_prompt, request_json
                )
                try:
                    parsed, usage = self.parse_response(response_json, response_model)
                except (json.JSONDecodeError, ValidationError, ValueError) as error:
                    current_prompt = self._parse_or_repair(
                        model=spec.key,
                        attempt=structured_attempt + 1,
                        original_prompt=prompt,
                        response_json=response_json,
                        error=error,
                    )
                    continue
                event(
                    "generate.ok",
                    model=spec.key,
                    wire_id=spec.wire_id,
                    response_model=response_model.__name__,
                    attempt=structured_attempt + 1,
                    duration_ms=int((time.time() - started) * 1000),
                    prompt_tokens=usage.prompt_tokens,
                    candidate_tokens=usage.candidate_tokens,
                    total_tokens=usage.total_tokens,
                    summary=summarize_parsed(parsed),
                    response_preview=(
                        json.dumps(response_json, ensure_ascii=False)[:800]
                        if log_bodies_enabled()
                        else None
                    ),
                )
                return GenerationResult(
                    parsed=parsed,
                    usage=usage,
                    request_json=request_json,
                    response_json=response_json,
                )
            raise RuntimeError("structured retry loop exhausted")
        finally:
            release_model_slot(spec.key)

    def _post_with_wall_clock(
        self,
        *,
        url: str,
        headers: dict[str, str],
        request_json: dict[str, object],
        wall_s: float,
    ) -> httpx.Response:
        """POST with hard wall-clock. Idle read timeout alone can hang forever on
        slow-drip streams; abort and close the stream even if bytes keep arriving.
        """
        deadline = time.monotonic() + wall_s
        request = httpx.Request("POST", url, headers=headers, json=request_json)
        try:
            with self._http_client.stream(
                "POST",
                url,
                headers=headers,
                json=request_json,
            ) as streamed:
                if time.monotonic() >= deadline:
                    streamed.close()
                    event(
                        "generate.wall_clock_timeout",
                        level="WARN",
                        wall_s=wall_s,
                        phase="headers",
                    )
                    raise httpx.ReadTimeout(
                        f"request wall clock exceeded after {wall_s:.0f}s",
                        request=request,
                    )
                chunks: list[bytes] = []
                for chunk in streamed.iter_bytes():
                    if time.monotonic() >= deadline:
                        streamed.close()
                        event(
                            "generate.wall_clock_timeout",
                            level="WARN",
                            wall_s=wall_s,
                            phase="body",
                            bytes_read=sum(len(part) for part in chunks),
                        )
                        raise httpx.ReadTimeout(
                            f"request wall clock exceeded after {wall_s:.0f}s",
                            request=request,
                        )
                    chunks.append(chunk)
                return httpx.Response(
                    status_code=streamed.status_code,
                    headers=streamed.headers,
                    content=b"".join(chunks),
                    request=streamed.request,
                )
        except httpx.ReadTimeout:
            raise
        except httpx.TransportError:
            raise

    def _log_transport_error(
        self,
        model: str,
        attempt: int,
        error: httpx.TransportError,
        prompt_tokens: int,
        wall_s: float,
        wall_failures: int,
    ) -> float:
        delay = self._exponential_retry_delay(attempt)
        event(
            "generate.transport_error",
            level="WARN",
            model=model,
            attempt=attempt + 1,
            error=str(error)[:300],
            wait_s=round(delay, 2),
            prompt_tokens=prompt_tokens,
            wall_s=wall_s,
            wall_failures=wall_failures,
        )
        return delay

    def _sleep_retryable_status(
        self,
        response: httpx.Response,
        model: str,
        attempt: int,
        http_started: float,
        prompt_tokens: int,
    ) -> None:
        """Back off after a retryable status; raise for status on the last attempt."""
        delay = self._retry_delay(response, attempt)
        event(
            "generate.retry",
            level="WARN",
            model=model,
            attempt=attempt + 1,
            http_status=response.status_code,
            wait_s=round(delay, 2),
            duration_ms=int((time.time() - http_started) * 1000),
            prompt_tokens=prompt_tokens,
            error=response.text[:300],
        )
        if attempt >= self._max_retries:
            response.raise_for_status()
        self._sleeper(delay)

    def _fail_http_error(
        self, response: httpx.Response, model: str, optional: bool
    ) -> None:
        event(
            "generate.http_error",
            level="ERROR",
            model=model,
            http_status=response.status_code,
            error=response.text[:500],
            optional=optional,
        )
        if optional and response.status_code == 422:
            from .config import mark_model_unavailable

            mark_model_unavailable(model)
        response.raise_for_status()

    def _send_with_retries(
        self,
        *,
        url: str,
        headers: dict[str, str],
        model: str,
        prompt: str,
        request_json: dict[str, object],
        optional: bool = False,
    ) -> dict[str, object]:
        prompt_tokens = self._estimator.count(prompt)
        wall_s = self._request_wall_s
        # Wall-clock hangs are expensive; do not burn full retry budget on them.
        wall_failures = 0
        max_wall_failures = 2
        for attempt in range(self._max_retries + 1):
            http_started = time.time()
            try:
                response = self._post_with_wall_clock(
                    url=url,
                    headers=headers,
                    request_json=request_json,
                    wall_s=wall_s,
                )
            except httpx.TransportError as error:
                if "wall clock" in str(error).lower():
                    wall_failures += 1
                delay = self._log_transport_error(
                    model, attempt, error, prompt_tokens, wall_s, wall_failures
                )
                if attempt >= self._max_retries or wall_failures >= max_wall_failures:
                    raise
                self._sleeper(delay)
                continue
            except Exception as error:
                event(
                    "generate.unexpected_error",
                    level="ERROR",
                    model=model,
                    attempt=attempt + 1,
                    error=str(error)[:300],
                )
                raise
            if response.status_code == 429 or 500 <= response.status_code < 600:
                self._sleep_retryable_status(
                    response, model, attempt, http_started, prompt_tokens
                )
                continue
            if response.status_code >= 400:
                self._fail_http_error(response, model, optional)
            response_json = response.json()
            if not isinstance(response_json, dict):
                raise ValueError("chat completion body must be a JSON object")
            usage = self._parse_usage(response_json)
            actual = usage.prompt_tokens if usage.prompt_tokens > 0 else prompt_tokens
            event(
                "generate.http_ok",
                model=model,
                attempt=attempt + 1,
                http_status=response.status_code,
                duration_ms=int((time.time() - http_started) * 1000),
                prompt_tokens=actual,
            )
            return response_json
        raise RuntimeError("HTTP retry loop exhausted")

    def _build_request(
        self,
        *,
        model_key: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> dict[str, object]:
        from .model_strategies import get_strategy

        strategy = get_strategy(model_key)
        schema = response_model.model_json_schema()
        request: dict[str, object] = {
            "model": strategy.wire_id(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": False,
                },
            },
        }
        # Strategy owns model-specific fields (e.g. GLM thinking off).
        request.update(strategy.request_extras())
        # Env override still wins if set explicitly on client.
        if self._reasoning_effort and "reasoning_effort" not in request:
            request["reasoning_effort"] = self._reasoning_effort
        return request

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

    @staticmethod
    def _retry_after_http_date(retry_after: str) -> float | None:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    @classmethod
    def _parse_retry_after(cls, retry_after: str) -> float | None:
        try:
            seconds = float(retry_after)
        except ValueError:
            return cls._retry_after_http_date(retry_after)
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
        return None

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            seconds = self._parse_retry_after(retry_after)
            if seconds is not None:
                return seconds
        return self._exponential_retry_delay(attempt)

    def _exponential_retry_delay(self, attempt: int) -> float:
        return min(60.0, 2**attempt + self._jitter())

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

    @staticmethod
    def _strip_thinking(text: str, *, model_key: str | None = None) -> str:
        """Strategy-aware think strip; never treat reasoning as the answer."""
        if model_key is not None:
            try:
                from .model_strategies import get_strategy

                return get_strategy(model_key).strip_output(text)
            except ValueError:
                pass
        stripped = text.strip()
        stripped = _THINK_CLOSED_RE.sub("", stripped).strip()
        lower = stripped.lower()
        if "<think>" in lower:
            stripped = stripped[: lower.find("<think>")].strip()
        return re.sub(r"</think>", "", stripped, flags=re.IGNORECASE).strip()

    @staticmethod
    def _strip_fences(text: str, *, model_key: str | None = None) -> str:
        stripped = GemmaClient._strip_thinking(text, model_key=model_key)
        if not stripped.startswith("```") or not stripped.endswith("```"):
            return stripped
        opening, separator, body = stripped.partition("\n")
        if not separator or opening.casefold() not in {"```", "```json"}:
            return stripped
        return body[:-3].strip()

    @staticmethod
    def _openai_text(response_json: dict[str, object]) -> str | None:
        # OpenAI chat.completion shape
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                # Never fall back to reasoning_content: on TNG GLM it is
                # null while think tags live inside content.
                return content
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return None

    @staticmethod
    def _gemini_parts_text(candidate: dict[str, object]) -> str | None:
        content = candidate.get("content")
        if not isinstance(content, dict):
            return None
        parts = content.get("parts")
        if not isinstance(parts, list):
            return None
        chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("thought") is True:
                continue
            part_text = part.get("text")
            if isinstance(part_text, str):
                chunks.append(part_text)
        joined = "".join(chunks)
        if joined.strip():
            return joined
        return None

    @staticmethod
    def _gemini_text(response_json: dict[str, object]) -> str | None:
        # Legacy Gemini shape (tests / old checkpoints)
        candidates = response_json.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            return None
        return GemmaClient._gemini_parts_text(candidate)

    @staticmethod
    def _extract_text(response_json: dict[str, object]) -> str:
        text = GemmaClient._openai_text(response_json)
        if text is not None:
            return text
        text = GemmaClient._gemini_text(response_json)
        if text is not None:
            return text
        raise ValueError("chat completion has no message content")

    @staticmethod
    def _parse_usage(response_json: dict[str, object]) -> Usage:
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            total = int(usage.get("total_tokens", prompt + completion) or 0)
            return Usage(prompt, completion, total)
        metadata = response_json.get("usageMetadata")
        if isinstance(metadata, dict):
            prompt = int(metadata.get("promptTokenCount", 0))
            candidate = int(metadata.get("candidatesTokenCount", 0))
            total = int(metadata.get("totalTokenCount", prompt + candidate))
            return Usage(prompt, candidate, total)
        return Usage(0, 0, 0)
