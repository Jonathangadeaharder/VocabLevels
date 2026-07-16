from __future__ import annotations

import json
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from .client import GenerationResult, Usage
from .ledger import Checkpoint, Ledger, prompt_hash

ResponseT = TypeVar("ResponseT", bound=BaseModel)
DEFAULT_SEMANTIC_ATTEMPTS = 3


class StructuredClient(Protocol, Generic[ResponseT]):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]: ...

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, object]: ...


def checkpointed_semantic_generate(
    *,
    client: StructuredClient[ResponseT],
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    response_model: type[ResponseT],
    max_output_tokens: int,
    validate: Callable[[ResponseT], ResponseT],
    expected_identity: object,
    semantic_attempts: int = DEFAULT_SEMANTIC_ATTEMPTS,
) -> ResponseT:
    if semantic_attempts <= 0:
        raise ValueError("semantic_attempts must be positive")
    digest = prompt_hash(prompt)
    existing = ledger.get(digest, model, batch_id)
    if existing is not None:
        try:
            parsed, _ = client.parse_response(existing.response_json, response_model)
            return validate(parsed)
        except ValueError:
            ledger.delete(digest, model, batch_id)

    current_prompt = prompt
    provenance: list[dict[str, object]] = []
    combined_usage = Usage(0, 0, 0)
    for attempt_number in range(semantic_attempts):
        result = client.generate(
            model=model,
            prompt=current_prompt,
            response_model=response_model,
            max_output_tokens=max_output_tokens,
        )
        combined_usage = _combined_usage(combined_usage, result.usage)
        attempt: dict[str, object] = {
            "request_json": result.request_json,
            "response_json": result.response_json,
        }
        try:
            parsed = validate(result.parsed)
        except ValueError as error:
            attempt["validation_error"] = _concise_error(error)
            provenance.append(attempt)
            if attempt_number + 1 >= semantic_attempts:
                raise
            current_prompt = _build_semantic_repair_prompt(
                original_prompt=prompt,
                error=error,
                expected_identity=expected_identity,
                invalid_output=result.parsed,
            )
            continue
        provenance.append(attempt)
        ledger.store(
            Checkpoint(
                prompt_hash=digest,
                model=model,
                batch_id=batch_id,
                request_json={"semantic_attempts": provenance},
                response_json=result.response_json,
                usage=combined_usage,
            )
        )
        return parsed
    raise RuntimeError("semantic attempt loop exhausted")


def _build_semantic_repair_prompt(
    *,
    original_prompt: str,
    error: ValueError,
    expected_identity: object,
    invalid_output: BaseModel,
) -> str:
    return (
        "Repair the semantically invalid structured response. Return only JSON "
        "matching the requested schema and exact identity.\n"
        f"Validation error: {_concise_error(error)}\n"
        "Exact expected identity:\n"
        f"{json.dumps(expected_identity, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Original request:\n{original_prompt}\n"
        "Invalid output:\n"
        f"{invalid_output.model_dump_json()}"
    )


def _concise_error(error: ValueError) -> str:
    return str(error).splitlines()[0][:500]


def _combined_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        candidate_tokens=first.candidate_tokens + second.candidate_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )
