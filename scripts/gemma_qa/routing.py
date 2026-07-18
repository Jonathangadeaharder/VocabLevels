"""Job-agnostic model selection over the active strategy pool.

Any available active model can run dual review or adjudication. Selection
prefers free per-model slots (max 2 in-flight) so concurrent batches spread
load without role silos (no “Gemma is only adj”).
"""

from __future__ import annotations

import itertools
import threading
from typing import TypeVar

from pydantic import BaseModel

from .client import GemmaClient, GenerationResult
from .config import (
    ACTIVE_POOL,
    MODEL_ADJUDICATION,
    MODEL_IDS,
    is_model_available,
    model_free_slots,
    resolve_model_spec,
)
from .model_strategies import ROLE_ADJUDICATION, ROLE_DUAL, get_strategy

ResponseT = TypeVar("ResponseT", bound=BaseModel)

_lock = threading.Lock()
_pair_counter = itertools.count(0)
_pick_counter = itertools.count(0)


def _available_for_role(role: str) -> list[str]:
    keys: list[str] = []
    for key in ACTIVE_POOL:
        if not is_model_available(key):
            continue
        strategy = get_strategy(key)
        if strategy.supports_role(role):
            keys.append(key)
    return keys


def _prefer_free(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda key: (-model_free_slots(key), key))


def select_models_for_job(
    *,
    count: int,
    role: str = ROLE_DUAL,
    exclude: tuple[str, ...] | list[str] | None = None,
    batch_index: int = 0,
) -> tuple[str, ...]:
    """Pick ``count`` distinct models that can perform ``role``.

    Every active strategy supports dual and adjudication; this is the single
    entry point for “any model, any job”.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    excluded = set(exclude or ())
    pool = [key for key in _available_for_role(role) if key not in excluded]
    if not pool:
        pool = [key for key in ACTIVE_POOL if is_model_available(key)]
    if not pool:
        raise RuntimeError(f"no models available for role={role}")
    ranked = _prefer_free(pool)
    with _lock:
        base = next(_pick_counter)
    offset = (base + batch_index) % len(ranked)
    ordered = ranked[offset:] + ranked[:offset]
    chosen: list[str] = []
    for key in ordered:
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= count:
            break
    # If pool smaller than count, wrap (same model twice only as last resort).
    while len(chosen) < count:
        chosen.append(ordered[len(chosen) % len(ordered)])
    return tuple(chosen)


def select_dual_models(*, batch_index: int = 0) -> tuple[str, str]:
    """Two independent reviewers — any two free models from the active pool."""
    pair = select_models_for_job(
        count=2,
        role=ROLE_DUAL,
        batch_index=batch_index,
    )
    with _lock:
        next(_pair_counter)  # keep counter hot for tests / diagnostics
    return pair[0], pair[1]


def resolve_adjudication_model(
    client: object,
    *,
    prompt: str | None = None,
    exclude: tuple[str, ...] | list[str] | None = None,
    batch_index: int = 0,
) -> str:
    """Third vote — any model not used in the dual pair (same pool, same strategy)."""
    _ = client
    _ = prompt
    picked = select_models_for_job(
        count=1,
        role=ROLE_ADJUDICATION,
        exclude=exclude,
        batch_index=batch_index,
    )
    return picked[0]


def select_adjudication_model(quota: object | None = None) -> str:
    _ = quota
    return resolve_adjudication_model(None)


class UnifiedQaClient:
    """Routes every model key through the shared GemmaClient + strategies."""

    def __init__(
        self,
        *,
        gemma: GemmaClient,
        antigravity: object | None = None,
        quota: object | None = None,
    ) -> None:
        _ = antigravity
        _ = quota
        self._client = gemma

    def adjudication_model(self) -> str:
        return resolve_adjudication_model(self)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[ResponseT],
        max_output_tokens: int,
    ) -> GenerationResult[ResponseT]:
        key = resolve_model_spec(model).key
        if key not in MODEL_IDS and model not in MODEL_IDS:
            raise ValueError(f"unsupported model: {model}")
        return self._client.generate(
            model=key,
            prompt=prompt,
            response_model=response_model,
            max_output_tokens=max_output_tokens,
        )

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, object]:
        return self._client.parse_response(response_json, response_model)

    def close(self) -> None:
        self._client.close()
